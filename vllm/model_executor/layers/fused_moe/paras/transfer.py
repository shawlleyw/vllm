# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preallocated NCCL permutation and CUDA IPC expert transports."""

import ctypes
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from .storage import ExpertArena


def load_peer_kernel():
    from torch.utils.cpp_extension import load

    source = Path(__file__).parent / "csrc"
    build = Path(sys.prefix) / "var/cache/paras-peer-kernel"
    build.mkdir(parents=True, exist_ok=True)
    return load(
        name="vllm_paras_peer",
        sources=[str(source / "binding.cpp"), str(source / "peer_access.cu")],
        build_directory=str(build),
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        verbose=False,
    )


class IpcMapping:
    """Keep IPC mappings alive for the lifetime of the managed arena."""

    class Handle(ctypes.Structure):
        _fields_ = [("reserved", ctypes.c_ubyte * 64)]

    def __init__(self, buffer, cpu_group):
        # Use the already loaded CUDA runtime, including CUDA 13's versioned name.
        from vllm.utils.system_utils import find_loaded_library

        library = find_loaded_library("libcudart")
        if library is None:
            raise RuntimeError("CUDA runtime is not loaded")
        self.cuda = ctypes.CDLL(library)
        self.cuda.cudaIpcGetMemHandle.argtypes = [
            ctypes.POINTER(self.Handle),
            ctypes.c_void_p,
        ]
        self.cuda.cudaIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            self.Handle,
            ctypes.c_uint,
        ]
        self.cuda.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        driver = ctypes.CDLL("libcuda.so.1")
        driver.cuMemGetAddressRange_v2.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_uint64,
        ]
        base, size = ctypes.c_uint64(), ctypes.c_size_t()
        self.check(
            driver.cuMemGetAddressRange_v2(
                ctypes.byref(base), ctypes.byref(size), buffer.data_ptr()
            )
        )
        handle = self.Handle()
        self.check(self.cuda.cudaIpcGetMemHandle(ctypes.byref(handle), base.value))
        record = (bytes(handle.reserved), buffer.data_ptr() - base.value)
        records: list[tuple[bytes, int] | None] = [None] * dist.get_world_size(
            cpu_group
        )
        dist.all_gather_object(records, record, group=cpu_group)
        self.opened = []
        self.addresses = []
        try:
            for rank, item in enumerate(records):
                assert item is not None
                raw, offset = item
                if rank == dist.get_rank(cpu_group):
                    self.addresses.append(buffer.data_ptr())
                    continue
                remote = self.Handle.from_buffer_copy(raw)
                pointer = ctypes.c_void_p()
                self.check(
                    self.cuda.cudaIpcOpenMemHandle(ctypes.byref(pointer), remote, 1)
                )
                if pointer.value is None:
                    raise RuntimeError("CUDA IPC returned a null mapping")
                self.opened.append(pointer)
                self.addresses.append(pointer.value + offset)
        except Exception:
            self.close()
            raise

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"CUDA IPC operation failed with error {code}")

    def close(self):
        for pointer in self.opened:
            self.check(self.cuda.cudaIpcCloseMemHandle(pointer))
        self.opened.clear()


class WeightTransfer:
    def __init__(self, arena: ExpertArena, method: str, cpu_group, device_group):
        if arena.buffer is None:
            raise ValueError("Transfer requires a materialized arena")
        self.buffer = arena.buffer
        self.arena = arena
        self.method = method
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.rank = dist.get_rank(cpu_group)
        if dist.get_world_size(cpu_group) != arena.layout.expert_tp_size:
            raise ValueError("Transfer group does not match expert TP layout")
        self.stream = torch.cuda.Stream()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.fence = torch.zeros(1, device=self.buffer.device, dtype=torch.int32)
        self.ipc = None
        if method == "peer_access":
            self.kernel = load_peer_kernel()
            self.ipc = IpcMapping(arena.buffer, cpu_group)
            self.peer_pointers = torch.tensor(
                self.ipc.addresses, dtype=torch.int64, device=self.buffer.device
            )
        elif method != "nccl":
            raise ValueError(method)
        # Initialize communicators, synchronization storage, and timing events now.
        with torch.cuda.stream(self.stream):
            self.start.record()
            dist.all_reduce(self.fence, group=device_group)
            self.end.record()
        self.stream.synchronize()

    def move(self, target: str) -> float:
        """Caller owns quiescence and treats every failure here as destructive."""
        order = self.arena.transfer_order(target)
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.start.record()
            for layer in order:
                if self.method == "nccl":
                    self._nccl_layer(layer, target)
                else:
                    self._peer_layer(layer, target)
                # Remote stores must finish on EVERY rank before the next layer
                # reuses an overlapping slab. The collective depends on this
                # stream's kernels, and following kernels depend on the collective.
                dist.all_reduce(self.fence, group=self.device_group)
            self.end.record()
        self.stream.synchronize()
        torch.cuda.current_stream().wait_stream(self.stream)
        return self.start.elapsed_time(self.end)

    def _nccl_layer(self, layer: int, target: str):
        a, layout = self.arena, self.arena.layout
        e, t = layout.experts // layout.ep_size, layout.expert_tp_size
        for name, (shape, _) in layout.tensors("tp").items():
            _, n, k = shape
            tail = (2, n // 2 * k) if name.startswith("w13") else (n, k)
            # NCCL and strided copies must preserve FP8 storage bits exactly.
            ep = a.view(f"ep.{layer}.{name}").view(torch.uint8)
            tp = a.view(f"tp.{layer}.{name}").view(torch.uint8)
            staging = a.view(f"scratch.{name}").view(torch.uint8)
            tail = (tail[0], tail[1] * a.entries[f"ep.{layer}.{name}"].dtype.itemsize)
            if target == "tp":
                staging.view(t, e, *tail).copy_(
                    ep.view(e, tail[0], t, tail[1]).permute(2, 0, 1, 3)
                )
                dist.all_to_all_single(
                    tp.view(-1), staging.view(-1), group=self.device_group
                )
            else:
                dist.all_to_all_single(
                    staging.view(-1), tp.view(-1), group=self.device_group
                )
                ep.view(e, tail[0], t, tail[1]).copy_(
                    staging.view(t, e, *tail).permute(1, 2, 0, 3)
                )

    def _peer_layer(self, layer: int, target: str):
        a, layout = self.arena, self.arena.layout
        source = "ep" if target == "tp" else "tp"
        suffix = "v2" if target == "tp" else "ep"
        for name, (shape, dtype) in layout.tensors("tp").items():
            _, n, k = shape
            args = [
                self.buffer.data_ptr(),
                self.peer_pointers,
                a.entries[f"{source}.{layer}.{name}"].offset,
                a.entries[f"{target}.{layer}.{name}"].offset,
                self.rank,
                layout.expert_tp_size,
                layout.experts // layout.ep_size,
            ]
            if name.startswith("w13"):
                kernel_name = "w13_" + suffix
                args.extend([n // 2 * k, 2, dtype.itemsize])
            else:
                kernel_name = "w2_" + suffix
                args.extend(
                    [n, k * layout.expert_tp_size * dtype.itemsize, k * dtype.itemsize]
                )
            getattr(self.kernel, kernel_name)(*args, self.stream.cuda_stream)

    def close(self):
        self.stream.synchronize()
        dist.barrier(group=self.cpu_group)
        if self.ipc is not None:
            self.ipc.close()
