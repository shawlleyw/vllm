# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create reduced-depth configs and canonical synthetic checkpoint weights.

All parallel configurations load the same global tensors through the normal
checkpoint loader. Independent per-rank dummy initialization would not describe
the same logical experts in EP and TP.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import regex as re
import torch
from safetensors.torch import save_file

MODELS = ("GLM-4.7-Flash", "GLM-4.5-Air", "Qwen3.6-35B-A3B")


def build(source, destination, layers, shard_bytes):
    if destination.exists():
        raise FileExistsError(f"Use a fresh destination: {destination}")
    destination.mkdir(parents=True)
    config = json.loads((source / "config.json").read_text())
    text_config = config.get("text_config", config)
    original_layers = text_config["num_hidden_layers"]
    if not 1 <= layers < original_layers:
        raise ValueError(f"Invalid reduced depth {layers} for {source.name}")
    text_config["num_hidden_layers"] = layers
    if "layer_types" in text_config:
        text_config["layer_types"] = text_config["layer_types"][:layers]
    for key in ("num_nextn_predict_layers", "mtp_num_hidden_layers"):
        if key in text_config:
            text_config[key] = 0
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    # Reuse tokenizer/processor assets, excluding the original config and weights.
    for path in source.iterdir():
        if (
            path.is_file()
            and path.name != "config.json"
            and path.suffix != ".safetensors"
            and not path.name.endswith(".index.json")
        ):
            (destination / path.name).symlink_to(path.resolve())
    index = json.loads((source / "model.safetensors.index.json").read_text())
    headers = {}
    for shard in sorted(set(index["weight_map"].values())):
        with (source / shard).open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            headers.update(json.loads(stream.read(size)))
    pending, weight_map = {}, {}
    pending_bytes = total_bytes = count = 0
    shard_number = 0

    def flush():
        nonlocal pending, pending_bytes, shard_number
        if not pending:
            return
        shard_number += 1
        name = f"model-{shard_number:05d}.safetensors"
        save_file(pending, str(destination / name), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(pending, name))
        pending, pending_bytes = {}, 0

    for name in sorted(index["weight_map"]):
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if (
            (match and int(match[1]) >= layers)
            or "visual." in name
            or name.startswith("mtp.")
            or ".mtp." in name
        ):
            continue
        spec = headers[name]
        dtype = {"BF16": torch.bfloat16, "F32": torch.float32}[spec["dtype"]]
        tensor = torch.empty(spec["shape"], dtype=dtype)
        if "norm" in name and name.endswith("weight"):
            # Qwen uses (1 + weight), except its gated linear-attention norm.
            offset_norm = (
                text_config["model_type"] == "qwen3_5_moe_text"
                and ".linear_attn.norm." not in name
            )
            tensor.fill_(0 if offset_norm else 1)
        elif name.endswith((".bias", ".dt_bias", ".A_log")):
            tensor.zero_()
        else:
            seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
            generator = torch.Generator().manual_seed(seed)
            tensor.normal_(mean=0, std=0.02, generator=generator)
        nbytes = tensor.numel() * tensor.element_size()
        if pending_bytes + nbytes > shard_bytes:
            flush()
        pending[name] = tensor
        pending_bytes += nbytes
        total_bytes += nbytes
        count += 1
    flush()
    (destination / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
            indent=2,
        )
        + "\n"
    )
    manifest = dict(
        synthetic=True,
        source=str(source),
        original_layers=original_layers,
        layers=layers,
        tensors=count,
        shards=shard_number,
        bytes=total_bytes,
        seed="sha256(parameter name), first 8 bytes little endian",
        weights=(
            "normal(0, 0.02); norm effective scale=1 "
            "(offset norms=0); .bias/dt_bias/A_log=0"
        ),
    )
    (destination / "synthetic.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"model": source.name, **manifest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/data/shaoyuw/models"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--glm-layers", type=int, default=5)
    parser.add_argument("--qwen-layers", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    for model in args.models:
        build(
            args.source / model,
            args.output / model,
            args.qwen_layers if model.startswith("Qwen") else args.glm_layers,
            512 * 1024**2,
        )
