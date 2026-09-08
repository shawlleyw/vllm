# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/paras")


class SwitchRequest(BaseModel):
    target: Literal["ep", "tp"]


@router.get("/status")
async def status(request: Request):
    return await request.app.state.engine_client.paras_status()


@router.post("/switch")
async def switch(body: SwitchRequest, request: Request):
    try:
        return await request.app.state.engine_client.paras_switch(body.target)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
