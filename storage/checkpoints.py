"""
Lightweight JSON checkpointing for LangGraph pipeline state.

This is a local-file substitute for the MongoDB checkpointing described in the
PRD. Each node appends a checkpoint entry after it produces its state update.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


DEFAULT_CHECKPOINT_DIR = Path("data/checkpoints")


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        try:
            return _jsonable(value.model_dump(mode="json"))
        except TypeError:
            return _jsonable(value.model_dump())
        except AttributeError:
            return _jsonable(value.dict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def save_pipeline_checkpoint(
    *,
    checkpoint_path: str,
    node_name: str,
    current_state: dict[str, Any],
    update: dict[str, Any],
) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    merged_state = {**current_state, **update}
    checkpoint = {
        "node": node_name,
        "pipeline_status": merged_state.get("pipeline_status"),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "state": _jsonable(merged_state),
    }

    if path.exists():
        payload = json.loads(path.read_text())
    else:
        payload = {"checkpoints": []}

    payload["latest"] = checkpoint
    payload.setdefault("checkpoints", []).append(checkpoint)
    path.write_text(json.dumps(payload, indent=2))
