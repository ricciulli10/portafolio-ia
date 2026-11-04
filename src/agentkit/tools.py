"""Typed tools: a pydantic model is the schema, the validator and the docs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


class ToolError(Exception):
    """An expected failure the model should read and recover from."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[Any], Any]
    # Decided per call, in code. The model cannot talk its way past this.
    needs_approval: Callable[[Any], bool] = lambda _: False

    def schema(self) -> dict:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
            "strict": True,
        }
