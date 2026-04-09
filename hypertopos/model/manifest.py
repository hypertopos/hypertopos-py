# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class AliasRef:
    alias_id: str
    version: int
    local_name: str


@dataclass
class Manifest:
    manifest_id: str
    agent_id: str
    snapshot_time: datetime
    status: Literal["active", "expired", "revoked"]
    line_versions: dict[str, int]
    pattern_versions: dict[str, int]
    alias_versions: dict[str, AliasRef] = field(default_factory=dict)

    def line_version(self, line_id: str) -> int | None:
        return self.line_versions.get(line_id)

    def pattern_version(self, pattern_id: str) -> int | None:
        return self.pattern_versions.get(pattern_id)


@dataclass
class Contract:
    manifest_id: str
    pattern_ids: list[str]

    def has_pattern(self, pattern_id: str) -> bool:
        return pattern_id in self.pattern_ids
