# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
"""Prepare script — loan-bearing accounts only (682 entities).

Reuses prepare_accounts and filters to has_loan == 'yes'.
Purpose: isolated population for loan stress pattern where mu/sigma
are calibrated to loan-holders, not the full 4,500-account population.
"""
from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from prepare_accounts import prepare as prepare_all_accounts


def prepare() -> pa.Table:
    full = prepare_all_accounts()
    mask = pc.equal(full.column("has_loan"), pa.scalar("yes"))
    return full.filter(mask)
