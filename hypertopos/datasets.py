# Copyright (C) 2026 Karol Kędzia
# Licensed under the Business Source License 1.1 (the "License");
# you may not use this file except in compliance with the License.
# See LICENSE.md in the repository root for full terms.
"""Synthetic, labelled datasets generated on the fly (deterministic per seed).

``synthetic_aml`` produces a transaction table shaped like public AML
benchmarks (one row per payment, sender, receiver, amount, time, currency,
payment format, laundering label) over a heterogeneous population of retail and
business accounts, with laundering typologies injected and labelled. Nothing
is downloaded and no file ships with the package: the same seed gives the same
tables on every machine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

TYPOLOGIES = (
    "fan_out",
    "fan_in",
    "cycle",
    "scatter_gather",
    "structuring",
    "pass_through",
)
_START = 1_767_225_600  # 2026-01-01 00:00:00 UTC
_DAY = 86_400
_THRESHOLD = 10_000.0
_FORMATS = np.array(["ach", "wire", "card", "cheque", "cash"])
_CURRENCIES = np.array(["EUR", "USD", "GBP"])


@dataclass(frozen=True)
class SyntheticAML:
    accounts: pa.Table  # account, segment, is_laundering_account, typology
    transactions: pa.Table  # tx_id, ts, from_account, to_account, amount, currency,
    #                         payment_format, is_laundering, typology


def synthetic_aml(
    n_accounts: int = 5_000,
    n_days: int = 60,
    laundering_share: float = 0.01,
    seed: int = 0,
) -> SyntheticAML:
    """Generate accounts and transactions with labelled laundering patterns.

    Args:
        n_accounts: Accounts in the population (about 30 % business).
        n_days: Length of the period, starting 2026-01-01 UTC.
        laundering_share: Share of accounts that take part in a laundering
            pattern; the patterns cycle through ``TYPOLOGIES``.
        seed: Random seed; the same arguments give the same tables.

    Returns:
        ``accounts`` (``account``, ``segment``, ``is_laundering_account``,
        ``typology``) and ``transactions`` (``tx_id``, ``ts`` as a UTC
        timestamp, ``from_account``, ``to_account``, ``amount``, ``currency``,
        ``payment_format``, ``is_laundering``, ``typology``).
    """
    if n_accounts < 100:
        raise ValueError("n_accounts must be at least 100")
    rng = np.random.default_rng(seed)
    ids = np.array([f"A{i:06d}" for i in range(n_accounts)])
    business = rng.random(n_accounts) < 0.3
    segment = np.where(business, "business", "retail")

    normal = _normal_traffic(rng, ids, business, n_days)
    account_typology: dict[str, str]
    if laundering_share == 0:
        laundering, account_typology = {k: v[:0] for k, v in normal.items()}, {}
    else:
        laundering, account_typology = _laundering(rng, ids, business, n_days, laundering_share)

    columns = {k: np.concatenate([normal[k], laundering[k]]) for k in normal}
    order = np.argsort(columns["ts"], kind="stable")
    columns = {k: v[order] for k, v in columns.items()}
    n = len(order)
    transactions = pa.table(
        {
            "tx_id": pa.array([f"T{i:08d}" for i in range(n)]),
            "ts": pa.array(columns["ts"] * 1_000_000, pa.timestamp("us", tz="UTC")),
            "from_account": columns["from_account"],
            "to_account": columns["to_account"],
            "amount": np.round(columns["amount"], 2),
            "currency": columns["currency"],
            "payment_format": columns["payment_format"],
            "is_laundering": columns["is_laundering"],
            "typology": pa.array(columns["typology"], pa.string()),
        }
    )
    typology = [account_typology.get(str(a)) for a in ids]
    accounts = pa.table(
        {
            "account": ids,
            "segment": segment,
            "is_laundering_account": [t is not None for t in typology],
            "typology": pa.array(typology, pa.string()),
        }
    )
    return SyntheticAML(accounts=accounts, transactions=transactions)


def _normal_traffic(
    rng: np.random.Generator, ids: np.ndarray, business: np.ndarray, n_days: int
) -> dict[str, np.ndarray]:
    n = len(ids)
    # Each account has its own activity, amount scale and a few regular payees.
    rate = np.where(business, rng.gamma(4.0, 1.5, n), rng.gamma(2.0, 0.4, n))  # per day
    counts = rng.poisson(rate * n_days)
    scale = np.where(business, rng.lognormal(7.0, 0.8, n), rng.lognormal(4.0, 0.7, n))
    payees = rng.integers(0, n, (n, 4))
    sender = np.repeat(np.arange(n), counts)
    m = len(sender)
    regular = rng.random(m) < np.where(business[sender], 0.5, 0.7)
    receiver = np.where(regular, payees[sender, rng.integers(0, 4, m)], rng.integers(0, n, m))
    receiver = np.where(receiver == sender, (receiver + 1) % n, receiver)
    amount = scale[sender] * rng.lognormal(0.0, 0.6, m)
    day = rng.integers(0, n_days, m)
    weekend = (day + 3) % 7 >= 5  # 2026-01-01 is a Thursday
    keep = ~weekend | (rng.random(m) < np.where(business[sender], 0.15, 0.6))
    hour = np.clip(rng.normal(np.where(business[sender], 12.0, 15.0), 3.5, m), 0, 23.99)
    ts = _START + day * _DAY + (hour * 3600).astype(np.int64)
    return _rows(
        ids[sender][keep],
        ids[receiver][keep],
        amount[keep],
        ts[keep],
        rng,
        label=None,
    )


def _laundering(
    rng: np.random.Generator,
    ids: np.ndarray,
    business: np.ndarray,
    n_days: int,
    share: float,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    n = len(ids)
    budget = max(len(TYPOLOGIES), int(round(n * share)))
    pool = rng.permutation(n)
    used = 0
    parts: list[dict[str, np.ndarray]] = []
    roles: dict[str, str] = {}
    k = 0
    while used < budget:
        typology = TYPOLOGIES[k % len(TYPOLOGIES)]
        k += 1
        size = {"fan_out": 6, "fan_in": 6, "cycle": 4, "scatter_gather": 7}.get(typology, 1)
        size = min(size, budget - used)
        if size < 1:
            break
        members = ids[pool[used : used + size]]
        used += size
        for m in members:
            roles[str(m)] = typology
        start = _START + int(rng.integers(0, max(1, n_days - 3))) * _DAY
        parts.append(_pattern(rng, typology, members, ids, start))
    merged = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    return merged, roles


def _pattern(
    rng: np.random.Generator,
    typology: str,
    members: np.ndarray,
    ids: np.ndarray,
    start: int,
) -> dict[str, np.ndarray]:
    def outsiders(count: int) -> np.ndarray:
        return ids[rng.integers(0, len(ids), count)]

    def hours(count: int, span_h: float) -> np.ndarray:
        return start + np.sort((rng.random(count) * span_h * 3600).astype(np.int64))

    hub = members[0]
    if typology == "fan_out" or typology == "fan_in":
        # One hub spreads to (or collects from) many accounts within a day.
        others = np.concatenate([members[1:], outsiders(10)])
        amount = rng.uniform(2_000, 9_000, len(others))
        src, dst = (
            (np.full(len(others), hub), others)
            if typology == "fan_out"
            else (others, np.full(len(others), hub))
        )
        return _rows(src, dst, amount, hours(len(others), 20), rng, typology)
    if typology == "cycle":
        # Money goes round the members and back to the start, a little smaller each hop.
        src = members
        dst = np.roll(members, -1)
        hops = 25_000 * 0.97 ** np.arange(len(members), dtype=np.float64)
        return _rows(src, dst, hops, hours(len(members), 48), rng, typology)
    if typology == "scatter_gather":
        # A source scatters to intermediaries, which all forward to one sink.
        source, sink, middle = members[0], members[-1], members[1:-1]
        amount = rng.uniform(4_000, 8_000, len(middle))
        src = np.concatenate([np.full(len(middle), source), middle])
        dst = np.concatenate([middle, np.full(len(middle), sink)])
        amt = np.concatenate([amount, amount * 0.98])
        return _rows(src, dst, amt, hours(len(src), 36), rng, typology)
    if typology == "structuring":
        # Many deposits just under the reporting threshold over a few days.
        count = 14
        return _rows(
            np.full(count, hub),
            outsiders(count),
            rng.uniform(0.90, 0.995, count) * _THRESHOLD,
            hours(count, 72),
            rng,
            typology,
        )
    # pass_through: large amounts in, nearly the same out within hours.
    count = 5
    incoming = rng.uniform(30_000, 60_000, count)
    src = np.concatenate([outsiders(count), np.full(count, hub)])
    dst = np.concatenate([np.full(count, hub), outsiders(count)])
    ts = hours(count, 60)
    return _rows(
        src,
        dst,
        np.concatenate([incoming, incoming * 0.99]),
        np.concatenate([ts, ts + 3600]),
        rng,
        typology,
    )


def _rows(
    src: np.ndarray,
    dst: np.ndarray,
    amount: np.ndarray,
    ts: np.ndarray,
    rng: np.random.Generator,
    label: str | None,
) -> dict[str, np.ndarray]:
    m = len(src)
    return {
        "from_account": np.asarray(src, dtype=object),
        "to_account": np.asarray(dst, dtype=object),
        "amount": np.asarray(amount, dtype=np.float64),
        "ts": np.asarray(ts, dtype=np.int64),
        "currency": _CURRENCIES[rng.choice(3, m, p=[0.7, 0.2, 0.1])],
        "payment_format": (
            _FORMATS[rng.choice(5, m, p=[0.4, 0.25, 0.2, 0.1, 0.05])]
            if label is None
            else np.where(rng.random(m) < 0.6, "wire", "ach")
        ),
        "is_laundering": np.full(m, label is not None),
        "typology": np.full(m, label, dtype=object),
    }
