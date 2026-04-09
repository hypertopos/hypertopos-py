# Copyright (C) 2026 Karol Kędzia
# SPDX-License-Identifier: Apache-2.0
"""Prepare NYC Yellow Taxi trip data for GDS sphere build.

Reads raw parquet, cleans, adds computed columns, exports entity line tables.

Usage:
    .venv/Scripts/python benchmark/nyc-taxi/sphere/prepare_trips.py
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SRC = Path("benchmark/data/nyc-taxi")
OUT = Path("benchmark/nyc-taxi/sphere/data")


def prepare() -> pa.Table:
    """Main preparation — called by hypertopos build (Tier 3 script)."""
    return _prepare_trips()


def _prepare_trips() -> pa.Table:
    print("Reading raw parquet...")
    raw = pq.read_table(SRC / "yellow_tripdata_2019-01.parquet")
    print(f"  Raw rows: {len(raw):,}")

    # Drop null essentials
    for col in ("passenger_count", "trip_distance", "fare_amount", "PULocationID", "DOLocationID"):
        mask = pc.is_valid(raw[col])
        raw = raw.filter(mask)

    # Drop airport_fee (all null)
    if "airport_fee" in raw.schema.names:
        raw = raw.drop("airport_fee")

    # Cast types
    raw = raw.set_column(
        raw.schema.get_field_index("passenger_count"),
        "passenger_count",
        pc.cast(raw["passenger_count"], pa.int32()),
    )
    raw = raw.set_column(
        raw.schema.get_field_index("RatecodeID"),
        "RatecodeID",
        pc.cast(pc.if_else(pc.is_valid(raw["RatecodeID"]), raw["RatecodeID"], 1.0), pa.int32()),
    )

    # Filter: fare > 0, distance > 0
    raw = raw.filter(pc.and_(pc.greater(raw["fare_amount"], 0), pc.greater(raw["trip_distance"], 0)))

    # Compute duration in minutes
    pickup_us = pc.cast(raw["tpep_pickup_datetime"], pa.int64())
    dropoff_us = pc.cast(raw["tpep_dropoff_datetime"], pa.int64())
    duration_min = pc.divide(pc.subtract(dropoff_us, pickup_us), 60_000_000)
    raw = raw.append_column("trip_duration_min", pc.cast(duration_min, pa.float64()))

    # Filter: duration > 0 and <= 300 min
    raw = raw.filter(
        pc.and_(pc.greater(raw["trip_duration_min"], 0), pc.less_equal(raw["trip_duration_min"], 300))
    )

    # Speed mph (capped at 100)
    hours = pc.divide(raw["trip_duration_min"], 60.0)
    speed = pc.divide(raw["trip_distance"], hours)
    speed = pc.if_else(pc.greater(speed, 100.0), 100.0, speed)
    raw = raw.append_column("speed_mph", speed)

    # Tip percentage
    tip_pct = pc.if_else(
        pc.greater(raw["fare_amount"], 0),
        pc.divide(raw["tip_amount"], raw["fare_amount"]),
        0.0,
    )
    raw = raw.append_column("tip_pct", tip_pct)

    # Cast to timestamp without tz (Windows tzdata workaround)
    pickup_no_tz = raw["tpep_pickup_datetime"].cast(pa.timestamp("us"))
    # Pickup hour and day of week
    raw = raw.append_column("pickup_hour", pc.cast(pc.hour(pickup_no_tz), pa.int32()))
    raw = raw.append_column("pickup_dow", pc.cast(pc.day_of_week(pickup_no_tz), pa.int32()))
    # pickup_date as string YYYY-MM-DD (avoid strftime tz issues)
    year = pc.cast(pc.year(pickup_no_tz), pa.string())
    month = pc.utf8_lpad(pc.cast(pc.month(pickup_no_tz), pa.string()), 2, "0")
    day = pc.utf8_lpad(pc.cast(pc.day(pickup_no_tz), pa.string()), 2, "0")
    pickup_date = pc.binary_join_element_wise(year, month, day, "-")
    raw = raw.append_column("pickup_date", pickup_date)

    # Primary key
    keys = pa.array([f"T-{i:07d}" for i in range(len(raw))], type=pa.string())
    raw = raw.append_column("primary_key", keys)

    # String FK columns for relations
    raw = raw.append_column("pu_zone", pc.cast(raw["PULocationID"], pa.string()))
    raw = raw.append_column("do_zone", pc.cast(raw["DOLocationID"], pa.string()))
    raw = raw.append_column("vendor", pc.cast(raw["VendorID"], pa.string()))
    raw = raw.append_column("rate_code", pc.cast(raw["RatecodeID"], pa.string()))
    raw = raw.append_column("pay_type", pc.cast(raw["payment_type"], pa.string()))

    print(f"  Clean rows: {len(raw):,}")
    return raw


def _prepare_zones() -> pa.Table:
    import csv
    rows = []
    with open(SRC / "taxi_zone_lookup.csv") as f:
        for r in csv.DictReader(f):
            rows.append({
                "primary_key": r["LocationID"],
                "borough": r["Borough"],
                "zone_name": r["Zone"],
                "service_zone": r["service_zone"],
            })
    return pa.Table.from_pylist(rows)


def _prepare_vendors() -> pa.Table:
    return pa.Table.from_pylist([
        {"primary_key": "1", "name": "Creative Mobile Technologies"},
        {"primary_key": "2", "name": "VeriFone Inc"},
    ])


def _prepare_rate_codes() -> pa.Table:
    return pa.Table.from_pylist([
        {"primary_key": "1", "name": "Standard rate"},
        {"primary_key": "2", "name": "JFK"},
        {"primary_key": "3", "name": "Newark"},
        {"primary_key": "4", "name": "Nassau/Westchester"},
        {"primary_key": "5", "name": "Negotiated fare"},
        {"primary_key": "6", "name": "Group ride"},
    ])


def _prepare_payment_types() -> pa.Table:
    return pa.Table.from_pylist([
        {"primary_key": "1", "name": "Credit card"},
        {"primary_key": "2", "name": "Cash"},
        {"primary_key": "3", "name": "No charge"},
        {"primary_key": "4", "name": "Dispute"},
        {"primary_key": "5", "name": "Unknown"},
        {"primary_key": "6", "name": "Voided trip"},
    ])


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Trips (event line)
    trips = _prepare_trips()
    pq.write_table(trips, OUT / "trips.parquet")
    print(f"  Wrote trips.parquet: {len(trips):,} rows")

    # Zones
    zones = _prepare_zones()
    pq.write_table(zones, OUT / "zones.parquet")
    print(f"  Wrote zones.parquet: {len(zones)} rows")

    # Vendors
    vendors = _prepare_vendors()
    pq.write_table(vendors, OUT / "vendors.parquet")
    print(f"  Wrote vendors.parquet: {len(vendors)} rows")

    # Rate codes
    rate_codes = _prepare_rate_codes()
    pq.write_table(rate_codes, OUT / "rate_codes.parquet")
    print(f"  Wrote rate_codes.parquet: {len(rate_codes)} rows")

    # Payment types
    payment_types = _prepare_payment_types()
    pq.write_table(payment_types, OUT / "payment_types.parquet")
    print(f"  Wrote payment_types.parquet: {len(payment_types)} rows")

    # Stats
    print("\n=== Summary ===")
    print(f"Total clean trips: {len(trips):,}")
    fare = trips["fare_amount"]
    dist = trips["trip_distance"]
    dur = trips["trip_duration_min"]
    speed = trips["speed_mph"]
    print(f"Avg fare: ${pc.mean(fare).as_py():.2f}")
    print(f"Avg distance: {pc.mean(dist).as_py():.2f} mi")
    print(f"Avg duration: {pc.mean(dur).as_py():.1f} min")
    print(f"Avg speed: {pc.mean(speed).as_py():.1f} mph")

    # Anomaly candidates
    fast = pc.sum(pc.greater(speed, 80.0)).as_py()
    expensive = pc.sum(pc.greater(fare, 200.0)).as_py()
    short = pc.sum(pc.less(dur, 1.0)).as_py()
    print(f"\nAnomaly candidates:")
    print(f"  Speed > 80 mph: {fast:,}")
    print(f"  Fare > $200: {expensive:,}")
    print(f"  Duration < 1 min: {short:,}")


if __name__ == "__main__":
    main()
