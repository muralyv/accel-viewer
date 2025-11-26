import logging
import os
import sys
import math
import re
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Defaults can be overridden via environment variables for flexibility.
DATA_DIR = Path(
    os.getenv("DATA_DIR", Path.cwd() / "data")
).resolve()
PARQUET_DIR = Path(os.getenv("PARQUET_DIR", DATA_DIR)).resolve()  # Where converted parquet files will live
AUTO_PARQUET = os.getenv("AUTO_PARQUET", "false").lower() in {"1", "true", "yes"}
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_SIGNALS: List[str] = []  # defaults disabled; user must supply
DEFAULT_SAMPLE_RATE = float(os.getenv("DEFAULT_SAMPLE_RATE", "30"))
ANNOTATION_FILE = Path(os.getenv("ANNOTATION_FILE", DATA_DIR / "annotations.csv")).resolve()

logger = logging.getLogger("accelerometer_viewer")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

ENV_SIGNALS = [
    s
    for s in (
        x.strip() for x in os.getenv("SIGNALS", "").replace(",", " ").split()
    )
    if s
]
FORCED_SIGNALS = ENV_SIGNALS
if FORCED_SIGNALS:
    logger.info("Forcing signals from CLI/env: %s", FORCED_SIGNALS)


class DatasetMeta(BaseModel):
    id: str
    file_name: str
    columns: List[str]
    start_time: Optional[str]
    sample_rate_hz: Optional[float]
    source: str  # csv or parquet


class WindowResponse(BaseModel):
    dataset: str
    start_time: str
    end_time: str
    window_seconds: float
    signals: List[str]
    sample_rate_hz: Optional[float]
    rows_returned: int
    time: List[str]
    series: Dict[str, List[float]]
    aggregated: bool = False
    bin_size: Optional[float] = None
    bin_unit: Optional[str] = None
    aggregate: Optional[str] = None


class Annotation(BaseModel):
    dataset: str
    start_time: str
    end_time: str
    label: Optional[str] = None
    file_name: Optional[str] = None


def _parse_time(value: str) -> datetime:
    """Parse ISO8601 (with optional trailing Z) into naive datetime (no timezone)."""
    try:
        cleaned = value.replace("Z", "")
        return datetime.fromisoformat(cleaned)
    except Exception as exc:  # pragma: no cover - defensive path
        raise HTTPException(
            status_code=400, detail=f"Invalid time format '{value}': {exc}"
        ) from exc


def _fmt(dt: datetime) -> str:
    """Format datetime without tz and without fractional seconds."""
    if dt.tzinfo:
        dt = dt.replace(tzinfo=None)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _flatten_signals(signals: Optional[Sequence[str]]) -> List[str]:
    """Split comma/space-delimited items into a flat list, trimming empties."""
    flat: List[str] = []
    for item in signals or []:
        for part in re.split(r"[,\s]+", item):
            if part:
                flat.append(part.strip())
    return flat


def _escape_path(p: Path) -> str:
    """Escape single quotes for DuckDB's string literal paths."""
    return str(p.as_posix()).replace("'", "''")


class Dataset:
    """Represents a single CSV/Parquet accelerometer file backed by DuckDB queries."""

    # Derived signals that can be computed on the fly.
    DERIVED_SIGNALS: Dict[str, Set[str]] = {}

    def __init__(self, path: Path):
        self.path = path
        self.id = path.stem.replace(" ", "_")
        self.parquet_path = PARQUET_DIR / f"{path.stem}.parquet"
        self.columns = self._read_columns()
        self.sample_rate_hz = self._estimate_sample_rate()
        self.start_time = self._read_start_time()

    @property
    def available_signals(self) -> List[str]:
        return list(self.columns)

    def _read_columns(self) -> List[str]:
        df = pd.read_csv(self.path, nrows=1)
        cols: List[str] = []
        for c in df.columns:
            normalized = c.strip()
            if not normalized or normalized.lower() == "time":
                continue
            if normalized not in cols:
                cols.append(normalized)
        return cols

    def _estimate_sample_rate(self, nrows: int = 400) -> Optional[float]:
        df = pd.read_csv(self.path, nrows=nrows)
        if "time" not in df.columns or len(df) < 2:
            return DEFAULT_SAMPLE_RATE
        times = pd.to_datetime(df["time"]).dt.tz_localize(None)
        deltas = times.diff().dt.total_seconds().dropna()
        median_delta = deltas.median()
        if not median_delta or median_delta <= 0:
            return DEFAULT_SAMPLE_RATE
        return round(1.0 / median_delta, 3)

    def _read_start_time(self) -> Optional[str]:
        df = pd.read_csv(self.path, nrows=1)
        if "time" not in df.columns or df.empty:
            return None
        ts = pd.to_datetime(df.loc[0, "time"]).tz_localize(None)
        return _fmt(ts.to_pydatetime())

    def _ensure_parquet(self) -> bool:
        if self.parquet_path.exists():
            return True
        if not AUTO_PARQUET:
            return False
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Converting %s -> %s", self.path.name, self.parquet_path.name)
        conn = duckdb.connect(database=":memory:")
        conn.execute("SET TimeZone='UTC';")
        conn.execute(
            f"""
            COPY (
                SELECT *
                FROM read_csv_auto('{_escape_path(self.path)}',
                                   timestampformat='%Y-%m-%dT%H:%M:%S%Z')
            ) TO '{_escape_path(self.parquet_path)}' (FORMAT PARQUET);
            """
        )
        conn.close()
        return True

    def _source_clause(self) -> str:
        """Return the DuckDB FROM clause to read this dataset."""
        if self.parquet_path.exists() or self._ensure_parquet():
            return f"read_parquet('{_escape_path(self.parquet_path)}')"
        return (
            "read_csv_auto("
            f"'{_escape_path(self.path)}', "
            "timestampformat='%Y-%m-%dT%H:%M:%S%Z'"
            ")"
        )

    def _normalize_signals(self, requested: Sequence[str]) -> List[str]:
        if requested:
            source = list(requested)
        elif FORCED_SIGNALS:
            source = list(FORCED_SIGNALS)
        else:
            source = list(self.available_signals)

        if not source:
            raise HTTPException(status_code=400, detail="No signals available in this dataset.")

        col_map = {c.lower(): c for c in self.columns}
        resolved: List[str] = []
        missing: List[str] = []
        for s in source:
            key = s.strip().lower()
            if key in col_map:
                resolved.append(col_map[key])
            else:
                missing.append(s)
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Signals not found in dataset {self.id}: {', '.join(missing)}. Available: {', '.join(self.available_signals)}",
            )
        return resolved

    def window(
        self,
        start: datetime,
        duration_seconds: float,
        signals: Sequence[str],
        bin_size: Optional[float],
        bin_unit: Optional[str],
        aggregate: str,
    ) -> WindowResponse:
        duration_seconds = float(duration_seconds)
        if duration_seconds <= 0:
            raise HTTPException(status_code=400, detail="duration_seconds must be > 0")
        requested_signals = self._normalize_signals(signals)

        derived_needed = {s for s in requested_signals if s in self.DERIVED_SIGNALS}
        base_signals: List[str] = [s for s in requested_signals if s in self.columns]
        for derived in derived_needed:
            base_signals.update(self.DERIVED_SIGNALS[derived])

        # Preserve the CSV column order for base signals
        ordered_available = [c for c in self.columns if c in base_signals]
        # Add any derived signals (they come after base signals)
        select_signals = ordered_available + [s for s in requested_signals if s in self.DERIVED_SIGNALS]
        source = self._source_clause()

        conn = duckdb.connect(database=":memory:", config={"threads": os.cpu_count()})
        conn.execute("SET TimeZone='UTC';")

        aggregate = aggregate.lower() if aggregate else "mean"
        do_agg = bin_size is not None and bin_size > 0

        unit_seconds_map = {
            "s": 1,
            "sec": 1,
            "secs": 1,
            "second": 1,
            "seconds": 1,
            "m": 60,
            "min": 60,
            "mins": 60,
            "minute": 60,
            "minutes": 60,
            "h": 3600,
            "hr": 3600,
            "hrs": 3600,
            "hour": 3600,
            "hours": 3600,
            "d": 86400,
            "day": 86400,
            "days": 86400,
        }

        # Interpret window based on aggregation mode:
        # - do_agg: duration_seconds is expressed in bin_unit units (e.g., 10 minutes)
        # - raw: duration_seconds is a raw sample count
        window_units = duration_seconds

        # Determine how many rows to pull based on the requested window.
        if do_agg:
            if not bin_unit or bin_unit.lower() not in unit_seconds_map:
                raise HTTPException(
                    status_code=400,
                    detail="bin_unit must be one of: s, sec, second, m, min, minute, h, hour, d, day",
                )
            if aggregate not in {"mean", "avg"}:
                raise HTTPException(status_code=400, detail="Only aggregate=mean is supported currently")
            if not self.sample_rate_hz:
                raise HTTPException(status_code=400, detail="Sample rate unavailable; cannot aggregate")
            duration_units = window_units  # already in bin_unit units
            bin_seconds = bin_size * unit_seconds_map[bin_unit.lower()]
            bins_needed = max(1, math.ceil(duration_units / bin_size))
            samples_per_bin = max(1, math.ceil(self.sample_rate_hz * bin_seconds))
            limit_rows = bins_needed * samples_per_bin
            window_span_seconds = duration_units * unit_seconds_map[bin_unit.lower()]
        else:
            # window_units is treated as a raw sample count
            limit_rows = max(1, int(math.ceil(window_units)))
            sr = self.sample_rate_hz or DEFAULT_SAMPLE_RATE
            window_span_seconds = limit_rows / sr

        try:
            select_cols = ["time"] + select_signals
            query = f"""
                SELECT {', '.join(select_cols)}
                FROM {source}
                WHERE time >= ?
                ORDER BY time
                LIMIT {limit_rows}
            """
            df = conn.execute(query, [start]).fetch_df()
        except duckdb.IOException as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Data file unavailable: {exc}. Close any app using the file (Excel) or convert to Parquet.",
            )
        finally:
            conn.close()

        if df.empty:
            return WindowResponse(
                dataset=self.id,
                start_time=_fmt(start),
                end_time=_fmt(start + timedelta(seconds=window_span_seconds)),
                window_seconds=duration_seconds,
                signals=list(requested_signals),
                sample_rate_hz=self.sample_rate_hz,
                rows_returned=0,
                time=[],
                series={s: [] for s in requested_signals},
                aggregated=do_agg,
                bin_size=bin_size if do_agg else None,
                bin_unit=bin_unit.lower() if do_agg and bin_unit else None,
                aggregate=aggregate if do_agg else None,
            )

        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
        df.columns = ["time" if c.lower() == "time" else c.strip() for c in df.columns]

        # Aggregation: group consecutive samples into bins based on sample_rate_hz and bin_seconds
        if do_agg:
            df = df.reset_index(drop=True)
            samples_per_bin = max(1, math.ceil(self.sample_rate_hz * bin_size * unit_seconds_map[bin_unit.lower()]))
            df["bin"] = df.index // samples_per_bin
            agg_dict = {sig: "mean" for sig in select_signals}
            agg_values = df.groupby("bin").agg(agg_dict)
            first_times = df.groupby("bin")["time"].first().values
            agg_values.insert(0, "time", first_times)
            df = agg_values.reset_index(drop=True)

        ordered_cols = ["time"] + list(requested_signals)
        df = df[ordered_cols]

        time_strings = [_fmt(t.to_pydatetime()) for t in df["time"]]
        series = {col: df[col].astype(float).tolist() for col in requested_signals}

        end_time = start + timedelta(seconds=window_span_seconds)

        return WindowResponse(
            dataset=self.id,
            start_time=_fmt(start),
            end_time=_fmt(end_time),
            window_seconds=duration_seconds,
            signals=list(requested_signals),
            sample_rate_hz=self.sample_rate_hz,
            rows_returned=len(df),
            time=time_strings,
            series=series,
            aggregated=do_agg,
            bin_size=bin_size if do_agg else None,
            bin_unit=bin_unit.lower() if do_agg and bin_unit else None,
            aggregate=aggregate if do_agg else None,
        )


class Catalog:
    def __init__(self, data_dir: Path):
        if not data_dir.exists():
            raise RuntimeError(f"Data directory '{data_dir}' does not exist")
        self.data_dir = data_dir
        self._datasets: Dict[str, Dataset] = {}
        self._load()

    def _load(self) -> None:
        for path in sorted(self.data_dir.glob("*.csv")):
            ds = Dataset(path)
            if ds.id in self._datasets:
                logger.warning("Duplicate dataset id %s, keeping the first", ds.id)
                continue
            self._datasets[ds.id] = ds
        logger.info("Catalog loaded %d datasets from %s", len(self._datasets), self.data_dir)

    def list(self) -> List[DatasetMeta]:
        return [
            DatasetMeta(
                id=ds.id,
                file_name=ds.path.name,
                columns=list(ds.available_signals),
                start_time=ds.start_time,
                sample_rate_hz=ds.sample_rate_hz,
                source="parquet" if ds.parquet_path.exists() else "csv",
            )
            for ds in self._datasets.values()
        ]

    def get(self, dataset_id: str) -> Dataset:
        if dataset_id not in self._datasets:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
        return self._datasets[dataset_id]


app = FastAPI(title="Accelerometer window API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

catalog = Catalog(DATA_DIR)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/datasets", response_model=List[DatasetMeta])
def list_datasets() -> List[DatasetMeta]:
    return catalog.list()


@app.get("/window", response_model=WindowResponse)
def window(
    dataset: str = Query(..., description="Dataset id from /datasets"),
    start_time: Optional[str] = Query(
        None, description="ISO8601 start time (e.g., 2021-05-17T00:00:00Z)"
    ),
    start_epoch_ms: Optional[float] = Query(
        None, description="Unix epoch milliseconds, alternative to start_time"
    ),
    duration_seconds: float = Query(
        10.0, gt=0, description="Window length in seconds (default 10)"
    ),
    signals: List[str] = Query(
        None,
        description="Signals to return (repeat param or comma separated). If omitted, CLI/env SIGNALS is used or all columns.",
    ),
    bin_size: Optional[float] = Query(
        None, gt=0, description="Aggregate window into bins of this size (e.g., 1)"
    ),
    bin_unit: Optional[str] = Query(
        "s", description="Unit for bin_size: s, m, h, d"
    ),
    aggregate: str = Query(
        "mean", description="Aggregation function (currently only mean)"
    ),
) -> WindowResponse:
    ds = catalog.get(dataset)

    if start_epoch_ms is not None:
        start_dt = datetime.fromtimestamp(start_epoch_ms / 1000.0)
    elif start_time:
        start_dt = _parse_time(start_time)
    elif ds.start_time:
        start_dt = _parse_time(ds.start_time)
    else:
        raise HTTPException(status_code=400, detail="Provide start_time or start_epoch_ms")

    flat_signals = _flatten_signals(signals)

    return ds.window(
        start=start_dt,
        duration_seconds=duration_seconds,
        signals=flat_signals,
        bin_size=bin_size,
        bin_unit=bin_unit,
        aggregate=aggregate,
    )


@app.post("/annotate")
def save_annotation(ann: Annotation) -> Dict[str, str]:
    """Append an annotation row to the annotation CSV."""
    if ann.file_name:
        # Force CSV extension and ignore directory components from user input for safety.
        name = Path(ann.file_name).name
        if not name.lower().endswith(".csv"):
            name = f"{name}.csv"
        target_file = (DATA_DIR / name).resolve()
    else:
        target_file = ANNOTATION_FILE
    target_file.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "dataset": ann.dataset,
        "start_time": ann.start_time,
        "end_time": ann.end_time,
        "label": ann.label or "",
    }
    file_exists = target_file.exists()
    with target_file.open("a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "start_time", "end_time", "label"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return {"status": "ok", "message": "Annotation saved"}


@app.get("/")
def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "message": "Use /datasets to list available datasets and /window to pull a time window.",
        "datasets_endpoint": "/datasets",
        "window_endpoint": "/window?dataset={id}&start_time=...&duration_seconds=10",
    }
