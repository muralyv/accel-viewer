import logging
from pathlib import Path

import duckdb

DATA_DIR = Path(__file__).parent / "data"
PARQUET_DIR = DATA_DIR
logger = logging.getLogger("prepare_parquet")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def csv_to_parquet(csv_path: Path) -> Path:
    parquet_path = PARQUET_DIR / f"{csv_path.stem}.parquet"
    if parquet_path.exists():
        logger.info("Parquet already exists for %s", csv_path.name)
        return parquet_path

    logger.info("Converting %s -> %s", csv_path.name, parquet_path.name)
    conn = duckdb.connect(database=":memory:")
    conn.execute("SET TimeZone='UTC';")
    conn.execute(
        f"""
        COPY (
            SELECT *
            FROM read_csv_auto('{csv_path.as_posix().replace("'", "''")}',
                               timestampformat='%Y-%m-%dT%H:%M:%S%Z')
        ) TO '{parquet_path.as_posix().replace("'", "''")}' (FORMAT PARQUET);
        """
    )
    conn.close()
    return parquet_path


def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit(f"Data directory not found: {DATA_DIR}")
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {DATA_DIR}")

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    for csv_path in csv_files:
        csv_to_parquet(csv_path)


if __name__ == "__main__":
    main()
