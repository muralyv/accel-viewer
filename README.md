# accel-viewer

FastAPI + Plotly viewer for accelerometer CSV/Parquet files. Streams small windows, supports aggregation, lets you select a range and save annotations to CSV. Packaged with a CLI (`accel-viewer`) for cross-platform use.

## Install
```
python -m venv .venv
.\.venv\Scripts\Activate    # or source .venv/bin/activate
pip install -e .
```

Data: place your CSVs in `data/` (create if missing). Each file needs a `time` column plus signal columns (e.g., X,Y,Z). Optional: preconvert to Parquet for speed:
```
python prepare_data.py
```

## Run
```
accel-viewer
```
Then open http://localhost:8000/

Environment overrides:
- `DATA_DIR`: where CSV/Parquet live (default `./data`)
- `PARQUET_DIR`: where Parquet is read/written (default `DATA_DIR`)
- `AUTO_PARQUET=true`: auto CSV→Parquet on first use
- `SIGNALS`: default signals if you don’t specify in UI/API (space/comma separated)
- `DEFAULT_SAMPLE_RATE`: used if sampling rate can’t be inferred (default 30)
- `ANNOTATION_FILE`: default annotation CSV (default `data/annotations.csv`)
- `HOST`/`PORT`: server bind (default 127.0.0.1:8000)

## Usage (UI)
- Pick dataset, set window (raw: sample count; aggregated: units of the selected bin), choose signals (auto-filled from columns).
- Aggregate by None/sec/min/hour/day; window/Prev/Next step in that unit.
- Select a range on the plot (range slider or drag-select). Selected start/end show in the annotation panel.
- Enter an annotation label and optional file name (forced to `.csv`, saved under `data/`), click “Save annotation”.

## API
- `GET /datasets` — list datasets with columns and sample rate estimate.
- `GET /window`
  - `dataset`
  - `start_time` (ISO) or `start_epoch_ms`
  - `duration_seconds` (raw: sample count; aggregated: bin units)
  - `signals` (repeat or comma/space separated; if omitted, uses `SIGNALS` or all columns)
  - `bin_size` + `bin_unit` (`s|m|h|d`)
  - `aggregate` (mean)
- `POST /annotate`
  - body: `{"dataset": "...", "start_time": "...", "end_time": "...", "label": "...", "file_name": "my_notes.csv"}` (file_name optional; forced to `.csv` under `data/`)

Annotations CSV columns: `dataset,start_time,end_time,label`

## Packaging notes
- Source layout: `src/accel_viewer/` contains the app and static assets.
- CLI entrypoint: `accel-viewer` runs `uvicorn accel_viewer.app:app`.
- Static files are bundled with the package. Data stays external.
