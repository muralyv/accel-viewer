# accel-viewer

FastAPI + Plotly viewer for accelerometer CSV files. 
=======
FastAPI-based viewer for accelerometer data with annotation of selected windows.

## Install

### 1) From source (all platforms)
```bash
git clone https://github.com/muralyv/accel-viewer.git
cd accel-viewer
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate
# macOS / Linux:
source .venv/bin/activate
pip install -e .
```
Run:
```bash
accel-viewer --data-dir /path/to/your/csvs
```

Data: place your CSVs in `data/` . Each file needs a `time` column plus signal columns (e.g., X,Y,Z). 
=======
### pipx (optional, isolated from base Python)
```bash
pip install pipx
pipx install "git+https://github.com/muralyv/accel-viewer.git"
accel-viewer --data-dir /path/to/your/csvs
```

### 2) Windows (local desktop)
```bash
git clone https://github.com/muralyv/accel-viewer.git
cd accel-viewer
python -m venv .venv
.\.venv\Scripts\Activate
pip install -e .
# point to your data
accel-viewer --data-dir "C:\data\accel"
```
Open http://localhost:8000/

### 3) macOS / Linux (local)
```bash
git clone https://github.com/muralyv/accel-viewer.git
cd accel-viewer
python -m venv .venv
source .venv/bin/activate
pip install -e .
accel-viewer --data-dir /path/to/csvs
```
Open http://localhost:8000/

### 4) Linux server / HPC
```bash
git clone https://github.com/muralyv/accel-viewer.git
cd accel-viewer
python -m venv .venv
source .venv/bin/activate
pip install -e .
export DATA_DIR=/projects/your_lab/accel_data
accel-viewer --host 0.0.0.0 --port 8000
```
SSH tunnel from laptop:
```bash
ssh -L 8000:localhost:8000 youruser@your-hpc-login-node
```
Then open http://localhost:8000/ locally.

### 5) Container (Docker example)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY static ./static
RUN pip install --no-cache-dir .
ENV DATA_DIR=/data
ENV PARQUET_DIR=/data
EXPOSE 8000
CMD ["accel-viewer", "--host", "0.0.0.0"]
```
Build/run:
```bash
docker build -t accel-viewer .
docker run -p 8000:8000 -v /path/to/csvs:/data accel-viewer
```
Open http://localhost:8000/.

## Run
- Put CSVs in `data/` (time column + signal columns).
- Start:
```bash
accel-viewer
```
- Open http://localhost:8000/


=======
Environment overrides:
- `DATA_DIR`: where CSV files live (default `./data`)
- `SIGNALS`: default signals if not specified (space/comma separated)
- `DEFAULT_SAMPLE_RATE`: fallback sample rate (default 30)
- `ANNOTATION_FILE`: default annotation CSV (default `data/annotations.csv`)
- `HOST` / `PORT`: bind (default 127.0.0.1:8000)

## Usage (UI)
- Pick dataset, set window  
  - Raw: window = sample count  
  - Aggregated: window = bin units (sec/min/hour/day)
- Aggregate by: None / sec / min / hour / day; Prev/Next steps by that unit.
- Select a range on the plot (range slider or drag-select); selected start/end show in the annotation panel.
- Enter annotation label and file name (forced to `.csv`, saved under `data/`), click “Save annotation”.

Annotations CSV columns:
```
dataset,start_time,end_time,label
```

## Packaging notes
- Source layout: `src/accel_viewer/` (app + static).
- CLI: `accel-viewer` runs `uvicorn accel_viewer.app:app`.
- Static files bundled; data stays external.
