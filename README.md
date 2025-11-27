# accel-viewer

FastAPI-based accelerometer data viewer with annotation of selected windows.

## Install

1) From source

Clone the repo and install in an isolated environment.

Option A: venv

```
git clone https://github.com/muralyv/accel-viewer.git
cd accel-viewer

python -m venv .venv
# Windows:
.\.venv\Scripts\Activate
# macOS / Linux:
source .venv/bin/activate

pip install -e .
```

Option B: Conda

```
git clone https://github.com/muralyv/accel-viewer.git
cd accel-viewer

conda create -n accel-viewer python=3.11 -y
conda activate accel-viewer
pip install -e .
```

## Data

Each CSV file needs:

- a time column (time stamp 2025-10-13 12:30:00)
- one or more signal columns (for example: X axis)

## Run

Linux / macOS:

```
export DATA_DIR=/projects/your_lab/accel_data
export ANNOTATION_FILE=/projects/your_lab/annotations.csv

accel-viewer 
```

Windows PowerShell:

```
$env:DATA_DIR = "C:\path\to\accel_data"
$env:ANNOTATION_FILE = "C:\path\to\annotations.csv"

accel-viewer 
```

Then open:

```
http://127.0.0.1:8000/
```

## pipx (optional, isolated from base Python)

If you prefer pipx:

```
pip install pipx
pipx install "git+https://github.com/muralyv/accel-viewer.git"
accel-viewer --data-dir /path/to/your/csvs
```

## Environment overrides

- DATA_DIR: where CSV files live (default ./data)
- SIGNALS: default signals if not specified (space/comma separated)
- DEFAULT_SAMPLE_RATE: fallback sample rate (default 30)
- ANNOTATION_FILE: default annotation CSV (default data/annotations.csv)
- HOST / PORT: bind address (default 127.0.0.1:8000)

## Usage (UI)

- Pick dataset and set window.
  - Raw mode: window = sample count.
  - Aggregated mode: window = bin units (sec / min / hour / day).
- Aggregate by: None / sec / min / hour / day; Prev/Next buttons step by that unit.
- Select a range on the plot (using the range slider or drag-select). The selected start_time and end_time are shown in the annotation panel.
- Enter an annotation label and a file name (saved under data/), then click "Save annotation".
  The annotation file will be saved as a .csv.

Annotations CSV columns:

```
dataset,start_time,end_time,label
```
