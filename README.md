# accel-viewer

![Status](https://img.shields.io/badge/status-work%20in%20progress-gray)
![License](https://img.shields.io/github/license/muralyv/accel-viewer)
[![Python Tests](https://github.com/muralyv/accel-viewer/actions/workflows/ci.yml/badge.svg)](https://github.com/muralyv/accel-viewer/actions/workflows/ci.yml)

FastAPI-based accelerometer data viewer with annotation of selected windows.  
It uses DuckDB and FastAPI to provide fast, efficient querying and visualization of large accelerometer datasets.

## Install

### 1) From source

**Note:** If you don’t have Git installed, go to the GitHub repository at  
https://github.com/muralyv/accel-viewer  
and click **“Code” → “Download ZIP”** to download the project manually, then unzip it.

Clone the repo and install in an isolated environment.



```bash
git clone https://github.com/muralyv/accel-viewer.git
cd accel-viewer
python --version  # expect Python 3.11

#### Option A: venv

python -m venv .venv

.\.venv\Scripts\Activate.ps1 # if you are in powershell
.\.venv\Scripts\activate.bat # if you are using cmd
source .venv/bin/activate # # macOS / Linux:


pip install -e .
```

#### Option B: Conda

```bash
git clone https://github.com/muralyv/accel-viewer.git
cd accel-viewer

conda create -n accel-viewer python=3.11 -y
conda activate accel-viewer
pip install -e .
```

## Data

Input data must be provided as CSV (`.csv`) files.

Each CSV file must have:

- The **first column** as a `time` column, with timestamps in **one** of the following formats:  
  - `YYYY-MM-DD HH:MM:SS` (e.g. `2025-10-13 12:30:00`)  
  - `YYYY-MM-DDTHH:MM:SS.sssZ` (e.g. `2025-10-13T12:30:00.033Z`, ISO 8601 / UTC)
- One or more signal columns (for example: `X`, `Y`, `Z`) containing numeric values.

## Run

Linux / macOS:

```bash
export DATA_DIR=/projects/your_lab/accel_data
export ANNOTATION_FILE=/projects/your_lab/annotations.csv  # optional

accel-viewer
```

Windows PowerShell:

```powershell
$env:DATA_DIR = "C:\path\to\accel_data"
$env:ANNOTATION_FILE = "C:\path\to\annotations.csv"

accel-viewer
```

Then open:

```text
http://127.0.0.1:8000/
```

## Environment overrides

- `DATA_DIR`: where CSV files live (default `./data`)
- `SIGNALS`: default signals if not specified (space/comma separated)
- `DEFAULT_SAMPLE_RATE`: fallback sample rate (default `30`)
- `ANNOTATION_FILE`: default annotation CSV (default `data/annotations.csv`)
- `HOST` / `PORT`: bind address (default `127.0.0.1:8000`)

## Usage (UI)

- **Dataset**  
  - All `.csv` files in `DATA_DIR` will be listed. Choose a file to visualize.

- **Window**  
  - Default is `60` (e.g., seconds or minutes depending on aggregation).  
  - You can adjust the window length. The unit is determined by **Aggregate by**.

- **Aggregate by**  
  - Default: `None (raw)`  
  - Options: `None (raw)`, `Second`, `Minute`, `Hour`, `Day`  
  - When aggregation is selected, raw values are averaged over the chosen time unit.

- **Signals**  
  - By default, all signal columns are displayed (unless overridden by the `SIGNALS` environment variable).  
  - You can remove unwanted signals or add specific ones (comma/space separated).

- **Prev / Next**  
  - Use **Prev** and **Next** to move backward/forward by one window.

### View panel

- The main plot shows the selected signals over the current window.
- You can zoom or use the range slider to inspect specific segments.

### Annotation panel

- Select a range on the plot (using drag-select or the range slider).  
  The selected `start_time` and `end_time` are shown in the annotation panel.
- Enter an annotation label and an annotation file name  
  (saved under `DATA_DIR` unless overridden by `ANNOTATION_FILE`), then click **“Save annotation”**.
- The annotation file will be saved as a `.csv`.

Annotations CSV columns:

```text
dataset,start_time,end_time,label
```
This tool was developed as part of HEAL-MS (Home-based actigraphy to predict change in neurological function in multiple sclerosis) project. https://msresearch.jhmi.edu/heal-ms/
