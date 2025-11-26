# accel-viewer

FastAPI + Plotly viewer for accelerometer CSV files. 

## Install
```
python -m venv .venv
.\.venv\Scripts\Activate    # or source .venv/bin/activate
pip install -e .
```

Data: place your CSVs in `data/` . Each file needs a `time` column plus signal columns (e.g., X,Y,Z). 
```
python prepare_data.py
```

## Run
```
accel-viewer
```
Then open http://localhost:8000/


