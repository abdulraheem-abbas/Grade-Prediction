# Student Performance

FastAPI prediction service with an Arabic web interface and a pretrained PyCaret model.

## Run locally

Use Python 3.11:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn student_performance_api:app --reload
```

Open http://127.0.0.1:8000 for the website or `/docs` for API documentation.

## Deploy on Render

Connect this GitHub repository as a Render Blueprint using `render.yaml`, or create
a Python Web Service with these settings:

- Python version: `3.11.11`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn student_performance_api:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`

The website and API share the same service URL. Keep
`student_performance_model.pkl` in the repository; deployment loads it without retraining.
Choose a service size with enough memory for PyCaret and its dependencies.

To retrain locally, run `python train.py`, then commit the updated model.
The displayed percentage bands are class labels, not predicted exact percentages.
