# API and Frontend for the RotorAI Project

An API that serves the RotorAI model and a frontend that communicates with the API for predictions.

## Table of Contents

- [Project Organization](#project-organization)
- [Running the FastAPI API](#running-the-fastapi-api)
- [Running the Streamlit Frontend](#running-the-streamlit-frontend)

## Project Organization

```text
├── app/                               <- Contains the FastAPI code.
│   ├── models/                        <- Contains models.
│   │   └── best_model.keras
│   ├── normalization_params/          <- Contains standardization parameters learnt during training.
│   │   ├── freq_mean.npy
│   │   ├── freq_std.npy
│   │   ├── time_mean.npy
│   │   └── time_std.npy
│   ├── __init__.py
│   └── main.py                        <- FastAPI code.
├── data/                              <- Contains data for individual prediction.
│   ├── bearing_fault.csv
│   ├── mechanical_looseness_fault.csv
│   ├── misalignment_fault.csv
│   ├── normal.csv
│   └── unbalance_fault.csv
└── frontend/                          <- Contains the Streamlit frontend code.
    ├── .streamlit/
    │   └── config.toml
    ├── __init__.py
    └── app.py
```

## Running the FastAPI API

After making sure that you are in the project root, run the following command:
```bash
python -m uvicorn app.main:app --reload --port 8000
```

## Running the Streamlit Frontend

After making sure that you are in the project root, run the following command:
```bash
streamlit run frontend/app.py --server.runOnSave true
```
