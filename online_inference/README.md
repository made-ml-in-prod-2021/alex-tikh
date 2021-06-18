## Prerequisites

* Python >= 3.8
* pip >= 19.0.3
* docker >= 20.10.6

## Installation

### From GitHub

```bash
git clone https://github.com/made-ml-in-prod-2021/alex-tikh.git
git checkout homework2
cd online_inference
docker build -t alexandertikh/online_inference:v3 .
```

### From DockerHub

```bash
docker pull alexandertikh/online_inference:v3
```

## Usage

### Run inference

```bash
docker run -p 8000:8000 alexandertikh/online_inference:v3
```

Run from another terminal or IDE:
```bash
python make_request.py
```

### Run tests

```bash
pip install -q pytest pytest-cov
python -m pytest . -v --cov
```

## Docker optimizations

The [optimizations](https://hub.docker.com/r/alexandertikh/online_inference/tags?page=1&ordering=last_updated) included:
- only the required packages in `requirements.txt`
- only the required source files to Docker image (no data, tests, etc.)
- packages' installation without cache
- different python base images:
    - **v1**: python:3.8 (514.12 MB compressed)
    - **v2**: python:3.8-slim (231.84 MB compressed)
    - **v3**: python:3.8-slim with pip no cache (144.32 MB compressed)

## Project structure

------------

    ├── data                           <- The data needed to make test requests.
    │
    ├── models                         <- Pretrained transformer and model.
    │   ├── logistic_regression.pkl
    │   └── transformer.pkl
    │
    ├── src                            <- Source code.
    │   ├── __init__.py                <- Makes src a Python module.
    │   │
    │   └── entities                   <- Parameters for different project modules.
    │       ├── heart_model.py
    │       └── heart_response.py
    │
    ├── tests                          <- Tests.
    │   └── test_app.py
    │
    ├── app.py                         <- FastAPI application code.
    │
    ├── Dockerfile                     <- Docker image building file.
    │
    ├── make_request.py                <- Script to make requests to application
    │                                     predict endpoint.
    │
    ├── README.md                      <- The README for developers using this project.
    │
    └── requirements.txt               <- The requirements file for reproducing the environment.

------------

## Самоанализ

Не сделал только валидацию входных данных

1) +3
2) +3
3) +2
4) -
5) +4
6) +3
7) +2
8) +1
9) +1

**Итого: 19 баллов**