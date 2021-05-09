# ML project for ML in Production course

## Prerequisites

* Python >= 3.8
* pip >= 20.1.1
* [Heart Disease UCI Dataset](https://www.kaggle.com/ronitf/heart-disease-uci)

## Installation

```bash
git clone https://github.com/made-ml-in-prod-2021/alex-tikh.git
git checkout homework1
cd ml_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Train model

```bash
python3 train_predict_pipeline.py configs/lr_config.yaml train
```

### Predict with model

```bash
python3 src/train_predict_pipeline.py configs/lr_config.yaml predict
```

## Project structure

------------

    ├── config                        <- Configuration files for project modules.
    │   ├── logging_config.yaml
    │   ├── lr_config.yaml
    │   └── rf_config.yaml
    │
    ├── data
    │   └── raw                        <- The original, immutable data dump.
    │
    ├── models                         <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks                      <- Jupyter notebooks.
    │
    ├── src                            <- Source code for use in this project.
    │   ├── __init__.py                <- Makes src a Python module.
    │   │
    │   ├── data                       <- Scripts to download or generate data.
    │   │   └── make_dataset.py
    │   │
    │   ├── entities                   <- Parameters for different project modules.
    │   │   ├── feature_params.py
    │   │   ├── model_params.py
    │   │   ├── split_params.py
    │   │   └── train_pipeline_params.py    
    │   │
    │   ├── features                   <- Scripts to turn raw data into features for modeling.
    │   │   ├── build_features.py
    │   │   └── custom_transformer.py  
    │   │
    │   ├── models                     <- Scripts to train models and then use trained models to make
    │   │   │                             predictions.
    │   │   └── classifier.py
    │
    ├── LICENSE
    │
    ├── log_info.log                   <- Training and prediction log file.
    │
    ├── README.md                      <- The top-level README for developers using this project.
    │
    ├── requirements.txt               <- The requirements file for reproducing the analysis environment, e.g.
    │                                     generated with `pip freeze > requirements.txt`
    │
    ├── setup.py                       <- Makes project pip installable (pip install -e .) so src can be imported.
    │
    └── train_predict_pipeline.py      <- Main script.


------------