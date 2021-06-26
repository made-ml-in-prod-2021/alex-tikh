import logging
import os
import pickle
from typing import List, Optional

import time
from uvicorn import Config
import contextlib
import threading

import pandas as pd
import uvicorn
from fastapi import FastAPI
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.entities import HeartModel, HeartResponse


logger = logging.getLogger(__name__)

PATH_TO_MODEL = 'models/logistic_regression.pkl'
PATH_TO_TRANSFORMER = 'models/transformer.pkl'


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


model: Optional[Pipeline] = None


def make_predict(
    data: List,
    features: List[str],
    model: Pipeline,
    transformer: ColumnTransformer
) -> List[HeartResponse]:
    data = pd.DataFrame(data, columns=features)
    ids = [int(x) for x in data.index]
    data = transformer.transform(data)
    predicts = model.predict(data)

    return [
        HeartResponse(id=id_, target=int(target_))
        for id_, target_ in zip(ids, predicts)
    ]


app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model():

    model_path = os.getenv("PATH_TO_MODEL") if os.getenv("PATH_TO_MODEL") else PATH_TO_MODEL
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)
    global model
    model = load_object(model_path)


@app.on_event("startup")
def load_transformer():

    transformer_path = os.getenv("PATH_TO_TRANSFORMER") if os.getenv("PATH_TO_TRANSFORMER") else PATH_TO_TRANSFORMER
    if transformer_path is None:
        err = f"PATH_TO_TRANSFORMER {transformer_path} is None"
        logger.error(err)
        raise RuntimeError(err)
    global transformer
    transformer = load_object(transformer_path)


@app.get("/status")
def health() -> str:
    return f"Model is ready: {not (model is None)}"


@app.get("/predict/", response_model=List[HeartResponse])
def predict(request: HeartModel):
    return make_predict(request.data, request.features, model, transformer)


# https://github.com/encode/uvicorn/issues/742
class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()


if __name__ == "__main__":
    config = Config("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
    server = Server(config=config)
    time.sleep(os.getenv("SLEEPING_TIME", 30))
    with server.run_in_thread():
        time.sleep(os.getenv("RUNNING_TIME", 120))
