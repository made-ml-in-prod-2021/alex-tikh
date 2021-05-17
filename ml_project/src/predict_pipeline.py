import logging
import logging.config
import yaml
import click
import pickle
import pandas as pd

from src.data import read_data
from src.features import make_features
from src.entities import PredictionPipelineParams, read_prediction_pipeline_params


logger = logging.getLogger('log_info')


def setup_logging(path: str) -> None:
    with open(path) as config_f:
        logging.config.dictConfig(yaml.safe_load(config_f))


def predict_pipeline(params: PredictionPipelineParams) -> pd.DataFrame:
    logger.info(f"start predict pipeline")

    logger.info(f"open data")
    df = read_data(params.input_data_predict_path)
    logger.debug(f"data shape: {df.shape}")

    logger.info(f"load model")
    with open(params.output_model_path, "rb") as f:
        model = pickle.load(f)

    logger.info(f"load transformer")
    with open(params.output_transformer_path, "rb") as f:
        transformer = pickle.load(f)

    logger.info(f"create features")
    transformed_df = make_features(transformer, df.drop(columns=['target']))

    logger.info(f"prediction")
    predicts = model.predict(transformed_df)

    logger.info(f"save predictions")
    pd.DataFrame(predicts, columns=["target"]).to_csv(params.output_data_predict_path, index=False)

    logger.info(f"predict pipeline is finished")
    return pd.DataFrame(predicts, columns=["target"])


@click.command(name="predict_pipeline")
@click.argument("config_path")
def pipeline(config_path: str):
    params = read_prediction_pipeline_params(config_path)
    setup_logging(params.logger_config)
    predict_pipeline(params)


if __name__ == "__main__":
    pipeline()
