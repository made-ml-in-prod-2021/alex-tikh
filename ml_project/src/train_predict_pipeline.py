import logging
import logging.config
import yaml
import click
import pickle
import pandas as pd

from src.data import read_data, split_train_val_data
from src.features import make_features, extract_target, build_transformer
from src.models import Classifier, get_score
from src.entities import TrainingPipelineParams, read_training_pipeline_params


logger = logging.getLogger('log_info')


def setup_logging(path: str) -> None:
    with open(path) as config_f:
        logging.config.dictConfig(yaml.safe_load(config_f))


def train_pipeline(params: TrainingPipelineParams) -> float:
    logger.info(f"start train pipeline")
    df = read_data(params.input_data_path)
    logger.info(f"load data, shape: {df.shape}")
    train_df, test_df = split_train_val_data(df, params.split_params)
    logger.info(f"train/test spit")
    logger.debug(f"train shape: {train_df.shape}")
    logger.debug(f"test shape: {test_df.shape}")
    transformer = build_transformer(params.feature_params)
    logger.info(f"feature engineering")
    transformer.fit(train_df.drop(columns=['target']))
    train_features = make_features(transformer, train_df.drop(columns=['target']))
    train_target = extract_target(train_df, params.feature_params)
    logger.info(f"create train features and target")
    model = Classifier(params.model_params)
    logger.info(f"fit model")
    model.fit(train_features, train_target)
    logger.info(f"model is fitted")
    test_features = make_features(transformer, test_df.drop(columns=['target']))
    test_target = extract_target(test_df, params.feature_params)
    logger.info(f"create test features and target")
    pred = model.predict(test_features)
    logger.info(f"made predictions")
    score = get_score(test_target, pred)
    logger.debug(f"ROC-AUC: {score}")
    model.dump(params.output_model_path)
    logger.info(f"save model")
    with open(params.output_transformer_path, "wb") as f:
        pickle.dump(transformer, f)
    logger.info(f"save transformer")
    logger.info(f"train pipeline is finished")
    return score


def predict_pipeline(params: TrainingPipelineParams) -> pd.DataFrame:
    logger.info(f"start predict pipeline")
    df = read_data(params.input_data_predict_path)
    logger.info(f"open data")
    logger.debug(f"data shape: {df.shape}")
    with open(params.output_model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"load model")
    with open(params.output_transformer_path, "rb") as f:
        transformer = pickle.load(f)
    logger.info(f"load transformer")
    transformed_df = make_features(transformer, df.drop(columns=['target']))
    predicts = model.predict(transformed_df)
    logger.info(f"prediction")
    pd.DataFrame(predicts, columns=["target"]).to_csv(params.output_data_predict_path, index=False)
    logger.info(f"predict pipeline is finished")
    return pd.DataFrame(predicts, columns=["target"])


@click.command(name="train_predict_pipeline")
@click.argument("config_path")
@click.argument("mode")
def pipeline(config_path: str, mode: str):
    params = read_training_pipeline_params(config_path)
    setup_logging(params.logger_config)
    if mode == "train":
        train_pipeline(params)
    elif mode == "predict":
        predict_pipeline(params)


if __name__ == "__main__":
    pipeline()
