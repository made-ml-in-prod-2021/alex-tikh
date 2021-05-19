import logging
import logging.config
import yaml
import click
import pickle

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

    logger.info(f"train/test spit")
    train_df, test_df = split_train_val_data(df, params.split_params)
    logger.debug(f"train shape: {train_df.shape}")
    logger.debug(f"test shape: {test_df.shape}")

    logger.info(f"feature engineering")
    transformer = build_transformer(params.feature_params)
    transformer.fit(train_df.drop(columns=['target']))

    logger.info(f"create train features and target")
    train_features = make_features(transformer, train_df.drop(columns=['target']))
    train_target = extract_target(train_df, params.feature_params)

    logger.info(f"fit model")
    model = Classifier(params.model_params)
    model.fit(train_features, train_target)
    logger.info(f"model is fitted")

    logger.info(f"create test features and target")
    test_features = make_features(transformer, test_df.drop(columns=['target']))
    test_target = extract_target(test_df, params.feature_params)

    logger.info(f"made predictions")
    pred = model.predict(test_features)

    score = get_score(test_target, pred)
    logger.debug(f"ROC-AUC: {score}")

    logger.info(f"save model")
    model.dump(params.output_model_path)

    logger.info(f"save transformer")
    with open(params.output_transformer_path, "wb") as f:
        pickle.dump(transformer, f)

    logger.info(f"train pipeline is finished")
    return score


@click.command(name="train_pipeline")
@click.argument("config_path")
def pipeline(config_path: str):
    params = read_training_pipeline_params(config_path)
    setup_logging(params.logger_config)
    train_pipeline(params)


if __name__ == "__main__":
    pipeline()
