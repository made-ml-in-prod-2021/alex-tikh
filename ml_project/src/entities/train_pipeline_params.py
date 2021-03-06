from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml

from .feature_params import FeatureParams
from .split_params import SplittingParams
from .model_params import RfParams, LrParams

from typing import Union

ModelParams = Union[RfParams, LrParams]


@dataclass
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    output_transformer_path: str
    logger_config: str
    split_params: SplittingParams
    feature_params: FeatureParams
    model_params: ModelParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
