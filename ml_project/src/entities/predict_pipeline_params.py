from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml


@dataclass
class PredictionPipelineParams:
    input_data_predict_path: str
    output_data_predict_path: str
    output_model_path: str
    output_transformer_path: str
    logger_config: str


PredictionPipelineParamsSchema = class_schema(PredictionPipelineParams)


def read_prediction_pipeline_params(path: str) -> PredictionPipelineParams:
    with open(path, "r") as input_stream:
        schema = PredictionPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
