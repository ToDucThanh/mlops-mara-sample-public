import argparse
import logging
import os
import random
import time

import mlflow
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, Request
from pandas.util import hash_pandas_object
from pydantic import BaseModel

from problem_config import ProblemConst, create_prob_config
from raw_data_processor import RawDataProcessor
from utils import AppConfig, AppPath

PREDICTOR_API_PORT = 8000


class Data(BaseModel):
    id: str
    rows: list
    columns: list


class ModelPredictor:
    def __init__(self, config_file_path_1, config_file_path_2):
        with open(config_file_path_1, "r") as f:
            self.config_1 = yaml.safe_load(f)
        logging.info(f"model-config-1: {self.config_1}")

        with open(config_file_path_2, "r") as f:
            self.config_2 = yaml.safe_load(f)
        logging.info(f"model-config-2: {self.config_2}")

        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)

        self.prob_config_1 = create_prob_config(
            self.config_1["phase_id"], self.config_1["prob_id"]
        )

        self.prob_config_2 = create_prob_config(
            self.config_2["phase_id"], self.config_2["prob_id"]
        )

        # load category_index
        self.category_index_1 = RawDataProcessor.load_category_index(self.prob_config_1)
        self.category_index_2 = RawDataProcessor.load_category_index(self.prob_config_2)

        # load model
        model_uri_1 = os.path.join(
            "models:/", self.config_1["model_name"], str(self.config_1["model_version"])
        )
        model_uri_2 = os.path.join(
            "models:/", self.config_2["model_name"], str(self.config_2["model_version"])
        )
        self.model_1 = mlflow.pyfunc.load_model(model_uri_1)
        self.model_2 = mlflow.pyfunc.load_model(model_uri_2)

    def detect_drift(self, feature_df) -> int:
        # watch drift between coming requests and training data
        time.sleep(0.02)
        return random.choice([0, 1])

    def predict_1(self, data: Data):
        start_time = time.time()

        # preprocess
        raw_df = pd.DataFrame(data.rows, columns=data.columns)
        feature_df = RawDataProcessor.apply_category_features(
            raw_df=raw_df,
            categorical_cols=self.prob_config_1.categorical_cols,
            category_index=self.category_index_1,
        )
        # save request data for improving models
        ModelPredictor.save_request_data(
            feature_df, self.prob_config_1.captured_data_dir, data.id
        )

        prediction = self.model_1.predict(feature_df)
        is_drifted = self.detect_drift(feature_df)

        run_time = round((time.time() - start_time) * 1000, 0)
        logging.info(f"Problem 1: prediction takes {run_time} ms")
        return {
            "id": data.id,
            "predictions": prediction.tolist(),
            "drift": is_drifted,
        }

    def predict_2(self, data: Data):
        start_time = time.time()

        # preprocess
        raw_df = pd.DataFrame(data.rows, columns=data.columns)
        feature_df = RawDataProcessor.apply_category_features(
            raw_df=raw_df,
            categorical_cols=self.prob_config_2.categorical_cols,
            category_index=self.category_index_2,
        )
        # save request data for improving models
        ModelPredictor.save_request_data(
            feature_df, self.prob_config_2.captured_data_dir, data.id
        )

        prediction = self.model_2.predict(feature_df)
        is_drifted = self.detect_drift(feature_df)

        run_time = round((time.time() - start_time) * 1000, 0)
        logging.info(f"Problem 2: prediction takes {run_time} ms")
        return {
            "id": data.id,
            "predictions": prediction.tolist(),
            "drift": is_drifted,
        }

    @staticmethod
    def save_request_data(feature_df: pd.DataFrame, captured_data_dir, data_id: str):
        if data_id.strip():
            filename = data_id
        else:
            filename = hash_pandas_object(feature_df).sum()
        output_file_path = os.path.join(captured_data_dir, f"{filename}.parquet")
        feature_df.to_parquet(output_file_path, index=False)
        return output_file_path


class PredictorApi:
    def __init__(self, predictor: ModelPredictor):
        self.predictor = predictor
        self.app = FastAPI()

        @self.app.get("/")
        async def root():
            return {"message": "hello"}

        @self.app.post("/phase-1/prob-1/predict")
        async def predict(data: Data, request: Request):
            self._log_request(request)
            response = self.predictor.predict_1(data)
            self._log_response(response)
            return response

        @self.app.post("/phase-1/prob-2/predict")
        async def predict(data: Data, request: Request):
            self._log_request(request)
            response = self.predictor.predict_2(data)
        
            self._log_response(response)
            return response

    @staticmethod
    def _log_request(request: Request):
        pass

    @staticmethod
    def _log_response(response: dict):
        pass

    def run(self, port):
        uvicorn.run(self.app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    config_path_1 = (
        AppPath.MODEL_CONFIG_DIR
        / ProblemConst.PHASE1
        / ProblemConst.PROB1
        / "model-1.yaml"
    ).as_posix()
    
    config_path_2 = (
        AppPath.MODEL_CONFIG_DIR
        / ProblemConst.PHASE2
        / ProblemConst.PROB2
        / "model-2.yaml"
    ).as_posix()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path-1", type=str, default=config_path_1)
    parser.add_argument("--config-path-2", type=str, default=config_path_2)
    parser.add_argument("--port", type=int, default=PREDICTOR_API_PORT)
    args = parser.parse_args()

    predictor = ModelPredictor(config_file_path_1=args.config_path_1, config_file_path_2=args.config_path_2)
    api = PredictorApi(predictor)
    api.run(port=args.port)
