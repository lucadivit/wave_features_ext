import os, pickle, numpy as np

import pandas as pd

from Logger import Logger


class Predictor:

    def __new__(cls):
        if not hasattr(cls, "_inst"):
            cls._inst = super(Predictor, cls).__new__(cls)
            cls.__logger = Logger().get_logger()
            cls.__thresholds = {
                os.environ["XGB_PATH"]: float(os.environ["XGB_THR"]),
                os.environ["RF_PATH"]: float(os.environ["RF_THR"]),
                os.environ["KNN_PATH"]: float(os.environ["KNN_THR"]),
            }
            cls.__logger.info("Thresholds Loaded")

            cls.__models = {}
            cls.__transformers = {}
            paths = [os.environ["XGB_PATH"], os.environ["RF_PATH"], os.environ["KNN_PATH"]]
            for path in paths:
                    cls.__models[path] = cls.__load_model(self=cls, file_path=path)
            cls.__logger.info("Models Loaded")

            cls.__transformers[os.environ["XGB_PATH"]] = cls.__load_model(self=cls, file_path=os.environ["XGB_TRN"])
            cls.__transformers[os.environ["RF_PATH"]] = cls.__load_model(self=cls, file_path=os.environ["RF_TRN"])
            cls.__transformers[os.environ["KNN_PATH"]] = cls.__load_model(self=cls, file_path=os.environ["KNN_TRN"])
            cls.__logger.info("Scalers Loaded")

            cls.__summarized_prediction_key = "final_prediction"
            cls.__max_perc_allowed = float(os.environ["MAX_PERC_ALLOWED"])
        return cls._inst

    def __load_model(self, file_path: str) -> object | None:
        model = None
        if len(file_path) > 0:
            with open(file_path, 'rb') as file:
                model = pickle.load(file)
        return model

    def __summarize_predictions(self, row: pd.Series) -> pd.Series:
        nb_cols = len(row)
        occur = (row == 1).sum()
        result = 0
        if occur >= nb_cols / 2:
            result = 1
        row[self.__summarized_prediction_key] = result
        return row

    def __finalize_prediction(self, df: pd.DataFrame) -> int:
        final_prediction = 0
        predictions = df[self.__summarized_prediction_key]
        nb_rows = predictions.shape[0]
        nb_1s = (predictions == 1).sum()
        if self.__max_perc_allowed == 0:
            if nb_1s > 0:
                final_prediction = 1
        else:
            ratio = nb_1s / nb_rows
            if ratio > self.__max_perc_allowed:
                final_prediction = 1
        return final_prediction

    def predict(self, data_to_predict: pd.DataFrame) -> int:
        tmp_df = pd.DataFrame()
        for model_path in self.__models:
            model = self.__models[model_path]
            scaler = self.__transformers[model_path]
            if scaler is not None:
                data_to_predict = scaler.transform(data_to_predict)
            predictions = model.predict_proba(data_to_predict)[:, 1]
            tmp_df[model_path] = (predictions >= self.__thresholds[model_path]).astype(int)
        tmp_df = tmp_df.apply(lambda row: self.__summarize_predictions(row=row), axis=1)
        final_prediction = self.__finalize_prediction(df=tmp_df)
        return final_prediction
