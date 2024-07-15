import os, pickle,  numpy as np
from Logger import Logger


class Predictor:

    def __new__(cls):
        if not hasattr(cls, "_inst"):
            cls._inst = super(Predictor, cls).__new__(cls)
            cls.__logger = Logger().get_logger()
            cls.__models = {}
            cls.__thresholds = {
                os.environ["XGB_PATH"]: float(os.environ["XGB_THR"]),
                os.environ["RF_PATH"]: float(os.environ["RF_THR"]),
                os.environ["KNN_PATH"]: float(os.environ["KNN_THR"]),
            }
            cls.__logger.info("Thresholds Loaded")
            paths = [os.environ["XGB_PATH"], os.environ["RF_PATH"], os.environ["KNN_PATH"]]
            for path in paths:
                with open(path, 'rb') as file:
                    cls.__models[path] = pickle.load(file)
            cls.__logger.info("Models Loaded")
        return cls._inst

    def predict(self, segment: np.ndarray) -> int:
        predictions = []
        final_prediction = 0
        for model_path in self.__models:
            model = self.__models[model_path]
            prediction = model.predict_proba(segment)[:, 1]
            prediction = (prediction >= self.__thresholds[model_path]).astype(int)
            predictions.append(prediction)
        if predictions.count(1) >= len(predictions) / 2:
            final_prediction = 1
        return final_prediction
