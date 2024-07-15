import os, pickle


class Predictor:

    def __new__(cls):
        if not hasattr(cls, "_inst"):
            cls._inst = super(Predictor, cls).__new__(cls)
            cls.__models = {}
            cls.__thresholds = {
                os.environ["XGB_PATH"]: float(os.environ["XGB_THR"]),
                os.environ["RF_PATH"]: float(os.environ["RF_THR"]),
                os.environ["KNN_PATH"]: float(os.environ["KNN_THR"]),
            }
            paths = [os.environ["XGB_PATH"], os.environ["RF_PATH"], os.environ["KNN_PATH"]]
            for path in paths:
                with open(path, 'rb') as file:
                    cls.__models[path] = pickle.load(file)
            print(cls.__models)
        return cls._inst