import logging, sys


class Logger:

    def __new__(cls):
        if not hasattr(cls, "_inst"):
            cls._inst = super(Logger, cls).__new__(cls)
            logging.basicConfig(level=logging.INFO,
                                format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                                handlers=[logging.StreamHandler(stream=sys.stdout)])
            cls.__logger = logging.getLogger()
            cls.__logger.propagate = False
        return cls._inst

    @classmethod
    def get_logger(cls) -> logging.Logger:
        return cls.__logger