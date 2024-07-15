from dotenv import load_dotenv
from Predictor import Predictor
from FeaturesExtractor import FeaturesExtractor
from ColumnWiseOutlierClipper import ColumnWiseOutlierClipper

load_dotenv()

predictor = Predictor()
extractor = FeaturesExtractor()
df = extractor.create_features(binary_file_path="test")

