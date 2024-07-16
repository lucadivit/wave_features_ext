# Wave Features Extraction

## Funzionamento

Per utilizzare lo script è necessario creare un file di environment chiamato 
`.env`. Un esempio può essere trovato nella cartella di progetto, dove sono
riportate anche le spiegazioni delle singole voci.

Snippet di codice:
```
# Import
from dotenv import load_dotenv
from Predictor import Predictor
from FeaturesExtractor import FeaturesExtractor
# Anche se non utilizzato esplicitamente, è necessario per il corretto caricamento
# del file pickle
from ColumnWiseOutlierClipper import ColumnWiseOutlierClipper

load_dotenv()

predictor = Predictor()
extractor = FeaturesExtractor()
# "test" è il file binario del quale effettuare la predizione
df = extractor.create_features(binary_file_path="test")
result = predictor.predict(data_to_predict=df)
print(result)
```

## Note
Il repo è scritto in `python3.12` su sistema `Ubuntu 24.04`.