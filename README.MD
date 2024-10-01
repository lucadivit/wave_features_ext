# Wave Features Extraction

Questo branch contiene codice utilizzabile per il riconoscimento dei malware da binari,
ed è relativo al talk "The Sound Of Malware".

## Funzionamento

Per utilizzare lo script è necessario creare un file di environment chiamato 
`.env`. Un esempio può essere trovato nella cartella di progetto, dove sono
riportate anche le spiegazioni delle singole voci.

Inoltre è necessario installare i requirements:
```
pip install -r requirements.txt
```

ed infine è richiesta la libreria FFMPEG
```
sudo apt-get install ffmpeg
```

NB: Se sei su sistema Windows il comando sopra non funzionerà per cui dovrai 
scaricare e aggiungere ffmpeg alle variabili di ambiente.

I dati forniti non corrispondono a quelli utilizzati in fase di studio e test, <u>i risultati sono quindi svincolati rispetto quelli presentati nel talk.</u>

Infine per lanciare il programma:
```
python3 main.py
```

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
Si consiglia vivamente di lanciare questo programma in SandBox (VM, Container, ecc) in quanto i file binari sono 
affetti da malware.