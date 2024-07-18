# Wave Features Extraction

## Funzionamento

Gli script necessari al funzionamento sono 3:
1. `copy_source.py`: Copia tutto il contenuto della cartella `/bin` nella cartella specificata.
Questi binari rappresentano il dataset legittimo. La folder del progetto nella quale vanno copiati è `/binaries/0/`.
2. `converter.py`: Converte il contenuto di `/binaries/{0|1}` in file `.wav`, dove il contenuto di `/binaries/0/` è legittimo ed
è popolato dallo step precedente, mentre la cartella `/binaries/1` deve essere creata prima del lancio di questo script, e deve contenere i 
binari malevoli scaricabili da [qui](https://github.com/MalwareSamples/Linux-Malware-Samples).
3. `features_extraction.py`: Questo file fa features extraction dai `.wav` creati nello step precedente
e genera in output un `.csv` .
4. `models.py`: Inizia il train dei modelli.

## Altro

1. Il file `support.py` è stato scritto per eseguire test veloci e per fare considerazioni. Non è 
collegato al funzionamento del resto.
2. Il file `constants.py` contiene le variabili costanti utilizzati nei diversi files.
3. Il file `analysis.py` contiene codice per esaminare il dataset e fare delle considerazioni su di esso.

## Note
Il repo è scritto in `python3.12` su sistema `Ubuntu 24.04`.
Link Utili:
1. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S016740482300192X)
2. [Spafe](https://github.com/SuperKogito/spafe)
3. [Malware Source 1](https://github.com/MalwareSamples/Linux-Malware-Samples)
4. 4. [Whole Dataset](https://drive.google.com/file/d/1LB-FmN4h9RSQyVr-EPrVjUV1BIvOtko2/view?usp=sharing)