# Wave Features Extraction
Questo repo contiene il codice relativo al talk "The Sound Of Malware".

E' diviso in 2 branch. 
1. Il `main` è di sperimentazione e permette di addestrare ed esportare modelli.
2. Il `pipeline` è ingegnerizzato e consente di utilizzare i modelli esportati per fare delle predizioni.

## Utilizzo
Per utilizzare il codice è necessario installare i requirements tramite comando:
```
pip install -r requirements.txt
```
Ed installare ffmpeg tramite comando:
```
sudo apt-get install ffmpeg
```
NB Se sei su sistema windows ffmpeg va insallato in modi alternativi.

## Funzionamento
Per prima cosa è necessario scaricare i binari. Per questo puoi seguire due strade:
1. Scarichi i binari malevoli dal link [Malware Source 1](https://github.com/MalwareSamples/Linux-Malware-Samples), e li inserisci nel path
`binaries/1/`, e i binari legittimi in `binaries/0/` (per questi puoi prendere spunto dal file `copy_source.py` che copia la cartella `/bin` di ubuntu),
2. Scarichi i binari forniti dal link [Whole Dataset](https://mega.nz/file/alNw0aCa#sg8x5pmCgeDyoT0BvXURL45-2GoYX86ILhKglx1Yi2w). Sono già divisi nelle folder 
richieste quindi ti basterà decomprimere il file nella root del progetto.

Una volta scaricati i files gli script necessari al funzionamento sono 3:
1. `converter.py`: Converte il contenuto di `/binaries/{0|1}` in file `.wav`, dove il contenuto di `/binaries/0/` è legittimo ed
è popolato dallo step precedente, mentre la cartella `/binaries/1` deve essere creata prima del lancio di questo script, e deve contenere i 
binari malevoli scaricabili da [qui](https://github.com/MalwareSamples/Linux-Malware-Samples).
2. `features_extraction.py`: Questo file fa features extraction dai `.wav` creati nello step precedente
e genera in output un `.csv` .
3. `models.py`: Inizia il train dei modelli.

## Altro

1. Il file `support.py` è stato scritto per eseguire test veloci e per fare considerazioni. Non è 
collegato al funzionamento del resto.
2. Il file `constants.py` contiene le variabili costanti utilizzati nei diversi files.
3. Il file `analysis.py` contiene codice per esaminare il dataset e fare delle considerazioni su di esso.

## Note
Il repo è scritto in `python3.12` su sistema `Ubuntu 24.04`.
Si consiglia di eseguire questo codice in ambienti controllati come VM o Container
in quanto vengono gestiti binari affetti da Malware.
Link Utili:
1. [Paper](https://www.sciencedirect.com/science/article/abs/pii/S016740482300192X)
2. [Spafe](https://github.com/SuperKogito/spafe)
3. [Malware Source 1](https://github.com/MalwareSamples/Linux-Malware-Samples)
4. [Whole Dataset](https://mega.nz/file/alNw0aCa#sg8x5pmCgeDyoT0BvXURL45-2GoYX86ILhKglx1Yi2w)