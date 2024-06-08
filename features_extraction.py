import os, librosa, numpy as np, pandas as pd, wave
from pydub import AudioSegment
from gammatone.gtgram import gtgram
from glob import glob
from scipy.fftpack import dct
from matplotlib import pyplot as plt
from spafe.features.bfcc import bfcc
from spafe.utils.preprocessing import SlidingWindow
from constants import (freq, classes, output_folder_name_converter,
                       channels, segment_folder, output_file, skipped_file,
                       y_name)

# Tunable
mfcc_axis = 0
gfcc_axis = 0
bfcc_axis = 0
sr = freq
channels = channels
n_ceps = 30
sec_split = 0.4


def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
    return duration


# Normalizzazione tramite padding
def load_and_normalize_audio(file_path, sr=44100, max_length=40000):
    y, sr = librosa.load(file_path, sr=sr)
    len_y = len(y)
    print(f"Audio is len {len_y}")
    if len_y < max_length:
        print("Padding...")
        y = np.pad(y, (0, max_length - len(y)), mode='constant')
    else:
        y = y[:max_length]
    return y, sr


# Normalizzazione tramite segmentazione
def load_and_segment_audio(file_path, segment_duration=10, overlap=0.5, sr=44100):
    if get_wav_duration(file_path=file_path) > segment_duration:
        y, sr = librosa.load(file_path, sr=sr)
        segment_length = int(segment_duration * sr)
        step_length = int(segment_length * (1 - overlap))
        segments = []
        for start in range(0, len(y), step_length):
            end = start + segment_length
            if end <= len(y):
                segments.append(y[start:end])
            else:
                # Pad the last segment if it is shorter than segment_length
                segment = np.pad(y[start:end], (0, segment_length - len(y[start:end])), mode='constant')
                segments.append(segment)
                break  # Stop if we have padded the last segment
    else:
        segments = None
    return segments, sr


def save_segments_as_mp3(segments, sr, output_dir, base_filename):
    os.makedirs(f"{output_dir}/{base_filename}", exist_ok=True)
    for i, segment in enumerate(segments):
        # Convert numpy array to waves segment
        audio_segment = AudioSegment(
            segment.tobytes(),
            frame_rate=sr,
            sample_width=segment.dtype.itemsize,
            channels=channels
        )
        # Save as MP3
        output_path = f"{output_dir}/{base_filename}/{base_filename}_segment_{i + 1}.mp3"
        audio_segment.export(output_path, format="mp3")
        print(f"Saved segment {i + 1} to {output_path}")


def compute_mfcc_features(y, sr):
    '''
    axis=1: Ogni elemento del vettore rappresenta la media dei valori di un singolo coefficiente MFCC attraverso tutto il segnale waves.
    Fornisce un'idea generale della distribuzione dei coefficienti MFCC per l'intero waves. Dim Output = n_mfcc
    axis=0: Ogni elemento del vettore rappresenta la media dei valori di tutti i coefficienti MFCC per un singolo frame temporale.
    Fornisce un'idea di come le caratteristiche spettrali cambiano nel tempo. Dim Output = n_t.
    Parametro Tunabile
    '''
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_ceps)
    mfcc_mean = np.mean(mfccs, axis=mfcc_axis)
    # mfcc_std = np.std(mfccs, axis=0)
    cols_mfcc = [f"mfcc_mean_{i}" for i, _ in enumerate(mfcc_mean)]
    mfcc_df = pd.DataFrame([mfcc_mean])
    mfcc_df.columns = cols_mfcc
    return mfcc_df


def compute_gfcc_features(y, sr):
    # Metodo 1
    n_filters = 64  # Numero di filtri Gammatone
    win_size = 0.05  # Durata della finestra in secondi
    hop_size = 0.02  # Durata dello step in secondi
    f_min = 50  # Dal digramma di mel le frequenze sono molto alte quindi si può alzare questo valore con perdita minima
    gammatone_spec = gtgram(y, sr, win_size, hop_size, n_filters, f_min)
    log_gammatone_spec = np.log(gammatone_spec + 1e-6)
    gfccs = dct(log_gammatone_spec, type=2, axis=0, norm='ortho')[:n_ceps]
    gfcc_mean = np.mean(gfccs, axis=gfcc_axis)
    # gfcc_std = np.std(gfccs, axis=0)
    cols_gfcc = [f"gfcc_mean_{i}" for i, _ in enumerate(gfcc_mean)]
    gfcc_df = pd.DataFrame([gfcc_mean])
    gfcc_df.columns = cols_gfcc

    # Metodo 2
    # gfccs = gfcc(y, fs=sr, num_ceps=13)
    # print(gfccs.shape)
    return gfcc_df


def compute_bfcc_features(y, sr):
    bfccs = bfcc(y, fs=sr, pre_emph=1, pre_emph_coeff=0.97, num_ceps=n_ceps,
                 window=SlidingWindow(0.03, 0.015, "hamming"), low_freq=0, nfilts=50,
                 high_freq=16000, normalize="mvn").T
    bfcc_mean = np.mean(bfccs, axis=bfcc_axis)
    cols_bfcc = [f"bfcc_mean_{i}" for i, _ in enumerate(bfcc_mean)]
    bfcc_df = pd.DataFrame([bfcc_mean])
    bfcc_df.columns = cols_bfcc
    return bfcc_df


# Funzione per estrarre caratteristiche
def extract_features(y, sr):
    print("Computing MFCC Features")
    mfcc_df = compute_mfcc_features(y=y, sr=sr)
    print("MFCC Shape", mfcc_df.shape)
    print("---------------------------")
    print("Computing GFCC Features")
    gfcc_df = compute_gfcc_features(y=y, sr=sr)
    print("GFCC Shape", gfcc_df.shape)
    print("---------------------------")
    print("Computing BFCC Features")
    bfcc_df = compute_bfcc_features(y=y, sr=sr)
    print("BFCC Shape", bfcc_df.shape)
    print("---------------------------")
    result = pd.concat([mfcc_df, gfcc_df, bfcc_df], axis=1)

    # Other features that i can build.
    '''
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    zero_crossings = librosa.feature.zero_crossing_rate(y)

    # Media e deviazione standard
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    zero_crossings_mean = np.mean(zero_crossings)
    zero_crossings_std = np.std(zero_crossings)

    # Concatenare tutte le caratteristiche
    features = np.concatenate([mfcc_mean, mfcc_std, chroma_mean, chroma_std, [zero_crossings_mean, zero_crossings_std]])
    return features
    '''

    return result


def plot_mel_spectogram(y):
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spettrogramma di Mel')
    plt.show()

# Constants
split_audio = False
process_segments = True

if split_audio:
    for file_class in classes:
        for audio in glob(f"{output_folder_name_converter}{file_class}/*.wav"):
            print(f"Splitting {audio}")
            base_name = audio.split("/")[-1].split(".")[0]
            segments, _ = load_and_segment_audio(file_path=audio, sr=sr, segment_duration=sec_split, overlap=0.3)
            if segments is not None:
                save_segments_as_mp3(segments=segments, sr=sr, output_dir=segment_folder + "/" + file_class,
                                     base_filename=base_name)
            else:
                print(f"Audio {audio} is shorter then {sec_split} seconds. Ignored")

if process_segments:
    dfs = []
    skipped = []
    errors = []
    subfolders = [x[0] for x in os.walk(segment_folder)]
    for i, folder in enumerate(subfolders):
        val = folder.split("/")
        sample_class = set(val).intersection(set(classes))
        for j, segment in enumerate(glob(folder + "/*.mp3")):
            try:
                y, sr = librosa.load(segment, sr=sr)
                df = extract_features(y=y, sr=sr)
                df["path"] = segment
                df[y_name] = list(sample_class)[0]
                dfs.append(df)
                # plot_mel_spectogram(y)
            except Exception as e:
                skipped.append(segment)
                errors.append(e)
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_csv(output_file, index=False)
    if len(skipped) > 0:
        pd.DataFrame({"path": skipped, "error": errors}).to_csv(skipped_file, index=False)