import os, librosa, numpy as np, pandas as pd
from pydub import AudioSegment
from glob import glob

# Normalizzazione tramide padding
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
    return segments, sr

def save_segments_as_mp3(segments, sr, output_dir, base_filename):
    os.makedirs(f"{output_dir}/{base_filename}", exist_ok=True)
    for i, segment in enumerate(segments):
        # Convert numpy array to audio segment
        audio_segment = AudioSegment(
            segment.tobytes(),
            frame_rate=sr,
            sample_width=segment.dtype.itemsize,
            channels=1
        )
        # Save as MP3
        output_path = f"{output_dir}/{base_filename}/{base_filename}_segment_{i+1}.mp3"
        audio_segment.export(output_path, format="mp3")
        print(f"Saved segment {i+1} to {output_path}")

# Funzione per estrarre caratteristiche
def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
    '''
    axis=1: Ogni elemento del vettore rappresenta la media dei valori di un singolo coefficiente MFCC attraverso tutto il segnale audio. 
    Fornisce un'idea generale della distribuzione dei coefficienti MFCC per l'intero audio. Dim Output = n_mfcc
    axis=0: Ogni elemento del vettore rappresenta la media dei valori di tutti i coefficienti MFCC per un singolo frame temporale. 
    Fornisce un'idea di come le caratteristiche spettrali cambiano nel tempo. Dim Output = n_t
    '''
    mfcc_mean = np.mean(mfccs, axis=0)
    # mfcc_std = np.std(mfccs, axis=0)
    cols_mfcc = [f"mfcc_mean_{i}" for i, _ in enumerate(mfcc_mean)]
    mfcc_df = pd.DataFrame([mfcc_mean])
    mfcc_df.columns = cols_mfcc
    print(mfcc_df)
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

sr = 22.05*10e3
segment_folder = "segmented_audio"
audio_folder = "audio"
split_audio = False
process_segments = True
if split_audio:
    for audio in glob(f"{audio_folder}/*.wav"):
        print(f"Splitting {audio}")
        base_name = audio.split("/")[-1].split(".")[0]
        segments, _ = load_and_segment_audio(file_path=audio, sr=sr, segment_duration=5, overlap=0.5)
        save_segments_as_mp3(segments=segments, sr=sr, output_dir=segment_folder, base_filename=base_name)

if process_segments:
    subfolders = [x[0] for x in os.walk(segment_folder)]
    for folder in subfolders:
        for segment in glob(folder + "/*.mp3"):
            y, sr = librosa.load(segment, sr=sr)
            extract_features(y=y, sr=sr)
            exit(0)



