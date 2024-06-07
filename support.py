import wave
from glob import glob
from constants import classes, output_folder_name_converter
import pandas as pd


def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
    return duration


def create_dataframe_duration(save=True):
    res = {"class": [], "fn": [], "duration": []}
    for file_class in classes:
        for file_path in glob(f"{output_folder_name_converter}{file_class}/*.wav"):
            fn = file_path.split("/")[-1]
            duration = get_wav_duration(file_path)
            res["class"].append(file_class)
            res["fn"].append(fn)
            res["duration"].append(duration)
    df = pd.DataFrame(res)
    if save:
        df.to_csv("durations.csv", index=False)
    return df

# create_dataframe_duration()
