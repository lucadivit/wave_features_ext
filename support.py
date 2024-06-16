import wave
from glob import glob
from constants import classes, output_folder_name_converter, sec_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

df = create_dataframe_duration(save=False)
time = sec_split
print("CLASS 0 TOTAL")
print(df[df["class"] == "0"].describe())
print(f"CLASS 0 TOTAL >= {time}")
print(df[(df["class"] == "0") & (df["duration"] >= time)].describe())

print("CLASS 1 TOTAL")
print(df[df["class"] == "1"].describe())
print(f"CLASS 1 TOTAL >= {time}")
print(df[(df["class"] == "1") & (df["duration"] >= time)].describe())

class_0 = df[df["class"] == "0"]["duration"].values
class_1 = df[df["class"] == "1"]["duration"].values

class_0_time = df[(df["class"] == "0") & (df["duration"] >= time)]["duration"].values
class_1_time = df[(df["class"] == "1") & (df["duration"] >= time)]["duration"].values

# sns.kdeplot(class_0, fill=True, color='blue', label='LEGITIMATE')
# sns.kdeplot(class_1, fill=True, color='red', label='MALWARE')
sns.kdeplot(class_0_time, fill=True, color='green', label='LEGITIMATE')
sns.kdeplot(class_1_time, fill=True, color='yellow', label='MALWARE')

plt.xlabel('Duration')
plt.ylabel('Density')

plt.show()
