
legitimate_class = "0"
malware_class = "1"
classes = [legitimate_class, malware_class]

input_folder_name_converter = "binaries/"
output_folder_name_converter = "waves/"
segment_folder = "segmented_audio"

freq = 44100
channels = 1
sec_split = 0.8
seed = 42
n_ceps = 14

output_file = "dataset_5.csv"
skipped_file = "skipped.csv"
y_name = "label"
path_name = "path"