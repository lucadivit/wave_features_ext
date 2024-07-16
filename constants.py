
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
perc_thr = 0.3

output_file = "dataset_5.csv"
skipped_file = "skipped.csv"
y_name = "label"
path_name = "path"
output_train_fn = "train_result.csv"
output_test_fn = "test_result.csv"
output_validation_fn = "validation_result.csv"
train_set_fn = "train_set.csv"
test_set_fn = "test_set.csv"
validation_set_fn = "validation_set.csv"
summary_fn = "summary.csv"
threshold_fn = "thresholds.json"