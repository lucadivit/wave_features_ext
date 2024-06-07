import os
import shutil


def is_executable(file_path):
    return os.path.isfile(file_path) and os.access(file_path, os.X_OK)


def copy_executables(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for file_name in os.listdir(src_dir):
        full_file_name = os.path.join(src_dir, file_name)

        if is_executable(full_file_name):
            try:
                shutil.copy(full_file_name, dst_dir)
                print(f"File binario copiato: {full_file_name}")
            except Exception as e:
                print(f"Errore durante la copia del file {full_file_name}: {e}")


# Esempio di utilizzo
src_directory = '/bin'
legitimate_class = "0"
dest_folder = "/home/lucadivit/PycharmProjects/wave_features_ext/binaries/"
dst_directory = f'{dest_folder}{legitimate_class}/'

copy_executables(src_directory, dst_directory)
