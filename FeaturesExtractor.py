import wave, os, uuid, numpy as np, pandas as pd, librosa, shutil, pickle
from gammatone.gtgram import gtgram
from scipy.fftpack import dct
from pydub import AudioSegment
from spafe.utils.preprocessing import SlidingWindow
from spafe.features.bfcc import bfcc
from glob import glob
from Logger import Logger


class FeaturesExtractor:

    def __new__(cls):
        if not hasattr(cls, "_inst"):
            cls._inst = super(FeaturesExtractor, cls).__new__(cls)
            cls.__logger = Logger().get_logger()
            cls.__freq = int(os.environ["FREQ"])
            cls.__channels = int(os.environ["CHANNELS"])
            cls.__n_ceps = int(os.environ["N_CEPS"])
            cls.__axis = int(os.environ["FCC_AXIS"])
            cls.__sec_split = float(os.environ["SEC_SPLIT"])
            cls.__overlap = float(os.environ["AUDIO_OVERLAP"])
            cls.__allowed_0_perc = float(os.environ["ALLOWED_0_PERC"])
            cls.__allowed_0_perc = None if cls.__allowed_0_perc < 0 else cls.__allowed_0_perc
            cls.__pad_shorter = bool(int(os.environ["PAD_SHORTER"]))
            with open(os.environ["CLIPPER_PATH"], 'rb') as file:
                cls.__clipper = pickle.load(file)
            cls.__cols_to_drop = ["mfcc_mean_0", "mfcc_mean_2", "mfcc_mean_4", "mfcc_mean_5",
                                  "mfcc_mean_6", "mfcc_mean_8", "mfcc_mean_10", "bfcc_mean_4",
                                  "bfcc_mean_12"]
            cls.__columns = ['mfcc_mean_1', 'mfcc_mean_3', 'mfcc_mean_7', 'mfcc_mean_9', 'mfcc_mean_11',
                             'mfcc_mean_12', 'mfcc_mean_13', 'gfcc_mean_0', 'gfcc_mean_1', 'gfcc_mean_2',
                             'gfcc_mean_3', 'gfcc_mean_4', 'gfcc_mean_5', 'gfcc_mean_6', 'gfcc_mean_7',
                             'gfcc_mean_8', 'gfcc_mean_9', 'gfcc_mean_10', 'gfcc_mean_11', 'gfcc_mean_12',
                             'gfcc_mean_13', 'bfcc_mean_0', 'bfcc_mean_1', 'bfcc_mean_2', 'bfcc_mean_3',
                             'bfcc_mean_5', 'bfcc_mean_6', 'bfcc_mean_7', 'bfcc_mean_8', 'bfcc_mean_9',
                             'bfcc_mean_10', 'bfcc_mean_11', 'bfcc_mean_13']
        return cls._inst

    def __convert_to_wav(self, binary_file_path, output_wave_path, sample_rate=44100, channels=1) -> None:
        """
          Converts a binary file to a WAV waves file with specified sample rate and channels.

          Args:
              binary_file_path (str): Path to the binary file.
              output_wave_path (str): Path to save the generated WAV file.
              sample_rate (int, optional): Sample rate of the waves (default: 44100).
              channels (int, optional): Number of waves channels (default: 1 for mono).
        """
        # Open the binary file in read-binary mode
        with open(binary_file_path, 'rb') as binary_file:
            # Read the binary data
            binary_data = binary_file.read()

        # Define WAV header parameters
        chunk_size = 4  # Number of bytes per sample
        fmt = ' '.join(['1' for _ in range(channels)])  # Format code (1 for PCM)
        subchunk1_size = (16 * channels)  # Size of PCM subchunk
        audio_length = len(binary_data)  # Length of waves data in bytes
        subchunk2_size = audio_length  # Size of waves data subchunk

        # Calculate number of frames (samples) based on channels
        num_frames = int(audio_length / (chunk_size * channels))

        # Create WAV header bytes
        wave_header = b''.join([
            # RIFF Chunk
            b'RIFF',  # Chunk ID
            self.__wav_int(chunk_size + 36),  # Chunk size (36 for header + data)
            b'WAVE',  # Format

            # fmt subchunk
            b'fmt ',  # Subchunk ID
            self.__wav_int(subchunk1_size),  # Subchunk size
            self.__wav_short(1),  # Audio format (1 for PCM)
            self.__wav_short(channels),  # Number of channels
            self.__wav_int(sample_rate),  # Sample rate
            self.__wav_int(sample_rate * channels * chunk_size),  # Byte rate
            self.__wav_short(chunk_size * channels),  # Block align
            self.__wav_short(chunk_size * 8),  # Bits per sample (8 bits for PCM)

            # data subchunk
            b'data',
            self.__wav_int(subchunk2_size)  # Subchunk size (waves data size)
        ])

        # Combine header and binary data
        wave_data = wave_header + binary_data

        # Open the output WAV file in write-binary mode
        with wave.open(output_wave_path, 'wb') as wave_file:
            # Set WAV parameters
            wave_file.setnchannels(channels)
            wave_file.setsampwidth(chunk_size)
            wave_file.setframerate(sample_rate)

            # Write WAV data
            wave_file.writeframes(wave_data)


    # Helper functions to convert integers to little-endian bytes
    def __wav_int(self, value) -> bytes:
        return value.to_bytes(4, byteorder='little', signed=True)

    def __wav_short(self, value) -> bytes:
        return value.to_bytes(2, byteorder='little', signed=True)

    def __delete_directory(self, directory: str) -> None:
        try:
            shutil.rmtree(directory)
            self.__logger.info(f"Directory {directory} Deleted.")
        except FileNotFoundError as e:
            self.__logger.exception(f"Directory {directory} Not Found.")
            raise e
        except PermissionError as e:
            self.__logger.exception(f"You Have Not Permission To Delete {directory}.")
            raise e
        except Exception as e:
            self.__logger.exception(f"Problem During {directory} Deletion: {e}")
            raise e

    def __delete_file(self, file_path: str) -> None:
        try:
            os.remove(file_path)
            self.__logger.info(f"File {file_path} Deleted.")
        except FileNotFoundError as e:
            self.__logger.exception(f"File {file_path} Not Found.")
            raise e
        except PermissionError as e:
            self.__logger.exception(f"You Have Not Permission To Delete {file_path}.")
            raise e
        except Exception as e:
            self.__logger.exception(f"Problem During {file_path} Deletion: {e}")
            raise e

    def __get_wav_duration(self, file_path: str) -> float:
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
        return duration

    def __compute_0_percentage(self, y: np.ndarray) -> float:
        total_elements = y.size
        zero_count = np.count_nonzero(y == 0)
        zero_percentage = zero_count / total_elements
        return zero_percentage

    def __load_and_normalize_audio(self, file_path: str, sr: int =44100, max_length: int =40000):
        # Normalizzazione tramite padding
        y, sr = librosa.load(file_path, sr=sr)
        len_y = len(y)
        self.__logger.info(f"Audio is len {len_y}")
        if len_y < max_length:
            self.__logger.info("Padding...")
            y = np.pad(y, (0, max_length - len(y)), mode='constant')
        else:
            y = y[:max_length]
        return y, sr

    # Normalizzazione tramite segmentazione
    def __load_and_segment_audio(self, file_path: str, segment_duration: float = 10., overlap: float = 0.5, sr: int = 44100,
                                 pad_shorter: bool = True, allowed_0_perc: float = None, check_only_padded: bool = True):

        def add_to_list(segments_list, segment, idx, allowed_0_perc, check_only_padded, is_padded):
            if check_only_padded:
                if is_padded:
                    if allowed_0_perc is None:
                        segments_list.append(segment)
                    else:
                        perc_of_0 = self.__compute_0_percentage(y=segment)
                        if perc_of_0 <= allowed_0_perc:
                            segments_list.append(segment)
                        else:
                            self.__logger.info(f"Segment {idx} Of {file_path} Dropped Cause Too Many 0s. "
                                               f"{perc_of_0} on {allowed_0_perc} Allowed")
                else:
                    segments_list.append(segment)
            else:
                if allowed_0_perc is None:
                    segments_list.append(segment)
                else:
                    perc_of_0 = self.__compute_0_percentage(y=segment)
                    if perc_of_0 <= allowed_0_perc:
                        segments_list.append(segment)
                    else:
                        self.__logger.info(f"Segment {idx} Of {file_path} Dropped Cause Too Many 0s. "
                                           f"{perc_of_0} on {allowed_0_perc} Allowed")
            return segments_list

        y, sr = librosa.load(file_path, sr=sr)
        audio_duration = self.__get_wav_duration(file_path=file_path)
        self.__logger.info(f"Audio {file_path} Is {audio_duration} sec. Len")
        is_padded = False
        if audio_duration < segment_duration:
            if pad_shorter:
                self.__logger.info(f"Padding Audio {file_path} To Desired Duration")
                y = librosa.util.fix_length(data=y, size=int(segment_duration * sr))
                is_padded = True
            else:
                self.__logger.info(f"Audio {file_path} Too Short, Skipped.")
                return None, sr
        segment_length = int(segment_duration * sr)
        step_length = int(segment_length * (1 - overlap))
        segments = []
        idx = 0
        for start in range(0, len(y), step_length):
            end = start + segment_length
            if end <= len(y):
                segment = y[start:end]
                segments = add_to_list(segments_list=segments, segment=segment, idx=idx, allowed_0_perc=allowed_0_perc,
                                       check_only_padded=check_only_padded, is_padded=is_padded)
            else:
                segment = np.pad(y[start:end], (0, segment_length - len(y[start:end])), mode='constant')
                segments = add_to_list(segments_list=segments, segment=segment, idx=idx, allowed_0_perc=allowed_0_perc,
                                       check_only_padded=check_only_padded, is_padded=is_padded)
                break
            idx += 1
        segments = None if len(segments) == 0 else segments
        return segments, sr

    def __save_segments_as_mp3(self, segments: np.ndarray, sr: int, output_dir: str, base_filename: str):
        os.makedirs(f"{output_dir}/{base_filename}", exist_ok=True)
        for i, segment in enumerate(segments):
            # Convert numpy array to waves segment
            audio_segment = AudioSegment(
                segment.tobytes(),
                frame_rate=sr,
                sample_width=segment.dtype.itemsize,
                channels=self.__channels
            )
            # Save as MP3
            output_path = f"{output_dir}/{base_filename}/{base_filename}_segment_{i + 1}.mp3"
            audio_segment.export(output_path, format="mp3")
            self.__logger.info(f"Saved segment {i + 1} to {output_path}")

    def __compute_mfcc_features(self, y: np.ndarray, sr: int):
        '''
        axis=1: Ogni elemento del vettore rappresenta la media dei valori di un singolo coefficiente MFCC attraverso tutto il segnale waves.
        Fornisce un'idea generale della distribuzione dei coefficienti MFCC per l'intero waves. Dim Output = n_mfcc
        axis=0: Ogni elemento del vettore rappresenta la media dei valori di tutti i coefficienti MFCC per un singolo frame temporale.
        Fornisce un'idea di come le caratteristiche spettrali cambiano nel tempo. Dim Output = n_t.
        Parametro Tunabile
        '''
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.__n_ceps)
        mfcc_mean = np.mean(mfccs, axis=self.__axis)
        # mfcc_std = np.std(mfccs, axis=0)
        cols_mfcc = [f"mfcc_mean_{i}" for i, _ in enumerate(mfcc_mean)]
        mfcc_df = pd.DataFrame([mfcc_mean])
        mfcc_df.columns = cols_mfcc
        return mfcc_df

    def __compute_gfcc_features(self, y: np.ndarray, sr: int):
        # Metodo 1
        n_filters = 64  # Numero di filtri Gammatone
        win_size = 0.05  # Durata della finestra in secondi
        hop_size = 0.02  # Durata dello step in secondi
        f_min = 50  # Dal digramma di mel le frequenze sono molto alte quindi si può alzare questo valore con perdita minima
        gammatone_spec = gtgram(y, sr, win_size, hop_size, n_filters, f_min)
        log_gammatone_spec = np.log(gammatone_spec + 1e-6)
        gfccs = dct(log_gammatone_spec, type=2, axis=0, norm='ortho')[:self.__n_ceps]
        gfcc_mean = np.mean(gfccs, axis=self.__axis)
        # gfcc_std = np.std(gfccs, axis=0)
        cols_gfcc = [f"gfcc_mean_{i}" for i, _ in enumerate(gfcc_mean)]
        gfcc_df = pd.DataFrame([gfcc_mean])
        gfcc_df.columns = cols_gfcc

        # Metodo 2
        # gfccs = gfcc(y, fs=sr, num_ceps=13)
        # print(gfccs.shape)
        return gfcc_df

    def __compute_bfcc_features(self, y: np.ndarray, sr: int):
        bfccs = bfcc(y, fs=sr, pre_emph=1, pre_emph_coeff=0.97, num_ceps=self.__n_ceps,
                     window=SlidingWindow(0.03, 0.015, "hamming"), low_freq=0, nfilts=50,
                     high_freq=16000, normalize="mvn").T
        bfcc_mean = np.mean(bfccs, axis=self.__axis)
        cols_bfcc = [f"bfcc_mean_{i}" for i, _ in enumerate(bfcc_mean)]
        bfcc_df = pd.DataFrame([bfcc_mean])
        bfcc_df.columns = cols_bfcc
        return bfcc_df

    # Funzione per estrarre caratteristiche
    def __extract_features(self, y: np.ndarray, sr: int) -> pd.DataFrame:
        mfcc_df = self.__compute_mfcc_features(y=y, sr=sr)
        gfcc_df = self.__compute_gfcc_features(y=y, sr=sr)
        bfcc_df = self.__compute_bfcc_features(y=y, sr=sr)
        result = pd.concat([mfcc_df, gfcc_df, bfcc_df], axis=1)
        return result

    def create_features(self, binary_file_path: str) -> pd.DataFrame:
        dfs = []
        unique_name = str(uuid.uuid4())
        wave_path = unique_name + ".wav"
        mp3_path = unique_name + "/"
        self.__logger.info(f"Processing File {binary_file_path}")
        self.__convert_to_wav(binary_file_path=binary_file_path, output_wave_path=wave_path,
                              sample_rate=self.__freq, channels=self.__channels)
        self.__logger.info(f"Binary {binary_file_path} Converted To Wave")
        base_name = wave_path.split("/")[-1].split(".")[0]
        segments, _ = self.__load_and_segment_audio(file_path=wave_path, sr=self.__freq,
                                                    segment_duration=self.__sec_split,
                                                    overlap=self.__overlap, pad_shorter=True,
                                                    allowed_0_perc=self.__allowed_0_perc)
        if segments is not None:
            self.__save_segments_as_mp3(segments=segments, sr=self.__freq, output_dir=mp3_path,
                                        base_filename=base_name)
            self.__logger.info(f"Converted File {binary_file_path} To mp3s.")
        else:
            raise Exception(f"Problem With {binary_file_path} MP3 Processing")

        subfolders = [x[0] for x in os.walk(mp3_path)]
        for i, folder in enumerate(subfolders):
            files = glob(folder + "/*.mp3")
            len_files = len(files)
            for j, segment in enumerate(files):
                print(f"Processing File {segment}. {j + 1}/{len_files}")
                try:
                    y, sr = librosa.load(segment, sr=self.__freq)
                    df = self.__extract_features(y=y, sr=self.__freq)
                    dfs.append(df)
                except Exception as e:
                    self.__logger.exception(f"Error During Features Extraction: {e}")
        self.__delete_file(file_path=wave_path)
        self.__delete_directory(directory=mp3_path)
        df = pd.concat(dfs, ignore_index=True)
        df = self.__process_dataset(df=df)
        return df

    def __process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=self.__cols_to_drop)
        df = self.__clipper.transform(df)
        df = df[self.__columns]
        return df
