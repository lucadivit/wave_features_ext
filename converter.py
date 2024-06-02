import wave

def convert_to_wave(binary_file_path, output_wave_path, sample_rate=44100, channels=1):
  """
  Converts a binary file to a WAV audio file with specified sample rate and channels.

  Args:
      binary_file_path (str): Path to the binary file.
      output_wave_path (str): Path to save the generated WAV file.
      sample_rate (int, optional): Sample rate of the audio (default: 44100).
      channels (int, optional): Number of audio channels (default: 1 for mono).
  """
  # Open the binary file in read-binary mode
  with open(binary_file_path, 'rb') as binary_file:
    # Read the binary data
    binary_data = binary_file.read()

  # Define WAV header parameters
  chunk_size = 4  # Number of bytes per sample
  fmt = ' '.join(['1' for _ in range(channels)])  # Format code (1 for PCM)
  subchunk1_size = (16 * channels)  # Size of PCM subchunk
  audio_length = len(binary_data)  # Length of audio data in bytes
  subchunk2_size = audio_length  # Size of audio data subchunk

  # Calculate number of frames (samples) based on channels
  num_frames = int(audio_length / (chunk_size * channels))

  # Create WAV header bytes
  wave_header = b''.join([
      # RIFF Chunk
      b'RIFF',  # Chunk ID
      wav_int(chunk_size + 36),  # Chunk size (36 for header + data)
      b'WAVE',  # Format

      # fmt subchunk
      b'fmt ',  # Subchunk ID
      wav_int(subchunk1_size),  # Subchunk size
      wav_short(1),  # Audio format (1 for PCM)
      wav_short(channels),  # Number of channels
      wav_int(sample_rate),  # Sample rate
      wav_int(sample_rate * channels * chunk_size),  # Byte rate
      wav_short(chunk_size * channels),  # Block align
      wav_short(chunk_size * 8),  # Bits per sample (8 bits for PCM)

      # data subchunk
      b'data',
      wav_int(subchunk2_size)  # Subchunk size (audio data size)
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
def wav_int(value):
  return value.to_bytes(4, byteorder='little', signed=True)

def wav_short(value):
  return value.to_bytes(2, byteorder='little', signed=True)

# Example usage
binary_file_path = "audio/song.wav"
output_wave_path = "output2.wav"
convert_to_wave(binary_file_path, output_wave_path)

print("Converted binary file to WAV successfully!")

