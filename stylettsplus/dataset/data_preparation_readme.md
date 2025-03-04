# Data Preparation for StyleTTS2

This guide explains how to prepare your data for training a StyleTTS2 model from scratch.

## Data Format Requirements

StyleTTS2 requires data in the following format:
- Audio files in WAV format, resampled to 24kHz
- A text file (`train_list.txt` and `val_list.txt`) containing entries in the format:
  ```
  filename.wav|phonemized_text|speaker_id
  ```

## Prerequisites

Install the required packages:

```bash
pip install phonemizer librosa soundfile tqdm numpy torch
sudo apt-get install espeak-ng
```

## Using the Data Preparation Script

The `prepare_data.py` script will:
1. Convert your text to phonemes using the phonemizer library
2. Resample your audio files to 24kHz
3. Create the required train_list.txt and val_list.txt files

### Basic Usage

```bash
python prepare_data.py --wav_dir /path/to/wav/files --text_dir /path/to/text/files
```

### All Options

```bash
python prepare_data.py --wav_dir /path/to/wav/files --text_dir /path/to/text/files --output_dir Data --output_sr 24000 --val_ratio 0.1 --speaker_id 0 --language en-us
```

- `--wav_dir`: Directory containing WAV files
- `--text_dir`: Directory containing text files (should have the same base filename as the WAV files)
- `--output_dir`: Output directory (default: "Data")
- `--output_sr`: Output sample rate (default: 24000)
- `--val_ratio`: Validation set ratio (default: 0.1)
- `--speaker_id`: Speaker ID for single speaker dataset (default: "0")
- `--language`: Language for phonemization (default: "en-us")

## Manual Data Preparation

If you prefer to prepare the data manually:

1. Resample your WAV files to 24kHz:
   ```bash
   for file in *.wav; do
     ffmpeg -i "$file" -ar 24000 -ac 1 "resampled_$file"
   done
   ```

2. Create a directory structure:
   ```
   Data/
   ├── wavs/
   │   ├── file1.wav
   │   ├── file2.wav
   │   └── ...
   ├── train_list.txt
   └── val_list.txt
   ```

3. Phonemize your text and create the list files manually in the format:
   ```
   filename.wav|phonemized_text|speaker_id
   ```

## Training Your Model

After preparing your data, you can train your StyleTTS2 model:

1. First stage training:
   ```bash
   accelerate launch train_first.py --config_path ./Configs/config.yml
   ```

2. Second stage training:
   ```bash
   python train_second.py --config_path ./Configs/config.yml
   ```

Make sure to update the `config.yml` file to point to your data directories.
