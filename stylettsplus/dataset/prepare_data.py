#!/usr/bin/env python3
"""
Data preparation script for StyleTTS2 training.
This script processes wav and text files to create the required format for StyleTTS2 training.
Uses parallel processing and GPU acceleration for faster processing.
"""

import os
import argparse
import glob
import tqdm
import soundfile as sf
import librosa
import re
from pathlib import Path
import phonemizer
from phonemizer.backend import EspeakBackend
import torch
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import time

# Check if CUDA is available for GPU acceleration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def process_text(text, language='en-us'):
    """
    Process text to phonemes using phonemizer
    """
    backend = EspeakBackend(language=language)
    phonemes = backend.phonemize([text], strip=True)[0]
    return phonemes

def resample_audio(input_path, output_path, target_sr=24000):
    """
    Resample audio to target sample rate
    """
    try:
        y, sr = librosa.load(input_path, sr=None)
        if sr != target_sr:
            # Use GPU for resampling if available
            if DEVICE.type == 'cuda' and torch.cuda.is_available():
                # Convert to tensor and move to GPU
                y_tensor = torch.tensor(y, device=DEVICE).float()
                # Resample using torchaudio if available, otherwise fallback to librosa
                try:
                    import torchaudio
                    import torchaudio.functional as F
                    y_resampled = F.resample(y_tensor, sr, target_sr)
                    y = y_resampled.cpu().numpy()
                except (ImportError, AttributeError):
                    # Fallback to librosa
                    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            else:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sf.write(output_path, y, target_sr)
        return True
    except Exception as e:
        print(f"Error resampling {input_path}: {str(e)}")
        return False

def clean_text(text):
    """
    Clean text by removing special characters and extra whitespace
    """
    # Remove special characters like % at the end
    text = re.sub(r'[%]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_file(wav_path, text_dir, output_dir, output_sr, language, speaker_id):
    """
    Process a single audio file and its corresponding text file
    Returns a tuple (success, filename, phonemes, speaker_id) or (False, filename, error_message, None)
    """
    filename = os.path.basename(wav_path)
    file_id = os.path.splitext(filename)[0]
    
    # Find corresponding text file
    text_path = os.path.join(text_dir, f"{file_id}.txt")
    if not os.path.exists(text_path):
        return (False, filename, f"No text file found for {filename}", None)
    
    try:
        # Read and process text
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # Clean text
        text = clean_text(text)
        
        # Skip empty text
        if not text:
            return (False, filename, f"Empty text for {filename}", None)
        
        # Convert text to phonemes
        phonemes = process_text(text, language=language)
        
        # Resample and save audio
        output_wav_path = os.path.join(output_dir, "wavs", filename)
        if not resample_audio(wav_path, output_wav_path, target_sr=output_sr):
            return (False, filename, f"Failed to resample audio for {filename}", None)
        
        # Return success
        return (True, filename, phonemes, speaker_id)
        
    except Exception as e:
        return (False, filename, str(e), None)

def main():
    parser = argparse.ArgumentParser(description="Prepare data for StyleTTS2 training")
    parser.add_argument("--wav_dir", type=str, default="hf_data/kore_wavs", help="Directory containing wav files")
    parser.add_argument("--text_dir", type=str, default="hf_data/kore_wavs", help="Directory containing text files")
    parser.add_argument("--output_dir", type=str, default="Data", help="Output directory")
    parser.add_argument("--output_sr", type=int, default=24000, help="Output sample rate")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--speaker_id", type=str, default="0", help="Speaker ID for single speaker dataset")
    parser.add_argument("--language", type=str, default="en-us", help="Language for phonemization")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes (default: CPU count)")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Set number of workers
    num_workers = args.num_workers if args.num_workers is not None else multiprocessing.cpu_count()
    print(f"Using {num_workers} worker processes")
    
    # Convert relative paths to absolute paths if needed
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wav_dir = os.path.join(base_dir, args.wav_dir) if not os.path.isabs(args.wav_dir) else args.wav_dir
    text_dir = os.path.join(base_dir, args.text_dir) if not os.path.isabs(args.text_dir) else args.text_dir
    output_dir = os.path.join(base_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "wavs"), exist_ok=True)
    
    # Get all wav files
    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
    total_files = len(wav_files)
    print(f"Found {total_files} wav files in {wav_dir}")
    
    if total_files == 0:
        print(f"No wav files found in {wav_dir}. Please check the directory path.")
        return
    
    # Process files in parallel
    start_time = time.time()
    data_entries = []
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Create a partial function with fixed arguments
    process_func = partial(
        process_file, 
        text_dir=text_dir, 
        output_dir=output_dir, 
        output_sr=args.output_sr, 
        language=args.language, 
        speaker_id=args.speaker_id
    )
    
    # Process files in batches to avoid memory issues
    batch_size = args.batch_size
    num_batches = (total_files + batch_size - 1) // batch_size  # Ceiling division
    
    with tqdm.tqdm(total=total_files, desc="Processing files") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_files)
            batch_files = wav_files[start_idx:end_idx]
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_func, wav_path) for wav_path in batch_files]
                
                for future in as_completed(futures):
                    result = future.result()
                    success, filename, result_data, speaker_id = result
                    
                    if success:
                        data_entries.append((filename, result_data, speaker_id))
                        processed_count += 1
                    else:
                        if "No text file found" in result_data or "Empty text" in result_data:
                            skipped_count += 1
                        else:
                            error_count += 1
                            print(f"Error processing {filename}: {result_data}")
                    
                    pbar.update(1)
    
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
    
    if len(data_entries) == 0:
        print("No data entries were processed successfully. Please check your input files.")
        return
    
    # Split into train and validation sets
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(data_entries)
    val_size = max(1, int(len(data_entries) * args.val_ratio))
    val_entries = data_entries[:val_size]
    train_entries = data_entries[val_size:]
    
    # Write train and validation lists
    with open(os.path.join(output_dir, "train_list.txt"), 'w', encoding='utf-8') as f:
        for filename, phonemes, speaker_id in train_entries:
            f.write(f"{filename}|{phonemes}|{speaker_id}\n")
    
    with open(os.path.join(output_dir, "val_list.txt"), 'w', encoding='utf-8') as f:
        for filename, phonemes, speaker_id in val_entries:
            f.write(f"{filename}|{phonemes}|{speaker_id}\n")
    
    print(f"Processed {processed_count} files successfully")
    print(f"Skipped {skipped_count} files")
    print(f"Encountered errors in {error_count} files")
    print(f"Created train set with {len(train_entries)} entries")
    print(f"Created validation set with {len(val_entries)} entries")
    print(f"Data saved to {output_dir}")
    print("Data preparation completed!")

if __name__ == "__main__":
    main()
