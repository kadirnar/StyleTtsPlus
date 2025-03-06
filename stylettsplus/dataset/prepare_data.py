#!/usr/bin/env python3
"""
Data preparation script for StyleTTS2 training.
This script processes wav and text files from kore_wavs and puck_wavs directories
to create train_list.txt and val_list.txt files for StyleTTS2 training.
Uses parallel processing with all available CPU cores for faster processing.
"""

import os
import glob
import tqdm
import soundfile as sf
import librosa
import re
from pathlib import Path
import phonemizer
from phonemizer.backend import EspeakBackend
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import time
import random

def process_text(text, language='en-us'):
    """
    Process text to phonemes using phonemizer
    """
    backend = EspeakBackend(language=language)
    phonemes = backend.phonemize([text], strip=True)[0]
    return phonemes

def resample_audio(input_path, output_path, target_sr=24000):
    """
    Resample audio to target sample rate using CPU
    """
    try:
        y, sr = librosa.load(input_path, sr=None)
        if sr != target_sr:
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

def process_directory(wav_dir, text_dir, output_dir, output_sr, language, speaker_id, num_workers, batch_size):
    """
    Process all files in a directory
    Returns a list of successfully processed entries
    """
    # Get all wav files
    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
    total_files = len(wav_files)
    print(f"Found {total_files} wav files in {wav_dir}")
    
    if total_files == 0:
        print(f"No wav files found in {wav_dir}. Skipping directory.")
        return []
    
    # Process files in parallel
    data_entries = []
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Create a partial function with fixed arguments
    process_func = partial(
        process_file, 
        text_dir=text_dir, 
        output_dir=output_dir, 
        output_sr=output_sr, 
        language=language, 
        speaker_id=speaker_id
    )
    
    # Process files in batches to avoid memory issues
    num_batches = (total_files + batch_size - 1) // batch_size  # Ceiling division
    
    with tqdm.tqdm(total=total_files, desc=f"Processing {os.path.basename(wav_dir)}") as pbar:
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
    
    print(f"{os.path.basename(wav_dir)} - Processed: {processed_count}, Skipped: {skipped_count}, Errors: {error_count}")
    return data_entries

def main():
    # Fixed parameters
    output_dir = "Data"
    output_sr = 24000
    val_ratio = 0.1
    language = "en-us"
    batch_size = 100
    
    # Source directories
    kore_dir = "hf_data/kore_wavs"
    puck_dir = "hf_data/puck_wavs"
    
    # Use all available CPU cores
    num_workers = multiprocessing.cpu_count()
    print(f"Using {num_workers} worker processes (all available CPU cores)")
    
    # Convert relative paths to absolute paths if needed
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../.."))  # Go up two levels to project root
    kore_dir = os.path.join(project_root, kore_dir)
    puck_dir = os.path.join(project_root, puck_dir)
    output_dir = os.path.join(project_root, output_dir) if not os.path.isabs(output_dir) else output_dir
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "wavs"), exist_ok=True)
    
    # Timer start
    start_time = time.time()
    
    # Process both directories
    print("\nProcessing Kore dataset...")
    kore_entries = process_directory(
        kore_dir, 
        kore_dir, 
        output_dir, 
        output_sr, 
        language, 
        "0",  # Use "0" as speaker ID for kore files 
        num_workers, 
        batch_size
    )
    
    print("\nProcessing Puck dataset...")
    puck_entries = process_directory(
        puck_dir, 
        puck_dir, 
        output_dir, 
        output_sr, 
        language, 
        "1",  # Use "1" as speaker ID for puck files
        num_workers, 
        batch_size
    )
    
    # Combine entries
    data_entries = kore_entries + puck_entries
    
    processing_time = time.time() - start_time
    print(f"\nProcessing completed in {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
    
    if len(data_entries) == 0:
        print("No data entries were processed successfully. Please check your input files.")
        return
    
    # Split data into train and validation sets
    random.seed(42)  # For reproducibility
    random.shuffle(data_entries)
    val_size = int(len(data_entries) * val_ratio)
    train_entries = data_entries[val_size:]
    val_entries = data_entries[:val_size]
    
    print(f"\nTotal processed entries: {len(data_entries)}")
    print(f"Train set size: {len(train_entries)}")
    print(f"Validation set size: {len(val_entries)}")
    
    # Create train_list.txt and val_list.txt files
    train_file = os.path.join(output_dir, "train_list.txt")
    val_file = os.path.join(output_dir, "val_list.txt")
    
    # Write train_list.txt
    with open(train_file, 'w', encoding='utf-8') as f:
        for filename, phonemes, spk_id in train_entries:
            f.write(f"{os.path.splitext(filename)[0]}.wav|{phonemes}|{spk_id}\n")
    
    # Write val_list.txt if there are validation entries
    if val_entries:
        with open(val_file, 'w', encoding='utf-8') as f:
            for filename, phonemes, spk_id in val_entries:
                f.write(f"{os.path.splitext(filename)[0]}.wav|{phonemes}|{spk_id}\n")
    
    print(f"Files written to {train_file} and {val_file}")
    print("Data preparation complete!")

if __name__ == "__main__":
    main()