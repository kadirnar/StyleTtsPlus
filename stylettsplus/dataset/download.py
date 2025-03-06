import os
import shutil
from datasets import load_dataset
from tqdm import tqdm
import soundfile as sf
import numpy as np

def main():
    # Create output directories
    os.makedirs("hf_data/kore_wavs", exist_ok=True)
    os.makedirs("hf_data/puck_wavs", exist_ok=True)
    
    # Load the dataset from Hugging Face
    print("Loading dataset...")
    dataset = load_dataset("shb777/gemini-flash-2.0-speech")
    
    # Process the dataset
    print("Processing dataset...")
    for split in dataset.keys():
        print(f"Processing {split} split...")
        for idx, item in enumerate(tqdm(dataset[split])):
            # Extract text content
            text = item.get('text')
            
            if text is not None:
                # Generate base filename
                base_idx = f"{split}_{idx:06d}"
                
                # Process kore audio
                kore_audio = item.get('kore')
                if kore_audio is not None:
                    # Create kore-specific filename
                    kore_filename = f"kore_{base_idx}"
                    
                    # Save kore audio as WAV
                    wav_path = os.path.join("hf_data/kore_wavs", f"{kore_filename}.wav")
                    sf.write(wav_path, kore_audio['array'], kore_audio['sampling_rate'])
                    
                    # Save text to a separate text file with the same name
                    text_path = os.path.join("hf_data/kore_wavs", f"{kore_filename}.txt")
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                
                # Process puck audio
                puck_audio = item.get('puck')
                if puck_audio is not None:
                    # Create puck-specific filename
                    puck_filename = f"puck_{base_idx}"
                    
                    # Save puck audio as WAV
                    wav_path = os.path.join("hf_data/puck_wavs", f"{puck_filename}.wav")
                    sf.write(wav_path, puck_audio['array'], puck_audio['sampling_rate'])
                    
                    # Save text to a separate text file with the same name
                    text_path = os.path.join("hf_data/puck_wavs", f"{puck_filename}.txt")
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(text)
    
    print(f"Audio files and text files saved to hf_data/kore_wavs/ and hf_data/puck_wavs/")
    
    # Clean up unnecessary files
    print("Cleaning up unnecessary files...")
    cache_dir = os.path.join("hf_data", ".cache")
    git_dir = os.path.join("hf_data", ".git")
    git_attributes = os.path.join("hf_data", ".gitattributes")
    license_file = os.path.join("hf_data", "LICENSE")
    
    # Remove .cache directory if it exists
    if os.path.exists(cache_dir):
        print(f"Removing {cache_dir}")
        shutil.rmtree(cache_dir)
    
    # Remove .git directory if it exists
    if os.path.exists(git_dir):
        print(f"Removing {git_dir}")
        shutil.rmtree(git_dir)
    
    # Remove .gitattributes file if it exists
    if os.path.exists(git_attributes):
        print(f"Removing {git_attributes}")
        os.remove(git_attributes)
    
    # Remove LICENSE file if it exists
    if os.path.exists(license_file):
        print(f"Removing {license_file}")
        os.remove(license_file)
    
    print("Cleanup completed.")

if __name__ == "__main__":
    main()
