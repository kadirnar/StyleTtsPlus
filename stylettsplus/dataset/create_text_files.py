import os
from datasets import load_dataset
from tqdm import tqdm

def main():
    # Load the dataset from Hugging Face to get the text data
    print("Loading dataset...")
    dataset = load_dataset("shb777/gemini-flash-2.0-speech")
    
    # Process the dataset
    print("Creating text files for each wav file...")
    for split in dataset.keys():
        print(f"Processing {split} split...")
        for idx, item in enumerate(tqdm(dataset[split])):
            # Extract text
            text = item.get('text')
            
            if text is not None:
                # Generate a filename
                filename = f"{split}_{idx:06d}"
                
                # Check if wav file exists
                wav_path = os.path.join("hf_data/kore_wavs", f"{filename}.wav")
                if os.path.exists(wav_path):
                    # Save text to a separate text file with the same name
                    text_path = os.path.join("hf_data/kore_wavs", f"{filename}.txt")
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(text)
    
    print("Text files created successfully.")

if __name__ == "__main__":
    main()
