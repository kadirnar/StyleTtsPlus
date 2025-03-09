import os
import soundfile as sf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import shutil

def calculate_audio_duration(audio_path):
    """
    Calculate the duration of an audio file in seconds.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Duration in seconds
    """
    audio_info = sf.info(audio_path)
    return audio_info.duration

def analyze_directory(directory, delete_long_files=False):
    """
    Analyze all audio files in a directory and calculate their durations.
    Optionally delete files longer than 30 seconds.
    
    Args:
        directory: Directory containing audio files
        delete_long_files: If True, delete files longer than 30 seconds
        
    Returns:
        List of durations in seconds
        Dictionary with statistics
        List of deleted files (if delete_long_files is True)
    """
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return [], {}, []
    
    # Get all WAV files in the directory
    wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    
    if not wav_files:
        print(f"No WAV files found in {directory}.")
        return [], {}, []
    
    print(f"Analyzing {len(wav_files)} audio files in {directory}...")
    
    # Calculate durations
    durations = []
    deleted_files = []
    
    for wav_file in tqdm(wav_files):
        audio_path = os.path.join(directory, wav_file)
        duration = calculate_audio_duration(audio_path)
        durations.append(duration)
        
        # Delete files longer than 30 seconds if requested
        if delete_long_files and duration > 30.0:
            # Find corresponding text file (same base name, different extension)
            base_name = os.path.splitext(wav_file)[0]
            text_file = base_name + ".txt"
            text_path = os.path.join(directory.replace("_wavs", "_texts"), text_file)
            
            # Delete audio file
            os.remove(audio_path)
            deleted_files.append(audio_path)
            
            # Delete text file if it exists
            if os.path.exists(text_path):
                os.remove(text_path)
                deleted_files.append(text_path)
            
            print(f"Deleted files: {audio_path} and {text_path} (duration: {duration:.2f}s)")
    
    # If we deleted files, recompute durations list
    if delete_long_files and deleted_files:
        durations = [calculate_audio_duration(os.path.join(directory, f)) 
                    for f in os.listdir(directory) if f.endswith('.wav')]
    
    # Calculate statistics
    if durations:
        stats = {
            'count': len(durations),
            'min': min(durations) if durations else 0,
            'max': max(durations) if durations else 0,
            'mean': np.mean(durations) if durations else 0,
            'median': np.median(durations) if durations else 0,
            'std': np.std(durations) if durations else 0,
        }
        
        # Count files by duration range
        duration_counts = defaultdict(int)
        for duration in durations:
            # Round to nearest 0.5 second
            rounded_duration = round(duration * 2) / 2
            duration_counts[rounded_duration] += 1
        
        stats['duration_counts'] = dict(sorted(duration_counts.items()))
        
        # Count files exceeding 7 seconds
        exceeding_files_7s = [d for d in durations if d > 7.0]
        stats['exceeding_count_7s'] = len(exceeding_files_7s)
        stats['exceeding_percentage_7s'] = (len(exceeding_files_7s) / len(durations) * 100) if durations else 0
        
        # Count files exceeding 30 seconds
        exceeding_files_30s = [d for d in durations if d > 30.0]
        stats['exceeding_count_30s'] = len(exceeding_files_30s)
        stats['exceeding_percentage_30s'] = (len(exceeding_files_30s) / len(durations) * 100) if durations else 0
    else:
        stats = {
            'count': 0,
            'min': 0,
            'max': 0,
            'mean': 0,
            'median': 0,
            'std': 0,
            'exceeding_count_7s': 0,
            'exceeding_percentage_7s': 0,
            'exceeding_count_30s': 0,
            'exceeding_percentage_30s': 0,
            'duration_counts': {}
        }
    
    return durations, stats, deleted_files

def plot_duration_histogram(durations, title, output_path):
    """
    Plot a histogram of audio durations.
    
    Args:
        durations: List of durations in seconds
        title: Title for the plot
        output_path: Path to save the plot
    """
    if not durations:
        print(f"No data to plot for {title}")
        return
        
    plt.figure(figsize=(12, 6))
    plt.hist(durations, bins=50, alpha=0.7, color='blue')
    plt.axvline(x=7.0, color='red', linestyle='--', label='7 seconds limit')
    plt.axvline(x=30.0, color='green', linestyle='--', label='30 seconds limit')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Number of files')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()

def main():
    # Define directories to analyze
    directories = [
        "hf_data/kore_wavs",
        "hf_data/puck_wavs",
    ]
    
    # Create output directory for plots
    os.makedirs("audio_analysis", exist_ok=True)
    
    # Ask if files longer than 30 seconds should be deleted
    delete_long_files = True  # Set to True to delete files longer than 30 seconds
    
    print(f"Will delete files longer than 30 seconds: {delete_long_files}")
    
    # Analyze each directory
    all_stats = {}
    all_deleted_files = []
    
    for directory in directories:
        durations, stats, deleted_files = analyze_directory(directory, delete_long_files)
        all_deleted_files.extend(deleted_files)
        
        if durations:
            # Print statistics
            print(f"\nStatistics for {directory}:")
            print(f"Total files: {stats['count']}")
            print(f"Min duration: {stats['min']:.2f} seconds")
            print(f"Max duration: {stats['max']:.2f} seconds")
            print(f"Mean duration: {stats['mean']:.2f} seconds")
            print(f"Median duration: {stats['median']:.2f} seconds")
            print(f"Standard deviation: {stats['std']:.2f} seconds")
            print(f"Files exceeding 7 seconds: {stats['exceeding_count_7s']} ({stats['exceeding_percentage_7s']:.2f}%)")
            print(f"Files exceeding 30 seconds: {stats['exceeding_count_30s']} ({stats['exceeding_percentage_30s']:.2f}%)")
            
            # Plot histogram
            plot_title = f"Audio Duration Distribution - {os.path.basename(directory)}"
            output_path = os.path.join("audio_analysis", f"{os.path.basename(directory)}_histogram.png")
            plot_duration_histogram(durations, plot_title, output_path)
            
            # Save statistics
            all_stats[directory] = stats
    
    # Print overall summary
    if all_stats:
        print("\nOverall Summary:")
        total_files = sum(stats['count'] for stats in all_stats.values())
        total_exceeding_7s = sum(stats['exceeding_count_7s'] for stats in all_stats.values())
        total_exceeding_30s = sum(stats['exceeding_count_30s'] for stats in all_stats.values())
        
        print(f"Total audio files analyzed: {total_files}")
        print(f"Total files exceeding 7 seconds: {total_exceeding_7s} ({total_exceeding_7s/total_files*100:.2f}% if any)")
        print(f"Total files exceeding 30 seconds: {total_exceeding_30s} ({total_exceeding_30s/total_files*100:.2f}% if any)")
        
        # Summary of deleted files
        if delete_long_files:
            print(f"\nTotal files deleted: {len(all_deleted_files)}")
            print("Note: Each audio file deletion counted along with its corresponding text file.")
    else:
        print("\nNo statistics available. No files were analyzed.")

if __name__ == "__main__":
    main()
