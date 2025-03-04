#!/usr/bin/env python3
"""
Script to update the StyleTTS2 config file with your data paths
"""

import os
import argparse
import yaml

def main():
    parser = argparse.ArgumentParser(description="Update StyleTTS2 config file")
    parser.add_argument("--config_path", type=str, default="Configs/config.yml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="Data", help="Data directory")
    parser.add_argument("--wav_dir", type=str, help="Directory containing wav files")
    parser.add_argument("--log_dir", type=str, default="Models/Custom", help="Log directory")
    parser.add_argument("--multispeaker", action="store_true", help="Enable multispeaker training")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config
    config['log_dir'] = args.log_dir
    
    # Update data paths
    config['data_params']['train_data'] = os.path.join(args.data_dir, "train_list.txt")
    config['data_params']['val_data'] = os.path.join(args.data_dir, "val_list.txt")
    
    if args.wav_dir:
        config['data_params']['root_path'] = args.wav_dir
    else:
        config['data_params']['root_path'] = os.path.join(args.data_dir, "wavs")
    
    # Update multispeaker setting
    config['model_params']['multispeaker'] = args.multispeaker
    
    # Save updated config
    output_config_path = os.path.join(os.path.dirname(args.config_path), "custom_config.yml")
    with open(output_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Updated config saved to {output_config_path}")

if __name__ == "__main__":
    main()
