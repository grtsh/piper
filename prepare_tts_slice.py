import os
import argparse
import random
import shutil
import librosa
import numpy as np
from tqdm import tqdm
import warnings
import csv # Import csv module for potentially more robust parsing

# Suppress specific librosa warnings if they become noisy
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')
# Suppress FutureWarning from librosa related to n_fft calculation
warnings.filterwarnings("ignore", category=FutureWarning, module='librosa')


def get_audio_duration(filepath):
    """Safely gets the duration of an audio file."""
    try:
        # Use librosa's optimized duration calculation
        duration = librosa.get_duration(filename=filepath)
        return duration
    except Exception as e:
        print(f"Warning: Could not process file {filepath}. Error: {e}")
        return None

def format_duration(seconds):
    """Formats seconds into H:M:S"""
    if seconds is None or not np.isfinite(seconds):
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def calculate_stats(durations):
    """Calculates min, max, avg duration and total duration."""
    if not durations:
        return 0, 0, 0, 0
    total_duration = sum(durations)
    min_dur = min(durations)
    max_dur = max(durations)
    avg_dur = total_duration / len(durations)
    return min_dur, max_dur, avg_dur, total_duration

def print_stats(title, durations, file_count=None):
    """Prints formatted statistics."""
    min_d, max_d, avg_d, total_d = calculate_stats(durations)
    num_files = file_count if file_count is not None else len(durations)
    print(f"\n--- {title} Statistics ---")
    if num_files > 0:
        print(f"Number of files: {num_files}")
        print(f"Total Duration : {format_duration(total_d)} ({total_d:.2f} seconds)")
        if durations: # Only print min/max/avg if we have durations
            print(f"Min Length     : {min_d:.3f} seconds")
            print(f"Max Length     : {max_d:.3f} seconds")
            print(f"Average Length : {avg_d:.3f} seconds")
        else:
             print("Durations not applicable or not calculated for this set.")
    else:
        print("No files found.")
    print("-" * (len(title) + 10))


def main(args):
    """Main function to filter and slice the LJSpeech-style dataset."""
    print(f"Starting dataset preparation with seed: {args.seed}")
    random.seed(args.seed)
    target_seconds = args.target_hours * 3600

    # --- Validate Input Directory and Structure ---
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return

    wavs_dir = os.path.join(args.input_dir, 'wavs')
    metadata_path = os.path.join(args.input_dir, 'metadata.csv')

    if not os.path.isdir(wavs_dir):
        print(f"Error: 'wavs' subdirectory not found inside '{args.input_dir}'.")
        return
    if not os.path.isfile(metadata_path):
         print(f"Error: 'metadata.csv' not found inside '{args.input_dir}'.")
         return

    # --- Handle Output Directory ---
    if os.path.exists(args.output_dir):
        print(f"Warning: Output directory '{args.output_dir}' exists. Removing it.")
        try:
            shutil.rmtree(args.output_dir)
        except OSError as e:
            print(f"Error removing directory {args.output_dir}: {e}")
            return
    try:
        os.makedirs(args.output_dir)
        # Create the 'wavs' subdirectory in the output
        os.makedirs(os.path.join(args.output_dir, 'wavs'))
        print(f"Created output directory structure: '{args.output_dir}/' and '{args.output_dir}/wavs/'")
    except OSError as e:
        print(f"Error creating directory structure in {args.output_dir}: {e}")
        return

    # --- Load Metadata ---
    print(f"Loading metadata from: {metadata_path}...")
    metadata = {}
    original_metadata_lines = 0
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            # Using simple split assuming standard LJSpeech format: stem|text
            # Using csv.reader might be slightly more robust if delimiters appear in text
            # reader = csv.reader(f, delimiter='|', quoting=csv.QUOTE_NONE)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                original_metadata_lines += 1
                parts = line.split('|', 1)
                if len(parts) == 2:
                    stem, text = parts
                    metadata[stem] = text # Store text associated with stem
                else:
                    print(f"Warning: Skipping malformed metadata line: {line}")
    except Exception as e:
        print(f"Error reading metadata file {metadata_path}: {e}")
        return

    print(f"Loaded {len(metadata)} valid entries from metadata.csv (Original lines: {original_metadata_lines}).")
    if not metadata:
         print("Error: No valid metadata entries loaded.")
         return


    # --- Scan Input WAVs Directory, Get Durations, and Match Metadata ---
    print(f"\nScanning WAVs directory: {wavs_dir}...")
    all_files_data = []
    skipped_audio_files = 0
    metadata_mismatches = 0
    valid_extensions = ('.wav',) # LJSpeech specifically uses wav

    all_audio_filepaths = []
    for filename in os.listdir(wavs_dir):
        if filename.lower().endswith(valid_extensions):
            all_audio_filepaths.append(os.path.join(wavs_dir, filename))

    print(f"Found {len(all_audio_filepaths)} potential WAV files. Calculating durations and matching metadata...")
    for filepath in tqdm(all_audio_filepaths, desc="Processing WAVs"):
        stem = os.path.splitext(os.path.basename(filepath))[0]

        if stem not in metadata:
            # print(f"Warning: No metadata entry found for WAV file stem: {stem}. Skipping file.")
            metadata_mismatches += 1
            continue # Skip this file if no metadata exists

        duration = get_audio_duration(filepath)
        if duration is not None:
            all_files_data.append({"path": filepath, "duration": duration, "stem": stem})
        else:
            skipped_audio_files += 1 # File exists, metadata exists, but couldn't get duration

    if skipped_audio_files > 0:
        print(f"Skipped {skipped_audio_files} WAV files due to processing errors.")
    if metadata_mismatches > 0:
         print(f"Skipped {metadata_mismatches} WAV files due to missing metadata entries.")

    if not all_files_data:
        print("Error: No valid audio files with matching metadata found.")
        # Clean up empty output dir
        try:
            shutil.rmtree(args.output_dir)
            print(f"Removed empty output directory: {args.output_dir}")
        except OSError:
            pass
        return

    # --- Original Dataset Statistics (Based on matched files) ---
    original_durations = [f['duration'] for f in all_files_data]
    print_stats("Original Matched Dataset", original_durations, file_count=len(all_files_data))


    # --- Filter by Length ---
    print(f"\nFiltering files between {args.min_len_sec:.2f}s and {args.max_len_sec:.2f}s...")
    eligible_files_data = [
        f for f in all_files_data
        if args.min_len_sec <= f['duration'] <= args.max_len_sec
    ]

    if not eligible_files_data:
        print("Error: No files found within the specified length constraints after matching metadata.")
        try:
            shutil.rmtree(args.output_dir)
            print(f"Removed empty output directory: {args.output_dir}")
        except OSError:
             pass
        return

    # --- Eligible Files Statistics ---
    eligible_durations = [f['duration'] for f in eligible_files_data]
    print_stats("Length-Filtered (Eligible)", eligible_durations)
    _, _, _, total_eligible_duration = calculate_stats(eligible_durations)

    # --- Select Random Slice ---
    print(f"\nSelecting random slice aiming for {args.target_hours} hours ({target_seconds:.2f} seconds)...")
    random.shuffle(eligible_files_data)

    selected_files_data = []
    current_total_duration = 0

    for file_data in eligible_files_data:
        if current_total_duration < target_seconds:
            selected_files_data.append(file_data)
            current_total_duration += file_data['duration']
        else:
            # Stop adding once the target is met or exceeded
            break

    if current_total_duration < target_seconds and len(eligible_files_data) == len(selected_files_data):
        # This condition means we added all eligible files but didn't reach the target.
        print(f"Warning: Total duration of eligible files ({format_duration(total_eligible_duration)}) "
              f"is less than the target duration ({format_duration(target_seconds)}). "
              f"Using all {len(eligible_files_data)} eligible files.")
    else:
         print(f"Target duration reached or exceeded.")


    # --- Final Selection Statistics ---
    final_durations = [f['duration'] for f in selected_files_data]
    print_stats("Final Selected Slice", final_durations)
    _, _, _, final_total_duration = calculate_stats(final_durations)


    # --- Copy Selected Files and Create New Metadata ---
    print(f"\nCopying {len(selected_files_data)} selected WAV files and creating new metadata.csv...")
    copied_files = 0
    copy_errors = 0
    output_metadata_lines = []
    output_wavs_dir = os.path.join(args.output_dir, 'wavs')

    # Sort selected files by stem for consistent metadata.csv order
    selected_files_data.sort(key=lambda x: x['stem'])

    for file_data in tqdm(selected_files_data, desc="Copying files & preparing metadata"):
        source_path = file_data['path']
        stem = file_data['stem']
        dest_filename = os.path.basename(source_path)
        dest_path = os.path.join(output_wavs_dir, dest_filename)

        # Copy WAV file
        try:
            shutil.copy2(source_path, dest_path)
            copied_files += 1
            # Prepare metadata line for this copied file
            transcript = metadata.get(stem) # Should always exist due to earlier check
            if transcript:
                 output_metadata_lines.append(f"{stem}|{transcript}")
            else:
                 # This shouldn't happen based on logic, but as a safeguard:
                 print(f"Internal Warning: Metadata missing for selected stem {stem} during copy phase.")
                 copy_errors += 1 # Treat missing metadata at this stage as an error too

        except Exception as e:
            print(f"Error copying {source_path} to {dest_path}: {e}")
            copy_errors += 1

    # --- Write New Metadata File ---
    output_metadata_path = os.path.join(args.output_dir, 'metadata.csv')
    try:
        with open(output_metadata_path, 'w', encoding='utf-8') as f:
            for line in output_metadata_lines:
                f.write(line + '\n')
        print(f"Successfully wrote {len(output_metadata_lines)} lines to {output_metadata_path}")
    except Exception as e:
        print(f"Error writing new metadata file {output_metadata_path}: {e}")

    # --- Final Summary ---
    print("\n--- Preparation Complete ---")
    print(f"Successfully copied: {copied_files} WAV files")
    if copy_errors > 0:
        print(f"Errors encountered during copy or metadata lookup: {copy_errors}")
    print(f"Total duration of copied data: {format_duration(final_total_duration)}")
    print(f"Dataset slice (wavs/ and metadata.csv) saved to: {args.output_dir}")
    print(f"Seed used: {args.seed}")
    print("--------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter LJSpeech-style audio dataset by length and create a random slice, updating metadata.csv.")

    parser.add_argument("--input_dir", required=True, help="Directory containing the full LJSpeech dataset (must contain 'wavs/' and 'metadata.csv').")
    parser.add_argument("--output_dir", required=True, help="Directory to save the sliced dataset (will be overwritten).")
    parser.add_argument("--target_hours", required=True, type=float, help="Desired total duration of the output slice in hours.")
    parser.add_argument("--max_len_sec", required=True, type=float, help="Maximum duration of individual audio files in seconds.")
    parser.add_argument("--min_len_sec", type=float, default=0.0, help="Minimum duration of individual audio files in seconds (default: 0).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible slicing (default: 42).")

    args = parser.parse_args()
    main(args)