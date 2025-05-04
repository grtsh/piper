import os
import torch
import torchaudio
import argparse
from tqdm import tqdm
from multiprocessing import Pool

# Argument parsing
parser = argparse.ArgumentParser(description="Validate cache files against wav files.")
parser.add_argument("--clean", action="store_true", help="Delete broken .pt and .spec.pt files if set.")
args = parser.parse_args()

wav_dir = "/workspace/Eva_K/wavs"
cache_dir = "/workspace/Eva_K_train/cache/22050"

def check_file(fname):
    if not (fname.endswith(".pt") and not fname.endswith(".spec.pt")):
        return None

    audio_id = fname.replace(".pt", "")
    wav_path = os.path.join(wav_dir, f"{audio_id}.wav")
    pt_path = os.path.join(cache_dir, fname)
    spec_path = pt_path.replace(".pt", ".spec.pt")

    if not os.path.exists(wav_path):
        return ("missing_wav", audio_id)

    try:
        wav_waveform, wav_sr = torchaudio.load(wav_path)
        pt_tensor = torch.load(pt_path, map_location="cpu")
        spec_tensor = torch.load(spec_path, map_location="cpu")

        if abs(wav_waveform.shape[1] - pt_tensor.shape[0]) > 5:
            return ("sample_mismatch", audio_id)

    except Exception:
        return ("load_error", audio_id)

    return ("good", audio_id)

files = os.listdir(cache_dir)

# Stats counters
stats = {
    "good": 0,
    "missing_wav": 0,
    "sample_mismatch": 0,
    "load_error": 0,
}

bad_files = []

# Multiprocessing with tqdm
with Pool(processes=24) as p:
    for result in tqdm(p.imap(check_file, files), total=len(files)):
        if result:
            error_type, audio_id = result
            stats[error_type] += 1
            if error_type != "good":
                bad_files.append(audio_id)

# If clean flag passed, remove broken cache entries
if args.clean:
    for audio_id in bad_files:
        pt = os.path.join(cache_dir, f"{audio_id}.pt")
        spec = os.path.join(cache_dir, f"{audio_id}.spec.pt")
        if os.path.exists(pt):
            os.remove(pt)
        if os.path.exists(spec):
            os.remove(spec)
    print(f"\nTotal broken entries cleaned: {len(bad_files)}")
else:
    print(f"\nCheck complete. Broken entries found (not deleted): {len(bad_files)}")

# Print statistics
print("\nStatistics:")
for k, v in stats.items():
    print(f"{k}: {v}")

