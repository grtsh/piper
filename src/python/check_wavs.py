import csv
from collections import Counter

metadata_path = "/workspace/Eva_K/metadata.csv"

# Collect all audio file names
audio_files = []

with open(metadata_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="|")
    for row in reader:
        audio_files.append(row[0])  # first column = wav name without .wav

# Count duplicates
counter = Counter(audio_files)

# Find duplicates
duplicates = [item for item, count in counter.items() if count > 1]

print(f"Total entries: {len(audio_files)}")
print(f"Unique entries: {len(counter)}")
print(f"Duplicate entries: {len(duplicates)}")

if duplicates:
    print("\nExample duplicates:")
    for dup in duplicates[:10]:
        print(f"- {dup}")
