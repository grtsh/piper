import csv

metadata_path = "/workspace/Eva_K/metadata.csv"
deduped_path = "/workspace/Eva_K/metadata_deduped.csv"

seen = set()

with open(metadata_path, "r", encoding="utf-8") as fin, \
     open(deduped_path, "w", encoding="utf-8", newline="") as fout:
    reader = csv.reader(fin, delimiter="|")
    writer = csv.writer(fout, delimiter="|")
    for row in reader:
        audio_id = row[0]
        if audio_id not in seen:
            writer.writerow(row)
            seen.add(audio_id)

print(f"Deduplicated metadata written to {deduped_path}")
print(f"Original lines: {len(seen)} unique entries.")
