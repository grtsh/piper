mkdir -p /workspace/FinalDataset/wavs
touch /workspace/FinalDataset/metadata.csv

find /workspace/Eva_K -type d -mindepth 1 -exec bash -c '
  for dir; do
    if [ -f "$dir/metadata.csv" ]; then
      wav_dir="$dir/wavs"
      while IFS= read -r line; do
        wav_file="$(echo "$line" | cut -d"|" -f1).wav"
        cp "$wav_dir/$wav_file" /workspace/FinalDataset/wavs/
        echo "$line" >> /workspace/FinalDataset/metadata.csv
      done < "$dir/metadata.csv"
    fi
  done
' bash {} +
