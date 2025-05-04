mkdir -p /workspace/FinalDataset/wavs_22050

find /workspace/FinalDataset/wavs -name "*.wav" -exec bash -c '
  for wav; do
    filename=$(basename "$wav")
    sox "$wav" -r 22050 /workspace/FinalDataset/wavs_22050/"$filename" gain -n -3
  done
' bash {} +
