find /workspace/Eva_K/wavs -name "*.wav" -exec bash -c '
  for wav; do
    if ! ffprobe -v error "$wav" > /dev/null 2>&1; then
      echo "BROKEN: $wav"
    fi
  done
' bash {} +
