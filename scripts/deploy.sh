# Sync local project to RunPod instance, excluding large folders.
#!/usr/bin/env bash
set -euo pipefail

ZIP_NAME="llama-inference.zip"
PROJECT_NAME="llama-inference"

rm -f "$ZIP_NAME" 2>/dev/null || true

echo "Zipping project (excluding large or cached folders)..."
zip -r "$ZIP_NAME" . \
  -x ".venv/*" "model-out/*" "models/*" ".idea/*" "__pycache__/*" "*.pyc" "*.pyo" "*.log" ".DS_Store" "uv.lock"

echo "Uploading $ZIP_NAME to RunPod..."
runpodctl send "$ZIP_NAME"

rm -f "$ZIP_NAME"

cat <<EOF

In runpod terminal run:

mv -f llama-inference.zip /workspace/ && \
cd /workspace && \
unzip -qo llama-inference && \
rm -f llama-inference.zip

EOF