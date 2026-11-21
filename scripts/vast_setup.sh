#!/usr/bin/env bash
# Run ON the vast.ai machine, from the repo root (after scripts/push_to_vast.sh):
#   bash scripts/vast_setup.sh            # test split only (17 GB)
#   bash scripts/vast_setup.sh test part1 # add training parts (~320 GB each, 5 parts)
# Env: DATA (default /workspace/data).
set -euo pipefail
PARTS=("${@:-test}")
DATA=${DATA:-/workspace/data}
mkdir -p "$DATA/egodex"

echo "== system packages"
if command -v apt-get >/dev/null && [ "$(id -u)" = 0 ]; then
  apt-get update -qq && apt-get install -y -qq aria2 unzip libgl1 libglib2.0-0 >/dev/null
fi

echo "== python env"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d .venv ] || uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -q -r requirements.txt
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/2**30), 'GB')"

[ -f models/hand_landmarker.task ] || { mkdir -p models; curl -sSfL -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task; }

echo "== EgoDex: ${PARTS[*]}"
for p in "${PARTS[@]}"; do
  if [ -d "$DATA/egodex/$p" ]; then echo "$p already extracted"; continue; fi
  url="https://ml-site.cdn-apple.com/datasets/egodex/$p.zip"
  if command -v aria2c >/dev/null; then
    aria2c -q -x 16 -s 16 -c -d "$DATA/egodex" -o "$p.zip" "$url"
  else
    curl -sSfL -C - -o "$DATA/egodex/$p.zip" "$url"
  fi
  unzip -q "$DATA/egodex/$p.zip" -d "$DATA/egodex" && rm "$DATA/egodex/$p.zip"
  [ -d "$DATA/egodex/$p" ] || { echo "expected $DATA/egodex/$p after unzip; got:"; ls "$DATA/egodex"; exit 1; }
done

echo "== index (paths differ from the Mac, so it is rebuilt here; splits are path-independent)"
roots=(); for p in "${PARTS[@]}"; do roots+=("$DATA/egodex/$p"); done
.venv/bin/python scripts/build_index.py "${roots[@]}" --out data --workers "$(nproc)"
echo "== ready. Next steps: see README.md (GPU runbook)"
