#!/usr/bin/env bash
# Day 26: bootstrap a local LLM for llama.cpp (CPU-only, no root).
#
# Idempotent: clones/builds llama.cpp and downloads a small GGUF model.
# Everything lands under history/ (gitignored), so nothing leaks into Git.
#
# Usage:
#   bash scripts/setup_llama_cpp.sh
#   bash scripts/setup_llama_cpp.sh --quant all      # Q4_K_M + Q2_K (Day 29)
#   bash scripts/setup_llama_cpp.sh --quant q2_k     # only the weaker Q2_K
#   bash scripts/setup_llama_cpp.sh --skip-build     # only download the model(s)
#   bash scripts/setup_llama_cpp.sh --skip-model     # only build llama.cpp
#   LLAMA_DIR=/opt/llama.cpp MODEL_DIR=/opt/models bash scripts/setup_llama_cpp.sh
#
# After a successful run:
#   llama-cli    -> history/llama.cpp/build/bin/llama-cli
#   llama-server -> history/llama.cpp/build/bin/llama-server
#   model        -> history/models/qwen2.5-3b-instruct-q4_k_m.gguf
#   model (Q2)   -> history/models/qwen2.5-3b-instruct-q2_k.gguf   (--quant q2_k|all)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$ROOT/history/llama.cpp}"
MODEL_DIR="${MODEL_DIR:-$ROOT/history/models}"
MODEL_FILE="${MODEL_FILE:-qwen2.5-3b-instruct-q4_k_m.gguf}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf}"
# Второй вариант — более слабое квантование Q2_K для Дня 29 (оптимизация/сравнение).
MODEL2_FILE="${MODEL2_FILE:-qwen2.5-3b-instruct-q2_k.gguf}"
MODEL2_URL="${MODEL2_URL:-https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q2_k.gguf}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

SKIP_BUILD=0
SKIP_MODEL=0
QUANT="q4_k_m"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build) SKIP_BUILD=1 ;;
    --skip-model) SKIP_MODEL=1 ;;
    --quant)      QUANT="$2"; shift ;;
    --llama-dir)  LLAMA_DIR="$2"; shift ;;
    --model-dir)  MODEL_DIR="$2"; shift ;;
    --jobs)       JOBS="$2"; shift ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -40
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

case "$QUANT" in
  q4_k_m|q2_k|all) ;;
  *) echo "Invalid --quant '$QUANT' (use q4_k_m, q2_k or all)" >&2; exit 2 ;;
esac

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

check_tool() {
  command -v "$1" >/dev/null 2>&1 || { echo "Required tool not found: $1" >&2; exit 1; }
}

build_llama_cpp() {
  check_tool git
  check_tool cmake
  check_tool g++

  if [[ ! -d "$LLAMA_DIR/.git" ]]; then
    log "Cloning llama.cpp into $LLAMA_DIR"
    mkdir -p "$(dirname "$LLAMA_DIR")"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
  else
    log "Updating llama.cpp in $LLAMA_DIR"
    git -C "$LLAMA_DIR" fetch --depth 1 origin
    git -C "$LLAMA_DIR" pull --ff-only --depth 1 origin HEAD || true
  fi

  log "Configuring CMake (Release, CPU, native optimizations)"
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON

  log "Building llama-cli and llama-server with $JOBS jobs"
  cmake --build "$LLAMA_DIR/build" --config Release -j "$JOBS" \
    --target llama-cli llama-server

  local bin_dir="$LLAMA_DIR/build/bin"
  [[ -x "$bin_dir/llama-cli" ]]    || { echo "llama-cli not built" >&2; exit 1; }
  [[ -x "$bin_dir/llama-server" ]] || { echo "llama-server not built" >&2; exit 1; }
  log "Binaries ready: $bin_dir/llama-cli, $bin_dir/llama-server"
}

download_model() {
  local file="$1" url="$2"
  local target="$MODEL_DIR/$file"
  if [[ -s "$target" ]]; then
    if [[ "$(head -c 4 "$target")" == "GGUF" ]]; then
      log "Model already present: $target"
      return
    fi
    log "Existing file is not a valid GGUF, re-downloading: $target"
    rm -f "$target"
  fi
  mkdir -p "$MODEL_DIR"
  log "Downloading model -> $target"
  if command -v curl >/dev/null 2>&1; then
    # -C - resumes a partially downloaded file (idempotent reruns).
    if ! curl -L --fail --progress-bar -C - "$url" -o "$target.part"; then
      echo "Download failed (curl). Partial file kept: $target.part" >&2
      exit 1
    fi
  elif command -v wget >/dev/null 2>&1; then
    if ! wget -c -O "$target.part" "$url"; then
      echo "Download failed (wget). Partial file kept: $target.part" >&2
      exit 1
    fi
  else
    echo "Neither curl nor wget is available" >&2
    exit 1
  fi
  if [[ "$(head -c 4 "$target.part")" != "GGUF" ]]; then
    echo "Downloaded file is not a valid GGUF model: $target.part" >&2
    exit 1
  fi
  mv "$target.part" "$target"
  log "Model ready: $target ($(du -h "$target" | cut -f1))"
}

# Use explicit if/then so that `set -e` remains active inside the functions.
if [[ "$SKIP_BUILD" -eq 0 ]]; then
  build_llama_cpp
else
  log "Skipping llama.cpp build (--skip-build)"
fi
if [[ "$SKIP_MODEL" -eq 0 ]]; then
  if [[ "$QUANT" == "q4_k_m" || "$QUANT" == "all" ]]; then
    download_model "$MODEL_FILE" "$MODEL_URL"
  fi
  if [[ "$QUANT" == "q2_k" || "$QUANT" == "all" ]]; then
    download_model "$MODEL2_FILE" "$MODEL2_URL"
  fi
else
  log "Skipping model download (--skip-model)"
fi

log "Done. Run:"
echo "  python3 -m agent.local_llm_scenarios --offline   # deterministic check"
echo "  python3 -m agent.local_llm_scenarios             # real local LLM check"
echo "  python3 -m agent.optimization_scenarios --offline # Day 29 optimization check"
echo "  python3 chat_local.py                            # interactive chat"
