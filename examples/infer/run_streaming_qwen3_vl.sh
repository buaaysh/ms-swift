#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash examples/infer/run_streaming_qwen3_vl.sh
#
# Override with env vars if needed:
#   MODEL=Qwen/Qwen3-VL-8B-Instruct \
#   DATA=/path/to/infer_chunk.jsonl \
#   OUTPUT=/path/to/pred.jsonl \
#   FRAME_PREFIX_MAP='obs://bucket/data=/mnt/obs/data' \
#   bash examples/infer/run_streaming_qwen3_vl.sh

MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
DATA="${DATA:-/path/to/infer_chunk.jsonl}"
OUTPUT="${OUTPUT:-/path/to/pred.jsonl}"
FRAME_PREFIX_MAP="${FRAME_PREFIX_MAP:-obs://yw-ads-training-2-gy1/data=/your/local/or/mounted/data}"

# Optional generation args
MAX_TOKENS="${MAX_TOKENS:-32}"
TEMPERATURE="${TEMPERATURE:-0}"
VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-32}"
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"

python examples/infer/streaming_qwen3_vl.py \
  --model "${MODEL}" \
  --data "${DATA}" \
  --output "${OUTPUT}" \
  --max-tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --video-max-token-num "${VIDEO_MAX_TOKEN_NUM}" \
  --fps-max-frames "${FPS_MAX_FRAMES}" \
  --image-max-token-num "${IMAGE_MAX_TOKEN_NUM}" \
  --frame-prefix-map "${FRAME_PREFIX_MAP}"

