# Copyright (c) ModelScope Contributors. All rights reserved.
"""Offline step-by-step Qwen3-VL inference for streaming scene JSONL data.

This runner targets scene records shaped like:
  {
    "scene_id": "...",
    "videos": [[frame0, frame1, ...]],
    "streaming_eval_steps": [
      {"step_idx": 0, "prefix_end": 5, "frames": [...], "user_text": "..."}
    ]
  }

The initial implementation uses recompute mode: every step sends the cumulative
frame prefix `videos[0][:prefix_end]` to the existing ms-swift Qwen3-VL
TransformersEngine. This validates data loading, Qwen3-VL preprocessing, and
generation before introducing KV-cache reuse.
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SYSTEM = '你是驾驶助手，可以根据用户指令生成安全准确的回复。'
DEFAULT_FORMAT_CONSTRAINT = (
    '只输出一个横向标签和一个纵向标签，格式必须为'
    '<LAT_LANE_CHANGE_LEFT>/<LAT_LANE_CHANGE_RIGHT>/<LAT_NUDGE_LEFT>/<LAT_NUDGE_RIGHT>/'
    '<LAT_LANE_KEEP>/<LAT_INTERSECTION_FOLLOW>/<LAT_TURN_LEFT>/<LAT_TURN_RIGHT>/<LAT_U_TURN>'
    '之一加上'
    '<LON_MAINTAIN>/<LON_ACCELERATE>/<LON_DECELERATE>/<LON_STOP>'
    '之一，最终格式为<...><...>，不要输出任何解释、标点或换行。'
)
LABEL_PATTERN = re.compile(r'(<LAT_[A-Z_]+>)\s*(<LON_[A-Z_]+>)')


def _json_dumps(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _load_json_records(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    if not text:
        return []

    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _parse_prefix_map(values: Optional[Sequence[str]]) -> List[Tuple[str, str]]:
    prefix_map = []
    for value in values or []:
        if '=' not in value:
            raise ValueError(f'Invalid --frame-prefix-map value: {value!r}. Expected OLD=NEW.')
        old, new = value.split('=', 1)
        prefix_map.append((old, new))
    return prefix_map


def _rewrite_frame_path(path: str, prefix_map: Sequence[Tuple[str, str]]) -> str:
    for old, new in prefix_map:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def _rewrite_frames(frames: Sequence[str], prefix_map: Sequence[Tuple[str, str]]) -> List[str]:
    return [_rewrite_frame_path(frame, prefix_map) for frame in frames]


def _get_record_frames(record: Dict[str, Any]) -> List[str]:
    videos = record.get('videos') or []
    if not videos:
        return []
    frames = videos[0]
    if not isinstance(frames, list):
        raise ValueError('This runner expects record["videos"][0] to be a list of frame paths.')
    return frames


def _get_system_message(record: Dict[str, Any], override_system: Optional[str], use_default_system: bool) -> Optional[str]:
    if override_system is not None:
        return override_system
    for message in record.get('messages') or []:
        if message.get('role') == 'system':
            return message.get('content')
    return DEFAULT_SYSTEM if use_default_system else None


def _build_messages(user_text: str, system: Optional[str]) -> List[Dict[str, str]]:
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': user_text})
    return messages


def _prepare_user_text(user_text: str, force_label_format: bool, format_constraint: str) -> str:
    if not force_label_format:
        return user_text
    return f'{user_text}\n\n{format_constraint}'


def _normalize_label_pred(pred: Optional[str]) -> Tuple[Optional[str], bool]:
    if not pred:
        return None, False
    match = LABEL_PATTERN.search(pred)
    if not match:
        return None, False
    return f'{match.group(1)}{match.group(2)}', True


def _select_step_frames(record: Dict[str, Any], step: Dict[str, Any], recompute: bool) -> List[str]:
    if not recompute:
        return step.get('frames') or []

    all_frames = _get_record_frames(record)
    prefix_end = step.get('prefix_end')
    if prefix_end is None:
        return step.get('frames') or all_frames
    return all_frames[:int(prefix_end)]


def _iter_steps(record: Dict[str, Any], max_steps: Optional[int]) -> Iterable[Dict[str, Any]]:
    steps = record.get('streaming_eval_steps') or []
    if max_steps is not None:
        steps = steps[:max_steps]
    return steps


def _parse_torch_dtype(value: Optional[str]):
    if value is None or value == 'auto':
        return None
    import torch

    mapping = {
        'float16': torch.float16,
        'fp16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
        'float32': torch.float32,
        'fp32': torch.float32,
    }
    if value not in mapping:
        raise ValueError(f'Unsupported torch dtype: {value}')
    return mapping[value]


def _create_engine(args: argparse.Namespace):
    from swift.infer_engine import TransformersEngine

    torch_dtype = _parse_torch_dtype(args.torch_dtype)
    return TransformersEngine(
        args.model,
        model_type=args.model_type,
        torch_dtype=torch_dtype,
        attn_impl=args.attn_impl,
        device_map=args.device_map,
        max_batch_size=1,
        use_hf=args.use_hf,
        revision=args.revision,
    )


def _infer_one_step(engine, messages: List[Dict[str, str]], frames: List[str], args: argparse.Namespace) -> str:
    from swift.infer_engine import InferRequest, RequestConfig

    request = InferRequest(messages=messages, videos=[frames])
    request_config = RequestConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        stop=args.stop,
    )
    response = engine.infer([request], request_config=request_config, use_tqdm=False)[0]
    return response.choices[0].message.content


def run(args: argparse.Namespace) -> None:
    if args.video_max_token_num is not None:
        os.environ.setdefault('VIDEO_MAX_TOKEN_NUM', str(args.video_max_token_num))
    if args.fps_max_frames is not None:
        os.environ.setdefault('FPS_MAX_FRAMES', str(args.fps_max_frames))
    if args.image_max_token_num is not None:
        os.environ.setdefault('IMAGE_MAX_TOKEN_NUM', str(args.image_max_token_num))

    records = _load_json_records(args.data)
    if args.max_scenes is not None:
        records = records[:args.max_scenes]

    prefix_map = _parse_prefix_map(args.frame_prefix_map)
    engine = None if args.dry_run else _create_engine(args)

    output_f = open(args.output, 'w', encoding='utf-8') if args.output else None
    try:
        for scene_idx, record in enumerate(records):
            scene_id = record.get('scene_id')
            system = _get_system_message(record, args.system, args.use_default_system)
            for step in _iter_steps(record, args.max_steps):
                frames = _select_step_frames(record, step, recompute=args.recompute)
                frames = _rewrite_frames(frames, prefix_map)
                user_text = _prepare_user_text(
                    step['user_text'], force_label_format=args.force_label_format, format_constraint=args.format_constraint)
                messages = _build_messages(user_text, system)
                pred = None if args.dry_run else _infer_one_step(engine, messages, frames, args)
                normalized_pred, pred_valid_format = _normalize_label_pred(pred)
                row = {
                    'scene_idx': scene_idx,
                    'scene_id': scene_id,
                    'step_idx': step.get('step_idx'),
                    'prefix_end': step.get('prefix_end'),
                    'frame_timestamp': step.get('frame_timestamp'),
                    'num_input_frames': len(frames),
                    'mode': 'recompute' if args.recompute else 'chunk',
                    'dry_run': args.dry_run,
                    'pred': pred,
                    'normalized_pred': normalized_pred,
                    'pred_valid_format': pred_valid_format,
                    'expected_assistant': step.get('expected_assistant'),
                }
                line = _json_dumps(row)
                if output_f is not None:
                    output_f.write(line + '\n')
                    output_f.flush()
                print(line, flush=True)
    finally:
        if output_f is not None:
            output_f.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Qwen3-VL model id or local checkpoint path.')
    parser.add_argument('--data', required=True, help='Path to scene JSONL, e.g. infer_chunk.jsonl.')
    parser.add_argument('--output', default=None, help='Optional JSONL output path.')
    parser.add_argument('--model-type', default=None, help='Optional ms-swift model_type override.')
    parser.add_argument('--revision', default=None)
    parser.add_argument('--use-hf', action='store_true', help='Use HuggingFace hub instead of ModelScope.')
    parser.add_argument('--attn-impl', default=None, choices=['eager', 'sdpa', 'flash_attn', 'flash_attention_2'])
    parser.add_argument('--device-map', default=None)
    parser.add_argument(
        '--torch-dtype',
        default='auto',
        choices=['auto', 'float16', 'fp16', 'bfloat16', 'bf16', 'float32', 'fp32'])

    parser.add_argument('--max-scenes', type=int, default=None)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--dry-run', action='store_true', help='Validate scene/step framing without loading a model.')
    parser.add_argument('--max-tokens', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=None)
    parser.add_argument('--top-k', type=int, default=None)
    parser.add_argument('--repetition-penalty', type=float, default=None)
    parser.add_argument('--stop', action='append', default=[])
    parser.add_argument('--force-label-format', action='store_true',
                        help='Append a strict output-format instruction for <LAT_...><LON_...>.')
    parser.add_argument('--format-constraint', default=DEFAULT_FORMAT_CONSTRAINT,
                        help='Custom format instruction appended when --force-label-format is enabled.')

    parser.add_argument('--recompute', action=argparse.BooleanOptionalAction, default=True,
                        help='Use cumulative frames videos[0][:prefix_end]. Disable to send only step["frames"].')
    parser.add_argument('--system', default=None, help='Override system message. Empty string disables system prompt.')
    parser.add_argument('--use-default-system', action=argparse.BooleanOptionalAction, default=False,
                        help='Use a driving assistant system prompt when the record has no system message.')
    parser.add_argument('--frame-prefix-map', action='append',
                        help='Rewrite frame path prefixes, e.g. obs://bucket/data=/mnt/data. Can be repeated.')

    parser.add_argument('--video-max-token-num', type=int, default=None)
    parser.add_argument('--fps-max-frames', type=int, default=None)
    parser.add_argument('--image-max-token-num', type=int, default=None)
    return parser.parse_args(argv)


if __name__ == '__main__':
    run(parse_args(sys.argv[1:]))
