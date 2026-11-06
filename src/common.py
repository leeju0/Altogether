"""Dataset, input preparation and local files. No OCR framework imports."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import uuid

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT.parent / 'data/train'
RUNS_DIR = ROOT / 'runs'
MODEL_LABELS = {'ppocr': 'PP-OCRv5 + Korean rec', 'vl': 'PaddleOCR-VL-1.6'}
MODEL_PATHS = {
    'det': '/data/LLM_model/PP-OCRv5/PP-OCRv5_server_det',
    'rec': '/data/LLM_model/PP-OCRv5/korean_PP-OCRv5_mobile_rec',
    'vl': '/data/LLM_model/PaddleOCR-VL-1.6',
    'layout': str(ROOT / 'models/PP-DocLayoutV3'),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default


def write_json(path, value):
    """Publish a complete file; interrupted writes never masquerade as success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('w', encoding='utf-8') as out:
            json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def load_records(dataset_dir=DEFAULT_DATASET):
    with (Path(dataset_dir) / 'records.jsonl').open(encoding='utf-8') as source:
        rows = [json.loads(line) for line in source if line.strip()]
    ids = [row['record_id'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('중복된 record_id가 있습니다.')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', rid) for rid in ids):
        raise ValueError('파일명으로 사용할 수 없는 record_id입니다.')
    return rows


def resolve_image(dataset_dir, row):
    root = Path(dataset_dir).resolve()
    path = (root / row['image_file']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('이미지 경로가 데이터 폴더를 벗어납니다.')
    if not path.exists():
        path = root / 'images' / Path(row['image_file']).name
    return path


def prepare_image(path, background='white'):
    """Intrinsic-size SVG, frame zero, RGB alpha composite; no EXIF/resize."""
    if background not in ('white', 'dark', 'auto_gray'):
        raise ValueError('background must be white, dark or auto_gray')
    path = Path(path)
    is_svg = path.suffix.lower() == '.svg'
    if is_svg:
        import cairosvg
        source = io.BytesIO(cairosvg.svg2png(url=str(path)))
    else:
        source = path
    decoder = 'pillow'
    try:
        opened = Image.open(source)
    except UnidentifiedImageError:
        # A train PNG has a bad iCCP checksum but valid pixel data. OpenCV
        # ignores the ancillary color profile; do not enable global truncated IO.
        data = path.read_bytes()
        if is_svg or not data.startswith(b'\x89PNG\r\n\x1a\n'):
            raise
        import cv2
        import numpy as np
        pixels = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if pixels is None:
            raise
        if pixels.ndim == 3:
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGRA2RGBA if pixels.shape[2] == 4 else cv2.COLOR_BGR2RGB)
        opened = Image.fromarray(pixels)
        opened.format = 'PNG'
        decoder = 'opencv-png-fallback'
    with opened as image:
        image.seek(0)
        meta = {'format': 'SVG' if is_svg else image.format,
                'source_size': list(image.size), 'frames': getattr(image, 'n_frames', 1),
                'frame_index': 0, 'background': background, 'exif_applied': False, 'decoder': decoder}
        rgba = image.convert('RGBA')
        alpha_min, alpha_max = rgba.getchannel('A').getextrema()
        meta['has_transparency'] = alpha_min < 255
        color = (32, 32, 32, 255) if background == 'dark' else (255, 255, 255, 255)
        rgb = Image.alpha_composite(Image.new('RGBA', rgba.size, color), rgba).convert('RGB')
        if background == 'auto_gray':
            # Only recover a visible transparent foreground lost on white.
            adjusted = alpha_min < 255 and alpha_max > 0 and all(lo >= 250 for lo, _ in rgb.getextrema())
            if adjusted:
                color = (224, 224, 224, 255)
                rgb = Image.alpha_composite(Image.new('RGBA', rgba.size, color), rgba).convert('RGB')
            meta.update(requested_background=background, background='light_gray' if adjusted else 'white',
                        background_rgb=list(color[:3]), background_adjusted=adjusted)
        # Strip orientation metadata so downstream decoders see the same pixels.
        rgb.info.clear()
        return rgb, meta


def create_run(dataset_dir, record_ids, models, background='white', device='gpu:2', cpu_threads=8, max_new_tokens=2048,
               gemma_config=None, compare_without_ocr=False, pipeline_mode='stream', *,
               records=None, evaluation_groups=None):
    rows = [dict(row) for row in (records if records is not None else load_records(dataset_dir))]
    available = {row['record_id'] for row in rows}
    ids = list(record_ids)
    models = list(dict.fromkeys(models))
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= available:
        raise ValueError('데이터에 존재하는 서로 다른 이미지 ID를 선택하세요.')
    direct = pipeline_mode == 'gemma_only'
    if direct and (models or not gemma_config or compare_without_ocr):
        raise ValueError('Gemma 단독 실행에는 Gemma 설정만 필요합니다.')
    if not direct and (not models or not set(models) <= MODEL_LABELS.keys()):
        raise ValueError('OCR 모델을 선택하세요.')
    if background not in ('white', 'dark', 'auto_gray') or not re.fullmatch(r'gpu:\d+', device):
        raise ValueError('지원하지 않는 배경 또는 장치입니다.')
    if not 1 <= int(cpu_threads) <= 64 or not 64 <= int(max_new_tokens) <= 8192:
        raise ValueError('CPU threads: 1~64, max_new_tokens: 64~8192')
    if gemma_config:
        from gemma import validate_config
        validate_config(gemma_config)
        if not direct and 'vl' not in models:
            raise ValueError('Gemma 정제·판정에는 PaddleOCR-VL 결과가 필요합니다.')
    if compare_without_ocr and not gemma_config:
        raise ValueError('OCR 유무 비교에는 Gemma 설정이 필요합니다.')
    if pipeline_mode not in ('stream', 'batch', 'gemma_only'):
        raise ValueError('지원하지 않는 pipeline_mode입니다.')
    if evaluation_groups is not None:
        if not isinstance(evaluation_groups, dict) or not evaluation_groups:
            raise ValueError('평가 그룹은 선택한 모든 ID를 중복 없이 정확히 한 번 포함해야 합니다.')
        if any(not isinstance(group, dict)
                or not isinstance(group.get('label'), str) or not group['label'].strip()
                       or not isinstance(group.get('record_ids'), list) or not group['record_ids']
                       for group in evaluation_groups.values()):
            raise ValueError('평가 그룹은 선택한 모든 ID를 중복 없이 정확히 한 번 포함해야 합니다.')
        grouped = [rid for group in evaluation_groups.values() for rid in group['record_ids']]
        if len(grouped) != len(set(grouped)) or set(grouped) != set(ids):
            raise ValueError('평가 그룹은 선택한 모든 ID를 중복 없이 정확히 한 번 포함해야 합니다.')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    run_dir = RUNS_DIR / run_id
    manifest = {'schema_version': 1, 'run_id': run_id, 'created_at': now(),
                'dataset_dir': str(Path(dataset_dir).resolve()), 'record_ids': ids,
                'models': models, 'background': background, 'device': device,
                'pipeline_mode': pipeline_mode,
                'cpu_threads': int(cpu_threads), 'max_new_tokens': int(max_new_tokens),
                'use_ocr_for_image_block': True,
                'model_paths': MODEL_PATHS, 'preprocessing': 'svg-native_frame0_alpha-v1_no-exif'}
    if gemma_config:
        manifest['gemma'] = gemma_config
    if background == 'auto_gray':
        manifest['preprocessing'] = 'svg-native_frame0_alpha-v2-auto-gray_no-exif'
    if compare_without_ocr:
        manifest['compare_without_ocr'] = True
    if evaluation_groups is not None:
        manifest['evaluation_groups'] = evaluation_groups
    if direct:
        for key in ('cpu_threads', 'max_new_tokens', 'use_ocr_for_image_block', 'model_paths'):
            manifest.pop(key)
    write_json(run_dir / 'manifest.json', manifest)
    # Snapshot displayed metadata, so later dataset edits do not alter an experiment.
    write_json(run_dir / 'records.json', [row for row in rows if row['record_id'] in set(ids)])
    write_json(run_dir / 'status.json', {'state': 'ready', 'completed': 0, 'errors': 0,
                                       'total': len(ids) if direct else len(ids) * (len(models) + (2 if gemma_config else 0) + int(compare_without_ocr)), 'updated_at': now()})
    return run_dir


def list_runs():
    return sorted((p.parent for p in RUNS_DIR.glob('*/manifest.json')), reverse=True)


def load_manifest(run_dir):
    manifest = read_json(Path(run_dir) / 'manifest.json')
    if manifest is None:
        raise FileNotFoundError('실험 manifest.json이 없습니다.')
    return manifest


def result_path(run_dir, model, record_id):
    if model not in (*MODEL_LABELS, 'cleaned', 'gemma', 'gemma_no_ocr') or not re.fullmatch(r'[A-Za-z0-9_-]+', record_id):
        raise ValueError('잘못된 모델 또는 이미지 ID')
    return Path(run_dir) / 'results' / model / (record_id + '.json')


def load_result(run_dir, model, record_id):
    return read_json(result_path(run_dir, model, record_id))


def get_status(run_dir):
    state = read_json(Path(run_dir) / 'status.json', {'state': 'ready', 'completed': 0, 'errors': 0, 'total': 0})
    if state.get('state') in ('running', 'stopping') and state.get('pid'):
        try:
            cmdline = Path(f"/proc/{state['pid']}/cmdline").read_bytes()
            if b'run_ocr.py' not in cmdline:
                state['state'] = 'interrupted'
        except FileNotFoundError:
            state['state'] = 'interrupted'
    return state


def run_elapsed_seconds(run_dir, status=None):
    """Accumulate active attempts; legacy runs use their recorded wall-clock span."""
    run_dir = Path(run_dir)
    status = get_status(run_dir) if status is None else status
    timing = read_json(run_dir / 'timing.json')
    elapsed = float(timing.get('elapsed_seconds', 0)) if timing else 0.0
    if timing and not timing.get('running'):
        return max(0.0, elapsed)
    if timing:
        started = timing.get('started_at')
    else:
        if status.get('state') == 'ready':
            return 0.0
        manifest = read_json(run_dir / 'manifest.json', {})
        started = manifest.get('created_at') or status.get('started_at')
    ended = now() if status.get('state') in ('running', 'stopping') else status.get('updated_at')
    if started and ended:
        elapsed += max(0.0, (datetime.fromisoformat(ended) - datetime.fromisoformat(started)).total_seconds())
    return max(0.0, elapsed)


def acquire_worker_lock():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    handle = (RUNS_DIR / '.worker.lock').open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError('다른 판정 작업이 실행 중입니다. 완료 또는 중지 후 실행하세요.') from None
    return handle


def launch_run(run_dir, gemma_config=None, compare_without_ocr=False):
    run_dir = Path(run_dir).resolve()
    manifest = load_manifest(run_dir)
    if manifest.get('pipeline_mode') != 'gemma_only':
        raise ValueError('과거 OCR 실험은 열람만 지원합니다. Gemma 단독 새 실행을 만드세요.')
    if compare_without_ocr:
        raise ValueError('새 실행은 이미지와 메타데이터만 사용합니다.')
    saved_config = manifest.get('gemma') or {}
    runtime_upgrade = False
    if (gemma_config is None and saved_config.get('guideline_mode') in (
            'sections_1_3_v1', 'sections_1_3_v1_0', 'sections_1_3_5_v1_1')
            and not saved_config.get('recheck_appropriate') and not saved_config.get('execution_mode')):
        gemma_config = {**saved_config, 'execution_mode': 'record_parallel_v1', 'parallel_slots': 2}
        runtime_upgrade = True
    handle = acquire_worker_lock()
    try:
        if gemma_config is not None:
            from gemma import validate_config
            validate_config(gemma_config)
            if manifest.get('gemma') and manifest['gemma'] != gemma_config and not runtime_upgrade:
                raise ValueError('기존 Gemma 설정 변경은 새 실행에서 하세요.')
            manifest['gemma'] = gemma_config
            write_json(run_dir / 'manifest.json', manifest)
        (run_dir / 'STOP').unlink(missing_ok=True)
        with (run_dir / 'worker.log').open('a', encoding='utf-8') as log:
            proc = subprocess.Popen([sys.executable, '-u', str(ROOT / 'run_ocr.py'), '--run', str(run_dir),
                                     '--lock-fd', str(handle.fileno())], cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                    pass_fds=(handle.fileno(),))
        return proc.pid
    finally:
        # Closing this descriptor leaves the inherited child lock held.
        handle.close()


def request_stop(run_dir):
    (Path(run_dir) / 'STOP').touch()


def _ratings_db(run_dir):
    conn = sqlite3.connect(Path(run_dir) / 'ratings.sqlite', timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute('''CREATE TABLE IF NOT EXISTS model_ratings (
        record_id TEXT NOT NULL, reviewer TEXT NOT NULL,
        classification_score INTEGER NOT NULL CHECK(classification_score BETWEEN 1 AND 5),
        classification_reason TEXT NOT NULL, updated_at TEXT NOT NULL,
        PRIMARY KEY(record_id, reviewer))''')
    # Keep legacy A/B preferences in their original table; these are a new scale.
    conn.execute('''CREATE TABLE IF NOT EXISTS quality_ratings (
        record_id TEXT NOT NULL, reviewer TEXT NOT NULL,
        detection_score INTEGER NOT NULL CHECK(detection_score BETWEEN 1 AND 5),
        text_score INTEGER NOT NULL CHECK(text_score BETWEEN 1 AND 5),
        detection_reason TEXT NOT NULL, text_reason TEXT NOT NULL, updated_at TEXT NOT NULL,
        PRIMARY KEY(record_id, reviewer))''')
    fields = {row[1] for row in conn.execute('PRAGMA table_info(quality_ratings)')}
    if 'classification_score' not in fields:
        conn.execute('ALTER TABLE quality_ratings ADD COLUMN classification_score INTEGER CHECK(classification_score BETWEEN 1 AND 5)')
    if 'classification_reason' not in fields:
        conn.execute("ALTER TABLE quality_ratings ADD COLUMN classification_reason TEXT NOT NULL DEFAULT ''")
    return conn


def save_rating(run_dir, record_id, reviewer, detection_score=None, text_score=None, detection_reason='', text_reason='',
                classification_score=None, classification_reason=''):
    if not reviewer.strip():
        raise ValueError('평가자 이름 또는 별칭을 입력하세요.')
    if record_id not in load_manifest(run_dir)['record_ids']:
        raise ValueError('실험에 없는 이미지입니다.')
    if load_manifest(run_dir).get('pipeline_mode') == 'gemma_only':
        if type(classification_score) is not int or not 1 <= classification_score <= 5:
            raise ValueError('모델 판정 품질을 1~5점으로 선택하세요.')
        with _ratings_db(run_dir) as conn:
            conn.execute('INSERT OR REPLACE INTO model_ratings VALUES (?, ?, ?, ?, ?)',
                         (record_id, reviewer.strip(), classification_score,
                          classification_reason.strip() if classification_score <= 2 else '', now()))
        return
    if any(type(score) is not int or not 1 <= score <= 5 for score in (detection_score, text_score)):
        raise ValueError('영역 검출 품질과 텍스트 품질을 각각 1~5점으로 선택하세요.')
    if classification_score is not None and (type(classification_score) is not int or not 1 <= classification_score <= 5):
        raise ValueError('모델 판정 품질은 1~5점으로 선택하세요.')
    with _ratings_db(run_dir) as conn:
        conn.execute('''INSERT OR REPLACE INTO quality_ratings
                     (record_id, reviewer, detection_score, text_score, detection_reason, text_reason,
                      updated_at, classification_score, classification_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (record_id, reviewer.strip(), detection_score, text_score,
                      detection_reason.strip() if detection_score <= 2 else '',
                      text_reason.strip() if text_score <= 2 else '', now(), classification_score,
                      classification_reason.strip() if classification_score in (1, 2) else ''))


def load_rating(run_dir, record_id, reviewer):
    table = 'model_ratings' if load_manifest(run_dir).get('pipeline_mode') == 'gemma_only' else 'quality_ratings'
    with _ratings_db(run_dir) as conn:
        row = conn.execute(f'SELECT * FROM {table} WHERE record_id=? AND reviewer=?',
                           (record_id, reviewer.strip())).fetchone()
    return dict(row) if row else None


def ratings_csv(run_dir):
    direct = load_manifest(run_dir).get('pipeline_mode') == 'gemma_only'
    table = 'model_ratings' if direct else 'quality_ratings'
    with _ratings_db(run_dir) as conn:
        rows = conn.execute(f'SELECT * FROM {table} ORDER BY record_id, reviewer').fetchall()
    out = io.StringIO(newline='')
    writer = csv.writer(out)
    fields = ['record_id', 'reviewer', 'detection_score', 'text_score', 'detection_reason', 'text_reason',
              'updated_at', 'classification_score', 'classification_reason']
    if direct:
        fields = ['record_id', 'reviewer', 'classification_score', 'classification_reason', 'updated_at']
    writer.writerow(fields)
    writer.writerows([row[field] for field in fields] for row in rows)
    return out.getvalue().encode('utf-8-sig')
