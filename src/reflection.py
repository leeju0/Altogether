"""Run-local Reflection guideline document storage."""
from pathlib import Path
import hashlib
import os


FILENAME = 'reflection_guidelines.md'
TYPE_FILENAME = 'type_reflection_guidelines.md'
FINAL_FILENAME = 'final_reflection_guidelines.md'
INITIAL_DOCUMENT = '''# Reflection 판단 가이드라인

이 문서는 train 정답을 본 Reflection 모듈이 생성한 일반화된 이진판정 기준이다.

## 누적 판단 기준

- 아직 추가된 기준이 없다.
'''
TYPE_INITIAL_DOCUMENT = '''# 오류 유형 후보 Reflection 가이드라인

이 문서는 train 정답을 본 Reflection 모듈이 생성한 오류 유형 후보 재검토 기준이다.

## 누적 재검토 체크리스트

- 아직 추가된 기준이 없다.
'''
FINAL_INITIAL_DOCUMENT = '''# 최종 오류 유형 선택 Reflection 가이드라인

이 문서는 train 정답을 본 Reflection 모듈이 생성한 최종 후보 선택 재검토 기준이다.

## 누적 재검토 체크리스트

- 아직 추가된 기준이 없다.
'''


def _read(run_dir, filename, initial):
    path = Path(run_dir) / filename
    if not path.exists():
        _write(run_dir, filename, initial)
    return path.read_text(encoding='utf-8')


def _read_frozen(run_dir, filename, expected_sha256):
    path = Path(run_dir) / filename
    if not path.is_file():
        raise ValueError('Frozen Reflection 가이드라인 스냅샷이 없습니다.')
    markdown = path.read_text(encoding='utf-8')
    if hashlib.sha256(markdown.encode()).hexdigest() != expected_sha256:
        raise ValueError('Frozen Reflection 가이드라인 SHA-256이 manifest와 다릅니다.')
    return markdown


def _write(run_dir, filename, markdown):
    if not isinstance(markdown, str) or not markdown.strip() or not markdown.lstrip().startswith('#'):
        raise ValueError('Reflection 문서는 Markdown 제목으로 시작해야 합니다.')
    path = Path(run_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(markdown, encoding='utf-8')
    os.replace(temp, path)


def read_document(run_dir):
    return _read(run_dir, FILENAME, INITIAL_DOCUMENT)


def read_frozen_document(run_dir, expected_sha256):
    return _read_frozen(run_dir, FILENAME, expected_sha256)


def write_document(run_dir, markdown):
    _write(run_dir, FILENAME, markdown)


def read_type_document(run_dir):
    return _read(run_dir, TYPE_FILENAME, TYPE_INITIAL_DOCUMENT)


def read_frozen_type_document(run_dir, expected_sha256):
    return _read_frozen(run_dir, TYPE_FILENAME, expected_sha256)


def write_type_document(run_dir, markdown):
    _write(run_dir, TYPE_FILENAME, markdown)


def read_final_document(run_dir):
    return _read(run_dir, FINAL_FILENAME, FINAL_INITIAL_DOCUMENT)


def read_frozen_final_document(run_dir, expected_sha256):
    return _read_frozen(run_dir, FINAL_FILENAME, expected_sha256)


def write_final_document(run_dir, markdown):
    _write(run_dir, FINAL_FILENAME, markdown)

