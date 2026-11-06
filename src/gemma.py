"""Owned local llama.cpp server and two image-grounded JSON contracts (stdlib only)."""
from __future__ import annotations

import base64
import ctypes
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import signal
import subprocess
import time
import urllib.error
import urllib.request

from common import ROOT, now, read_json, write_json

LABELS = ('적절', '무의미형', '파일명형', '중복형', '장식오용형', '불충분형', '무관형')
ERROR_PRIORITY = ('파일명형', '무관형', '무의미형', '장식오용형', '중복형', '불충분형')
METADATA_FIELDS = ('alt_text', 'page_title', 'context_text', 'in_link', 'link_dest', 'heading_path', 'kogl_marker')
INPUT_VERSION = 'v2-aux-whitespace'
MODEL_VARIANTS = {
    'gemma4-31b-iq4nl': {
        'label': 'Gemma 4 31B · IQ4_NL',
        'model_path': str(ROOT.parent / 'model/weights/gemma-4-31B-it-IQ4_NL.gguf'),
        'mmproj_path': str(ROOT.parent / 'model/weights/mmproj-BF16.gguf'),
    },
}
EMPTY_ALT_LINK_GATE = 'empty_alt_link_v1'
EMPTY_ALT_LINK_VISUAL_REVIEW = 'empty_alt_link_visual_review_v1'
SAME_LINK_ALT_REDUNDANCY_GATE = 'same_link_alt_redundancy_v1'
LINK_STRUCTURE_GATE = 'link_structure_v2'
DETERMINISTIC_BINARY_GATE_SOURCES = (
    EMPTY_ALT_LINK_GATE, SAME_LINK_ALT_REDUNDANCY_GATE)
REFLECTION_TRAIN = 'train_update_v1'
REFLECTION_FROZEN = 'frozen_guideline_v1'
REFLECTION_ERRORS_ONLY = 'initial_binary_errors_only_v1'
FROZEN_FEEDBACK_GATE = 'applicability_gate_v1'
FROZEN_EVIDENCE_PIPELINE = 'visual_coverage_v9'
TYPE_CANDIDATE_RECALL = 'recall_union_v1'
DECORATIVE_STRUCTURED_GATE = 'decorative_structured_gate_v2'
REFLECTION_DOCUMENT_COMPACT_THRESHOLD = 5000
REFLECTION_DOCUMENT_MAX_TOKENS = 6144
_MACHINE_LINK_PREFIX = re.compile(
    r'(?i)^(?:https?://|//|/|#|\.\.?/|www\.|mailto:|tel:|javascript:)')
_MACHINE_LINK_PATH = re.compile(r'(?i)^[^\s]+\.(?:do|jsp|html?|php|aspx?)(?:[?#]|$)')


def _machine_link_target(value):
    value = str(value or '').strip()
    return bool(_MACHINE_LINK_PREFIX.match(value)
                or _MACHINE_LINK_PATH.match(value)
                or ('?' in value and '=' in value))


def empty_alt_link_gate(data, *, legacy_readable_link=False):
    """Return a binary decision only for high-confidence empty-alt link structures."""
    alt = str((data or {}).get('alt_text') or '').strip()
    if alt or (data or {}).get('in_link') is not True:
        return None
    link_dest = str((data or {}).get('link_dest') or '').strip()
    if not link_dest or _MACHINE_LINK_PREFIX.match(link_dest):
        return {'verdict': '부적절',
                'reason': '빈 alt이며 사용자가 이해할 수 있는 같은 링크의 표시 문구가 확인되지 않는다.',
                'image_evidence': '링크 안의 이미지가 빈 alt로 낭독에서 제외된다.',
                'context_evidence': ('link_dest가 비어 있다.' if not link_dest
                                     else 'link_dest가 표시 문구가 아닌 URL·경로·앵커 형식이다.')}
    if legacy_readable_link and re.search(r'[가-힣A-Za-z0-9]', link_dest):
        return {'verdict': '적절',
                'reason': '같은 링크의 표시 문구가 목적을 전달하므로 이미지의 빈 alt가 중복 낭독을 피한다.',
                'image_evidence': '링크 안의 이미지가 빈 alt로 낭독에서 제외된다.',
                'context_evidence': f'같은 링크의 표시 문구: {link_dest[:300]}'}
    return None


def same_link_alt_redundancy_gate(data):
    """Return an improper verdict when a readable same-link label repeats a nonempty alt."""
    data = data or {}
    if data.get('in_link') is not True:
        return None
    alt = re.sub(r'\s+', ' ', str(data.get('alt_text') or '')).strip()
    link_dest = re.sub(r'\s+', ' ', str(data.get('link_dest') or '')).strip()
    if not alt or not link_dest or _machine_link_target(link_dest):
        return None
    comparable_alt = re.sub(r'\s*(?:이미지|아이콘)\s*$', '', alt,
                            flags=re.IGNORECASE).strip()
    if (not comparable_alt or comparable_alt.casefold() not in link_dest.casefold()
            or not re.search(r'[가-힣A-Za-z0-9]', link_dest)):
        return None
    return {
        'verdict': '부적절',
        'reason': '같은 링크의 표시 문구가 alt의 정보를 이미 포함해 중복 낭독된다.',
        'image_evidence': f'alt에서 비교한 문구: {comparable_alt[:300]}',
        'context_evidence': f'같은 링크의 표시 문구: {link_dest[:300]}',
    }


def link_structure_gate(data, gate_version, *, legacy_readable_link=False):
    """Return ``(decision, source)`` for versioned deterministic link gates."""
    if gate_version not in (EMPTY_ALT_LINK_GATE, LINK_STRUCTURE_GATE):
        return None, None
    decision = empty_alt_link_gate(data, legacy_readable_link=legacy_readable_link)
    if decision is not None:
        return decision, EMPTY_ALT_LINK_GATE
    if gate_version == LINK_STRUCTURE_GATE:
        decision = same_link_alt_redundancy_gate(data)
        if decision is not None:
            return decision, SAME_LINK_ALT_REDUNDANCY_GATE
    return None, None


def uses_empty_alt_link_visual_review(data):
    """Whether the official guide requires visual review before accepting an empty alt."""
    data = data or {}
    alt = str(data.get('alt_text') or '').strip()
    link_dest = str(data.get('link_dest') or '').strip()
    return bool(not alt and data.get('in_link') is True and link_dest
                and not _MACHINE_LINK_PREFIX.match(link_dest)
                and re.search(r'[가-힣A-Za-z0-9]', link_dest))
DEFAULT_MODEL_VARIANT = 'gemma4-31b-iq4nl'
SCHEMAS = {
    'classify': {'type': 'object', 'properties': {
        'label': {'type': 'string', 'enum': list(LABELS)},
        **{key: {'type': 'string', 'minLength': 1, 'maxLength': 350}
           for key in ('reason', 'image_evidence', 'context_evidence')}},
        'required': ['label', 'reason', 'image_evidence', 'context_evidence'], 'additionalProperties': False},
    'classify_types': {'type': 'object', 'properties': {
        'matched_types': {'type': 'array', 'maxItems': 6, 'items': {
            'type': 'object', 'properties': {
                'label': {'type': 'string', 'enum': list(LABELS[1:])},
                'reason': {'type': 'string', 'minLength': 1, 'maxLength': 350}},
            'required': ['label', 'reason'], 'additionalProperties': False}},
        **{key: {'type': 'string', 'minLength': 1, 'maxLength': 350}
           for key in ('reason', 'image_evidence', 'context_evidence')}},
        'required': ['matched_types', 'reason', 'image_evidence', 'context_evidence'], 'additionalProperties': False},
}
SCHEMAS['classify_verdict_types'] = {
    **SCHEMAS['classify_types'],
    'properties': {'verdict': {'type': 'string', 'enum': ['적절', '부적절']},
                   **SCHEMAS['classify_types']['properties']},
    'required': ['verdict', *SCHEMAS['classify_types']['required']],
}
SCHEMAS['binary_verdict'] = {
    'type': 'object', 'properties': {
        'verdict': {'type': 'string', 'enum': ['적절', '부적절']},
        **{key: {'type': 'string', 'minLength': 1, 'maxLength': 350}
           for key in ('reason', 'image_evidence', 'context_evidence')}},
    'required': ['verdict', 'reason', 'image_evidence', 'context_evidence'], 'additionalProperties': False,
}
IMAGE_ROLES = ('정보 전달', '기능 전달', '텍스트 전달', '자료 식별', '장식')
OBSERVATION_ROLES = ('기능성', '정보성', '텍스트 포함 이미지', '장식성')
ROLE_CRITERIA = {
    '정보 전달': '사진의 주요 대상·장면, 차트·도표의 핵심 수치나 결론을 전달한다.',
    '기능 전달': '링크의 목적지 또는 버튼의 동작을 이해할 수 있다.',
    '텍스트 전달': '이미지의 핵심 문구를 전달한다. 긴 내용은 핵심과 상세 설명 위치를 안내해도 된다.',
    '자료 식별': '책·간행물 등 해당 자료를 구별할 수 있는 제목·호수 등 식별 정보를 전달한다.',
    '장식': '제거해도 정보 손실이 없는 순수 장식은 빈 alt가 적절하고, 불필요한 내용성 alt는 부적절하다.',
}
SCHEMAS['image_role'] = {
    'type': 'object', 'properties': {
        'role': {'type': 'string', 'enum': list(IMAGE_ROLES)},
        'reason': {'type': 'string', 'minLength': 1, 'maxLength': 150}},
    'required': ['role', 'reason'], 'additionalProperties': False,
}
SCHEMAS['visual_evidence'] = {
    'type': 'object', 'properties': {
        'role': {'type': 'string', 'enum': list(IMAGE_ROLES)},
        'reason': {'type': 'string', 'minLength': 1, 'maxLength': 150},
        **{key: {'type': 'array', 'maxItems': 3,
                 'items': {'type': 'string', 'minLength': 1, 'maxLength': 150}}
           for key in ('visible_facts', 'context_quotes', 'required_information')},
    },
    'required': ['role', 'reason', 'visible_facts', 'context_quotes', 'required_information'],
    'additionalProperties': False,
}
SCHEMAS['image_role_observation'] = {
    'type': 'object', 'properties': {
        'roles': {'type': 'array', 'minItems': 1, 'maxItems': 4, 'uniqueItems': True,
                  'items': {'type': 'string', 'enum': list(OBSERVATION_ROLES)}},
        'description': {'type': 'string', 'minLength': 1, 'maxLength': 350},
    },
    'required': ['roles', 'description'], 'additionalProperties': False,
}
SCHEMAS['evidence_verdict'] = {
    'type': 'object', 'properties': {
        'verdict': {'type': 'string', 'enum': ['적절', '부적절']},
        **{key: {'type': 'string', 'minLength': 1, 'maxLength': 350}
           for key in ('reason', 'image_evidence', 'context_evidence')},
        'checks': {'type': 'array', 'maxItems': 3, 'items': {
            'type': 'object', 'properties': {
                'information': {'type': 'string', 'minLength': 1, 'maxLength': 350},
                'alt_quote': {'type': 'string', 'maxLength': 350},
                'context_quote': {'type': 'string', 'maxLength': 350},
                'status': {'type': 'string', 'enum': ['충족', '미충족', '확인불가']},
            },
            'required': ['information', 'alt_quote', 'context_quote', 'status'],
            'additionalProperties': False}},
        'needs_review': {'type': 'boolean'},
    },
    'required': ['verdict', 'reason', 'image_evidence', 'context_evidence', 'checks', 'needs_review'],
    'additionalProperties': False,
}
SCHEMAS['role_verdict_v1'] = {
    'type': 'object', 'properties': {
        'image_roles': {'type': 'array', 'minItems': 1, 'maxItems': 5,
                        'items': {'type': 'string', 'enum': list(IMAGE_ROLES)}},
        'image_evidence': SCHEMAS['binary_verdict']['properties']['image_evidence'],
        'criteria_checks': {'type': 'array', 'minItems': 1, 'maxItems': 6, 'items': {
            'type': 'object', 'properties': {
                **{key: {'type': 'string', 'minLength': 1, 'maxLength': 350}
                   for key in ('criterion', 'alt_evidence', 'context_evidence')},
                'status': {'type': 'string', 'enum': ['충족', '미충족', '확인불가']}},
            'required': ['criterion', 'alt_evidence', 'context_evidence', 'status'],
            'additionalProperties': False}},
        **{key: SCHEMAS['binary_verdict']['properties'][key]
           for key in ('context_evidence', 'reason', 'verdict')}},
    'required': ['image_roles', 'image_evidence', 'criteria_checks', 'context_evidence', 'reason', 'verdict'],
    'additionalProperties': False,
}
SCHEMAS['error_types'] = {
    'type': 'object', 'properties': {
        'matched_types': {'type': 'array', 'minItems': 1, 'maxItems': 6, 'items': {
            'type': 'object', 'properties': {
                'label': {'type': 'string', 'enum': list(LABELS[1:])},
                'reason': {'type': 'string', 'minLength': 1, 'maxLength': 350}},
            'required': ['label', 'reason'], 'additionalProperties': False}},
        **{key: {'type': 'string', 'minLength': 1, 'maxLength': 350}
           for key in ('reason', 'image_evidence', 'context_evidence')}},
    'required': ['matched_types', 'reason', 'image_evidence', 'context_evidence'], 'additionalProperties': False,
}
SCHEMAS['final_error_type'] = {
    'type': 'object', 'properties': {
        'label': {'type': 'string', 'enum': list(LABELS[1:])},
        **{key: {'type': 'string', 'minLength': 1, 'maxLength': 350}
           for key in ('reason', 'image_evidence', 'context_evidence')},
    },
    'required': ['label', 'reason', 'image_evidence', 'context_evidence'], 'additionalProperties': False,
}
SCHEMAS['decorative_observation'] = {
    'type': 'object',
    'properties': {
        'visual_kind': {'type': 'string', 'enum': [
            'simple_icon', 'identity_symbol', 'theme_visual', 'photo', 'ui_screenshot',
            'information_graphic', 'text_rich_graphic', 'other']},
        'other_error': {'type': 'string', 'enum': [
            'filename', 'meaningless', 'mismatch', 'same_link_duplicate',
            'information_omission', 'none']},
        'pure_decorative_role': {'type': 'boolean'},
        'decorative_pattern': {'type': 'string', 'enum': [
            'error_status_icon', 'redundant_identity_visual', 'theme_background',
            'auxiliary_link_icon', 'promotional_atmosphere', 'none']},
        'confidence': {'type': 'string', 'enum': ['high', 'medium', 'low']},
        'reason': {'type': 'string', 'maxLength': 350},
    },
    'required': ['visual_kind', 'other_error', 'pure_decorative_role',
                 'decorative_pattern', 'confidence', 'reason'],
    'additionalProperties': False,
}


DECORATIVE_ROUTER_PROMPT = '''[Instruction]
원본 이미지와 페이지 문맥만 보고 이미지의 역할을 판정하라. 대체텍스트 내용은 제공되지 않는다. 장식 가능성을 놓치지 않는 후보 라우터이며 최종 alt 품질 판정이 아니다.

[판정 기준]
- verdict="적절": 이미지를 제거해도 고유 정보·목적지·동작·상태가 사라지지 않는 장식성 이미지이거나, 장식성 여부를 확실히 배제할 수 없다. 다음 전용 검산으로 보낸다.
- verdict="부적절": 이미지가 고유 정보·목적지·동작·상태를 명확히 전달하는 정보성·기능성 이미지라서 장식성을 확실히 배제할 수 있다. 다음 검산을 생략한다.

[Rule]
1. page_title, context_text, in_link, link_dest, heading_path와 이미지가 각각 제공하는 정보를 비교한다.
2. 로고·아이콘·마스코트·분위기용 사진·섹션 상징 그래픽은 주변 문맥과 같은 내용만 반복하는지 확인한다.
3. 확실하지 않으면 verdict="적절"로 보내 재검산한다.
4. 아래 JSON 객체 하나만 반환한다. 각 근거는 한국어 350자 이내다.

[Output format]
{"verdict":"적절","reason":"장식 가능성 또는 불확실성 근거","image_evidence":"이미지의 역할 근거","context_evidence":"페이지 문맥의 대체 정보 근거"}
또는 verdict만 "부적절"로 바꾼 같은 형식의 객체 하나를 반환한다.

[Output]'''


DECORATIVE_OBSERVER_PROMPT = '''[Instruction]
원본 이미지와 alt_text, 페이지 문맥을 관찰해 아래 여섯 필드를 채워라. 최종 라벨을 고르지 말고 관찰 결과만 구조화하라.

[필드 판정]
1. visual_kind
- simple_icon: 화살표·글머리·단순 상태/강조 아이콘
- identity_symbol: 로고·국기·마스코트·기관/서비스 상징
- theme_visual: 추상 일러스트·배경·스톡 이미지·주제 분위기 이미지
- photo: 실제 인물·장소·시설·행사·사물을 촬영한 사진
- ui_screenshot: 화면·앱·설정·조작 안내
- information_graphic: 차트·표·지도·문서·절차도·흐름도·인포그래픽
- text_rich_graphic: 기간·대상·장소·방법·수치 등 제목 외 실질 문구가 있는 포스터·배너·카드뉴스
- other

2. other_error
- filename: 파일명·경로·확장자·편집 흔적
- meaningless: "이미지", "사진", 숫자·기호 한 글자처럼 내용 없음
- mismatch: alt와 이미지 내용이 다름
- same_link_duplicate: in_link=true이며 정규화한 alt 전체가 link_dest에 그대로 포함됨. 단, alt가 링크 문구에 없는 "바로가기 화살표" 같은 보조 모양을 설명하면 해당하지 않음
- information_omission: 정보성 사진·화면·도표·포스터가 가진 장면·조작 위치·단계·수치·기간·대상·장소·방법을 alt가 빠뜨림
- none

3. pure_decorative_role
이 데이터셋에서 올바른 alt가 빈 문자열이어야 하면 true다. 이미지 자체의 종류보다 현재 배치에서 추가로 전달하는 정보가 있는지를 본다.
- 인접 문장이나 같은 링크의 사람이 읽는 문구가 대상·목적지·동작·상태를 이미 완전히 전달하고, 아이콘·로고·마스코트·국기·화살표·분위기 이미지의 alt가 그 문구를 반복하면 true다.
- 완전한 상태 문장 옆 아이콘, 제목 옆 주제 일러스트, 구체적 사건 기록이 아닌 섹션용 스톡 사진·대표 사진은 true다. 넓은 섹션명 alt가 사진의 구체적 장면과 다르더라도 사진이 분위기만 보조한다면 true다.
- 사진의 구체적 인물·장소·행동, UI의 고유 동작·상태, 도표·절차·지도·포스터의 핵심 정보가 이미지에만 있으면 false다.
- 조항·단계·범주를 구별하는 정보성 아이콘은 alt가 그 범주를 식별하면 false다. 특히 개인정보처리방침 목차처럼 각 조항 아이콘의 alt가 조항 성격을 알려 주는 경우, 인접 제목과 비슷해도 false다.
- 링크가 이미지 하나뿐이고 주변에 목적지를 읽어 주는 문구가 없으면 기능성 이미지이므로 false다. URL·경로뿐인 link_dest는 사람이 읽는 문구로 보지 않는다.

4. decorative_pattern
- error_status_icon: 완전한 오류·상태 문장 옆 단순 아이콘
- redundant_identity_visual: 이름과 기능이 문맥에 이미 있는 비링크 로고·국기·마스코트
- theme_background: 제목 옆 분위기용 배경·일러스트·스톡 이미지
- auxiliary_link_icon: 목적지·동작을 완전히 제공하는 링크의 화살표·글머리·강조 아이콘
- promotional_atmosphere: 링크 밖 카드에서 실제 장면 기록이 아니라 분위기만 보조하는 사진·홍보 이미지
- none

5. confidence: high, medium, low 중 하나다.
6. reason: 위 필드의 근거를 한국어 350자 이내로 쓴다.

[필수 경계]
- 설정/앱 화면, 여러 단계·화살표·순서를 담은 절차도·흐름도는 본문이 같은 내용을 제공해도 정보성이다.
- 촬영 사진은 실제 장면을 전달하는지, 분위기만 보조하는지 구분한다.
- "에러", "경고", 구체적인 아이콘명·마스코트명·기관명은 무의미 표현이 아니다.
- 장식 역할과 다른 오류가 함께 떠오르면 other_error도 기록하되 pure_decorative_role 판단을 숨기지 않는다.

[Output format]
{"visual_kind":"simple_icon","other_error":"none","pure_decorative_role":true,"decorative_pattern":"error_status_icon","confidence":"high","reason":"구조화 관찰 근거"}

[Output]'''


def decorative_observation_decision(value, data):
    """Accept only a confident, structured decorative-role observation."""
    data = data or {}
    visual = value['visual_kind']
    error = value['other_error']
    pattern = value['decorative_pattern']
    role = value['pure_decorative_role']
    confidence = value['confidence']
    if confidence == 'low':
        return False
    if visual in ('photo', 'ui_screenshot', 'information_graphic',
                  'text_rich_graphic', 'identity_symbol', 'other'):
        return False
    if same_link_alt_redundancy_gate(data):
        return False
    if data.get('in_link') is True and _machine_link_target(data.get('link_dest')):
        return False
    alt = re.sub(r'\s+', '', str(data.get('alt_text') or '')).casefold()
    context = re.sub(r'\s+', '', str(data.get('context_text') or '')).casefold()
    if visual == 'simple_icon' and alt in ('new', 'newicon', 'new아이콘', '새글', '신규'):
        return False
    if visual == 'simple_icon' and not data.get('in_link'):
        if alt and alt == context:
            return False
    if error != 'none':
        return False
    return bool(role or pattern == 'error_status_icon')


def final_candidate_labels(assessment):
    """Return the fixed, ordered final-label pool from a candidate assessment."""
    matches = assessment.get('matched_types') if isinstance(assessment, dict) else None
    if not isinstance(matches, list):
        raise ValueError('최종 오류 유형 후보 집합이 없습니다.')
    labels = []
    for item in matches:
        label = item.get('label') if isinstance(item, dict) else None
        if label not in ERROR_PRIORITY:
            raise ValueError('최종 오류 유형 후보 집합이 잘못되었습니다.')
        if label not in labels:
            labels.append(label)
    if not labels:
        raise ValueError('최종 오류 유형 후보 집합이 비어 있습니다.')
    return labels


def final_metadata_gate(data, candidates, initial_final=None):
    """Resolve repeated metadata-only boundaries when the target stays in the fixed pool."""
    data = data or {}
    labels = set(final_candidate_labels(candidates))
    alt = re.sub(r'\s+', ' ', str(data.get('alt_text') or '')).strip()
    link = re.sub(r'\s+', ' ', str(data.get('link_dest') or '')).strip()
    label = reason = None
    if alt == '해당 이미지 아래 설명 추가' and '불충분형' in labels:
        label, reason = '불충분형', '작성 지시문은 설명할 의도만 나타내고 이미지의 핵심 정보를 제공하지 않는다.'
    elif alt == '미리보기' and '불충분형' in labels:
        label, reason = '불충분형', '미리보기는 동작만 전달하고 대상·목적지·이미지 핵심 정보를 빠뜨린다.'
    elif alt == 'external_image' and '무의미형' in labels:
        label, reason = '무의미형', 'external_image는 이미지의 대상·역할·목적을 알 수 없는 범용 기술 토큰이다.'
    elif alt == '캡차이미지' and '불충분형' in labels:
        label, reason = '불충분형', '캡차의 유형만 알리고 사용자가 입력해야 할 이미지 속 값을 제공하지 않는다.'
    elif (data.get('in_link') is True and alt and alt.casefold() in link.casefold()
          and '홈페이지' in link and '홈페이지' not in alt and '불충분형' in labels):
        label, reason = '불충분형', 'alt가 서비스 이름만 제공하고 같은 링크 문구에 드러난 홈페이지 목적지를 빠뜨린다.'
    elif re.fullmatch(r'콘텐츠\s*\(?\d+\)?', alt, re.IGNORECASE) and '무의미형' in labels:
        label, reason = '무의미형', '일반 명칭과 순번만으로는 이미지의 대상·목적·동작을 식별할 수 없다.'
    elif re.match(r'^/?\d+\s*페이지(?:_|\s)', alt) and '불충분형' in labels:
        label, reason = '불충분형', '페이지 번호는 파일명 근거가 아니며 나머지 역할 표시는 이미지의 고유 대상·상태를 충분히 전달하지 못한다.'
    elif (re.search(r'\s+(?:img|이미지|사진)$', alt, re.IGNORECASE)
          and (initial_final or {}).get('label') in ('파일명형', '무의미형')
          and '불충분형' in labels):
        label, reason = '불충분형', '구체 대상명 뒤의 이미지 유형 접미사는 파일명이나 무의미 표현이 아니며, 남은 후보 중 정보 충분성을 검토해야 한다.'
    if label is None:
        return None
    return {
        'label': label, 'reason': reason,
        'image_evidence': candidates.get('image_evidence', reason),
        'context_evidence': candidates.get('context_evidence', reason),
    }


def expand_type_candidates(initial, revised):
    """Keep every initial candidate and append only newly found candidates."""
    initial_labels = final_candidate_labels(initial)
    revised_labels = final_candidate_labels(revised)
    revised_by_label = {item['label']: item for item in revised['matched_types']}
    initial_by_label = {item['label']: item for item in initial['matched_types']}
    merged = [revised_by_label.get(label, initial_by_label[label]) for label in initial_labels]
    merged.extend(revised_by_label[label] for label in revised_labels if label not in initial_by_label)
    return {**revised, 'matched_types': merged}


def candidate_priority_decision(assessment, priority):
    """Build an auditable final decision without inventing evidence."""
    labels = final_candidate_labels(assessment)
    winner = next((label for label in priority if label in labels), None)
    if winner is None:
        raise ValueError('최종 오류 유형 후보 우선순위가 잘못되었습니다.')
    match = next(item for item in assessment['matched_types'] if item['label'] == winner)
    return {
        'label': winner,
        'reason': match['reason'],
        'image_evidence': assessment['image_evidence'],
        'context_evidence': assessment['context_evidence'],
    }


def enforce_final_candidate_pool(decision, assessment, priority):
    """Recover a schema-contract violation using the established candidate priority."""
    labels = final_candidate_labels(assessment)
    if isinstance(decision, dict) and decision.get('label') in labels:
        return decision, None
    recovered = candidate_priority_decision(assessment, priority)
    return recovered, {
        'rule': 'candidate_priority_contract_fallback_v1',
        'invalid_decision': decision,
        'recovered_decision': recovered,
    }
SCHEMAS['reflection_feedback'] = {
    'type': 'object', 'properties': {
        'feedback': {'type': 'string', 'minLength': 1, 'maxLength': 350},
    },
    'required': ['feedback'], 'additionalProperties': False,
}
SCHEMAS['reflection_feedback_gate'] = {
    'type': 'object', 'properties': {
        'action': {'type': 'string', 'enum': ['피드백 제공', '통과']},
        'feedback': {'type': 'string', 'maxLength': 350},
        'reason': {'type': 'string', 'minLength': 1, 'maxLength': 350},
    },
    'required': ['action', 'feedback', 'reason'], 'additionalProperties': False,
}
SCHEMAS['reflection_document'] = {
    'type': 'object', 'properties': {
        # Length is managed by Python's prompt branch rather than JSON grammar.
        'markdown': {'type': 'string'},
    },
    'required': ['markdown'], 'additionalProperties': False,
}


def _exit_when_parent_dies(expected_parent):
    # Linux workers are single-threaded at spawn. Also close the race where the
    # parent dies between fork() and prctl(); SIGKILL releases CUDA on OOM/crash.
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_PDEATHSIG')
    if os.getppid() != expected_parent:
        os._exit(1)


def owned_process(command, **kwargs):
    return subprocess.Popen(command, preexec_fn=partial(_exit_when_parent_dies, os.getpid()), **kwargs)


def _guideline_sections_1_to_3_and_5():
    source = (ROOT.parent / 'data/docs/라벨링-가이드라인-v1.1.md').read_text(encoding='utf-8')
    sections_1_to_3 = source[source.index('## 1.'):source.index('## 4.')].rstrip()
    section_5 = source[source.index('## 5.'):source.index('## 6.')].rstrip()
    return f'{sections_1_to_3}\n\n{section_5}'


def _guideline_binary_prompt(official, reflection=False):
    instruction = (
        '최초 적절 판정이 공식 기준을 빠뜨렸는지 독립적으로 재검토하라. 최초 판정을 그대로 따르지 말고 '
        '원본 이미지·메타데이터와 공식 기준으로 다시 판단하라.' if reflection else
        '원본 이미지와 alt_text를 포함한 전체 메타데이터를 공식 기준에 따라 검토하고 '
        'alt_text를 "적절" 또는 "부적절"로 이진 판정하라.')
    initial = ('- initial_assessment: 별도 세션에서 받은 최초 이진판정 JSON. 반박 가능한 참고 자료이며 정답이 아니다.\n'
               if reflection else '')
    return f'''[Instruction]
{instruction}

[Input]
- image: 상자나 주석 없는 원본 이미지.
- alt_text: 평가할 대체텍스트 원문. 빈 문자열도 공식 기준에 따라 같은 절차로 판정한다.
- page_title: 수집 페이지 제목.
- context_text: 이미지 인접 텍스트.
- in_link: 링크·버튼 내부 여부.
- link_dest: 수집된 링크 관련 문자열.
- heading_path: 문서 내 헤딩 경로.
- kogl_marker: 수집 페이지의 공공누리 표시 여부.
{initial}
[공식 판정 기준]
아래는 「대체텍스트 품질 라벨링 가이드라인 v1.1」의 판정 관련 1~3절과 판정 범위를 정한 5절 원문이다.

{official}

[판정 절차]
1. alt_text를 평가하기 전에 이미지와 문맥으로 기능성·정보성·텍스트 포함·장식성 역할을 확인한다.
2. 그 역할에서 반드시 전달해야 하는 정보나 기능을 구체적으로 확인한다.
3. alt_text와 실제 인접·링크 문구가 필요한 정보를 전달하는지 원본과 대조한다.
4. 공식 6개 오류 유형 중 하나라도 충족하면 "부적절", 어느 유형에도 해당하지 않고 역할에 필요한 정보를 전달하면 "적절"이다.
5. 빈 alt도 별도 예외로 처리하지 않고, 장식성·링크 문구의 대체 여부·이미지의 추가 정보 누락을 공식 규칙대로 확인한다.
6. 보류는 출력하지 않는다. 주어진 이미지와 메타데이터 안에서 가장 근거가 강한 이진판정을 선택한다.

[Rule]
1. 오류 유형명은 최종 출력하지 않는다. reason에는 적용한 공식 기준과 대조 결과를 쓴다.
2. 입력에 없는 DOM 구조·주변 문구·이미지 정보를 만들어내지 않는다.
3. 아래 형식의 JSON 객체 하나만 반환한다. 각 근거는 한국어 350자 이내다.

[Output format]
{{"verdict":"적절","reason":"공식 기준에 따른 판정 근거","image_evidence":"이미지에서 확인한 내용","context_evidence":"메타데이터에서 확인한 내용"}}
또는 verdict만 "부적절"로 바꾼 같은 형식의 객체 하나를 반환한다.

[Output]'''


def _empty_alt_link_visual_review_prompt(official):
    return f'''[Instruction]
빈 alt와 사람이 읽을 수 있는 같은 링크의 문구가 있는 썸네일을 검사하라. 링크 문구와 인접 문맥에 없는 실질 정보가 이미지에만 있는지를 판정하라.

[Input]
- image: 상자나 주석 없는 원본 이미지.
- alt_text: 빈 문자열인 대체텍스트.
- page_title: 수집 페이지 제목.
- context_text: 이미지 인접 텍스트.
- in_link: true인 링크·버튼 내부 여부.
- link_dest: 사람이 읽을 수 있는 같은 링크의 표시 문구.
- heading_path: 문서 내 헤딩 경로.
- kogl_marker: 수집 페이지의 공공누리 표시 여부.

[공식 판정 기준]
아래는 「대체텍스트 품질 라벨링 가이드라인 v1.1」의 판정 관련 1~3절과 판정 범위를 정한 5절 원문이다.

{official}

[판정 절차]
1. 이미지가 정보 전달용 그래픽인지 현장 사진인지 확인한다.
2. 정보 전달용 그래픽이면 기간·일시, 장소, 대상·자격요건, 금액·수치, 참여방법 같은 실질 정보를 확인한다.
3. 그 실질 정보가 link_dest와 context_text를 합쳐도 없고 이미지에만 있으면 "부적절"이다.
4. 현장 사진이거나 이미지가 더하는 것이 제목 반복·기관명·로고·발간등록번호·슬로건뿐이면 "적절"이다.

[Rule]
1. 등록일·게시일은 행사 기간으로 간주하지 않는다.
2. 이미지 안의 현수막·패널 글자가 현장 사진의 배경인 경우 정보 전달용 그래픽으로 바꾸어 판단하지 않는다.
3. verdict는 "적절" 또는 "부적절" 중 하나다.
4. 아래 형식의 JSON 객체 하나만 반환한다. 각 근거는 한국어 350자 이내다.

[Output format]
{{"verdict":"적절","reason":"판정 근거","image_evidence":"이미지에서 확인한 내용","context_evidence":"링크 문구와 인접 문맥에서 확인한 내용"}}

[Output]'''


def _guideline_types_prompt(official, independent=False):
    binary_input = ('' if independent else
                    '- binary_assessment: 앞 단계의 최종 적절/부적절 판정 JSON. 판정 근거로 참고하되 원본과 다시 대조한다.\n')
    evidence = '원본 이미지·메타데이터' if independent else '원본 이미지·메타데이터·binary_assessment'
    return f'''[Instruction]
최종 이진판정이 "부적절"인 alt_text에 적용되는 공식 오류 유형을 모두 찾아 후보 집합을 생성하라.

[Input]
- image: 상자나 주석 없는 원본 이미지.
- alt_text, page_title, context_text, in_link, link_dest, heading_path, kogl_marker: 평가 입력 메타데이터.
{binary_input.rstrip()}

[공식 판정 기준]
아래는 「대체텍스트 품질 라벨링 가이드라인 v1.1」의 판정 관련 1~3절과 판정 범위를 정한 5절 원문이다.

{official}

[판정 절차]
1. 공식 6개 부적절 유형을 각각 독립적으로 {evidence}와 대조한다.
2. 이 단계는 최종 선택이 아니라 정답 누락을 줄이는 후보 생성이므로 후보 재현율을 우선한다.
3. 유형을 뒷받침하는 관찰 근거가 하나라도 있거나 명확한 배제 근거가 없으면 matched_types에 포함한다.
4. 입력이 공식 정의와 직접 모순되어 해당 유형을 명확히 배제할 수 있을 때만 제외한다.
5. 둘 이상이 성립할 가능성이 있으면 모두 남긴다. 이 단계에서는 후보를 좁히거나 최종 유형 하나를 선택하지 않는다.

[Rule]
1. matched_types는 1개 이상 6개 이하이며 같은 라벨을 중복하지 않는다.
2. 각 후보 reason에는 해당 공식 기준과 후보로 유지한 관찰 근거 또는 배제하지 못한 불확실성을 쓴다.
3. 아래 형식의 JSON 객체 하나만 반환한다. 각 근거는 한국어 350자 이내다.

[Output format]
{{"matched_types":[{{"label":"불충분형","reason":"후보 근거"}}],"reason":"후보 집합의 종합 근거","image_evidence":"이미지에서 확인한 내용","context_evidence":"메타데이터에서 확인한 내용"}}

[Output]'''


def _guideline_final_prompt(official, independent=False):
    binary_input = ('' if independent else
                    '- binary_assessment: 앞 단계의 최종 적절/부적절 판정 JSON.\n')
    return f'''[Instruction]
공식 가이드라인과 원본 입력을 다시 확인하여 candidate_assessment의 오류 유형 후보 중 최종 라벨 하나를 선택하라.

[Input]
- image: 상자나 주석 없는 원본 이미지.
- alt_text, page_title, context_text, in_link, link_dest, heading_path, kogl_marker: 평가 입력 메타데이터.
{binary_input.rstrip()}
- candidate_assessment: 앞 단계가 생성한 복수 오류 유형 후보 JSON.

[공식 판정 기준]
아래는 「대체텍스트 품질 라벨링 가이드라인 v1.1」의 판정 관련 1~3절과 판정 범위를 정한 5절 원문이다.

{official}

[판정 절차]
1. 각 후보가 공식 정의와 경계 규칙을 충족하는지 원본 이미지·메타데이터와 다시 대조한다.
2. 실제로 성립하는 후보가 둘 이상이면 공식 가이드라인의 단일 라벨 우선순위를 적용한다.
3. candidate_assessment의 matched_types 안에서 최종 라벨 하나를 선택한다.

[Rule]
1. label은 candidate_assessment.matched_types에 포함된 라벨 중 하나여야 한다.
2. reason에는 선택한 정의·경계 규칙·우선순위 적용 근거를 쓴다.
3. 아래 형식의 JSON 객체 하나만 반환한다. 각 근거는 한국어 350자 이내다.

[Output format]
{{"label":"불충분형","reason":"최종 선택 근거","image_evidence":"이미지에서 확인한 내용","context_evidence":"메타데이터에서 확인한 내용"}}

[Output]'''


def _reflection_feedback_prompt(official):
    return f'''[Instruction]
학습 데이터의 최초 이진판정을 독립적으로 검토하라. 최초 판정이 맞았다고 전제하거나 틀렸다고 전제하지 말고, 재판정 모델이 원본 입력에서 반드시 다시 확인해야 할 한 가지 기준을 일반화된 피드백 1~2문장으로 작성하라.

[Input]
- image: 상자나 주석 없는 원본 이미지.
- alt_text, page_title, context_text, in_link, link_dest, heading_path, kogl_marker: 최초 이진판정과 같은 메타데이터.
- initial_assessment: 최초 이진판정 JSON.
- actual_label: 학습 데이터의 실제 7범주 라벨.
- current_guidelines: 지금까지 누적된 Reflection 판단 가이드라인.

[공식 판정 기준]
{official}

[Rule]
1. actual_label을 이진 기준으로 바꾸어 `적절`은 적절, 나머지 6개 유형은 부적절로 본 뒤 initial_assessment의 verdict와 일치하는지 먼저 확인한다.
2. 두 이진 판정이 일치하면 최초 결론을 뒷받침하는 핵심 근거를 다시 확인하도록 쓴다. 존재하지 않는 결함을 찾거나 반대 결론으로 바꾸도록 유도하지 않는다. 두 판정이 다르면 최초 판정이 놓친 비교 조건을 다시 확인하도록 쓴다.
3. current_guidelines와 일관되게 작성한다. 새 사례가 기존 기준과 충돌해 보이면 적용 조건이나 예외를 좁혀 충돌을 해소한다.
4. actual_label은 피드백의 검토 방향을 정하는 감독 정보로만 사용한다. 정답 라벨이나 최종 판정 방향을 직접 말하지 않는다. "적절/부적절/중복형으로 판정하라"처럼 결론을 지시하지 않는다.
5. feedback은 반복적인 서론 없이 핵심 비교 조건부터 바로 시작한다. 한 가지 조건을 "~인지 다시 확인하라" 또는 "~를 다시 고려하라"라는 재검토 지시로 쓰며, 다시 확인하라/고려하라 형식을 사용한다.
6. 검토와 관련된 입력 필드 이름(alt_text, context_text, in_link, link_dest 등) 또는 이미지 관찰 대상을 명시한다. 관련 입력 필드 이름은 쓰되 필드의 고유 문자열을 그대로 옮기지는 않는다.
7. 현재 문제의 record ID·고유명사·URL·날짜를 암기하는 문장을 쓰지 않는다.
8. 다른 문제에도 적용할 수 있는 관찰 조건만 350자 이내로 쓴다.
9. 아래 JSON 객체 하나만 반환한다.

[Output format]
{{"feedback":"인접 텍스트가 이미지의 기능과 정보를 이미 모두 전달하는지 다시 확인하라."}}

[Output]'''


def _frozen_reflection_feedback_prompt(official):
    return f'''[Instruction]
원본 이미지와 메타데이터를 먼저 대조하여 최초 이진판정에 독립 재검토가 필요한 구체적인 위험 신호가 있는지 선별하라. 위험 신호가 하나라도 원본에서 확인되면 "피드백 제공", 판정에 필요한 근거가 이미 직접 대응되면 "통과"를 선택하라.

[Input]
- image: 상자나 주석 없는 원본 이미지.
- alt_text, page_title, context_text, in_link, link_dest, heading_path, kogl_marker: 최초 이진판정과 같은 메타데이터.
- initial_assessment: 최초 이진판정 JSON.
- current_guidelines: train에서 학습을 마친 뒤 고정한 Reflection 판단 가이드라인.

[공식 판정 기준]
{official}

[선별 절차]
1. 이미지가 기능성·정보성·텍스트 포함·장식성 중 어떤 역할인지 확인하고, 그 역할에서 필요한 핵심 정보나 기능을 최대 3개로 특정한다.
2. 각 핵심 항목이 alt_text, 실제 context_text, 사람이 읽는 link_dest의 어느 표현으로 전달되는지 대조한다. page_title·heading_path·URL·경로는 직접 대체 문구로 세지 않는다.
3. initial_assessment가 위 대응을 실제 근거로 제시했는지 확인한다. "문맥이 충분하다", "주제가 같다", "장식이다"처럼 구체적 대응이나 제거 가능성을 설명하지 못한 근거는 재검토 완료로 보지 않는다.

[재검토 위험 신호]
- 포스터·배너·카드뉴스·화면 캡처·지도·도표·절차도·QR처럼 실질 정보가 있는 이미지에서 기간·대상·장소·방법·수치·상태·목적지 중 보이는 항목과 대체 문구의 대응이 확인되지 않는다.
- 특정 인물·행사·시설·행동을 기록한 사진인데 alt와 직접 인접한 캡션은 페이지 주제나 제목만 말하고 실제 대상·행동·장면을 설명하지 않는다.
- 링크 밖 빈 alt인데 실제 인접 캡션이 이미지의 대상·행동·장면을 직접 대신하는지, 또는 이미지가 고유 정보를 주지 않는 장식·대표 이미지인지 최초 근거에서 확인되지 않는다.
- 최초 적절 근거가 page_title·heading_path·URL·일반적인 본문 주제만으로 이미지에만 있는 정보를 보완했다고 간주한다.
- 최초 부적절 근거가 구체적인 대상명·시설명·장소명 뒤의 `img`·`이미지`·`사진` 접미사나 더 자세한 장면 묘사의 부재만 지적한다. 대상 식별만으로 현재 이미지 역할을 충족하는지 누적 기준과 다시 대조한다.
- 최초 부적절 근거가 `N페이지_`·번호·밑줄 형식만으로 파일명 흔적을 확정하지만, 이를 제외한 나머지가 실제 대상·처리 전후·복원 상태·디지털화 결과를 자연어로 설명한다.
- 최초 부적절 근거가 `첨부파일 아이콘`·`이미지 파일`·`이미지없음`처럼 목록의 파일 존재·형식이나 플레이스홀더 상태를 직접 알리는 문구를 단순히 `아이콘`·`이미지`라는 단어가 있다는 이유로 상투어로 취급한다.

[Rule]
1. 위 위험 신호가 원본에서 하나라도 확인되면 "피드백 제공"을 선택한다. 최초 결론을 바꾸라고 지시하지 말고, 확인되지 않은 핵심 항목 하나를 원본과 대체 문구에서 다시 대조하도록 쓴다.
2. 핵심 항목이 alt_text·실제 context_text·사람이 읽는 link_dest에 구체적으로 대응되거나, 이미지가 순수 장식·고유 사실 없는 일반 대표 이미지임이 확인되면 "통과"한다.
3. 최초 부적절 판정이 누락·불일치·기능 부재를 구체적인 원본 근거로 이미 확인했다면 통과한다.
4. 작은 글자나 보이지 않는 정보를 만들어내지 않는다. 이미지에서 실제로 확인할 수 있는 대상만 위험 신호로 사용한다.
5. "피드백 제공"이면 feedback을 `확인 대상: ... / 현재 전달 근거: ... / 다시 대조할 점: ...인지 다시 확인하라.` 형식으로 쓰고, "통과"이면 빈 문자열로 쓴다.
6. feedback과 reason은 각각 350자 이내이며 아래 JSON 객체 하나만 반환한다.

[Output format]
{{"action":"피드백 제공","feedback":"확인 대상: 이미지의 핵심 정보 / 현재 전달 근거: alt와 실제 인접 문구 / 다시 대조할 점: 누락된 핵심 항목이 있는지 다시 확인하라.","reason":"정보성 이미지와 대체 문구 사이에 구체적인 미확인 대응이 있다."}}
또는
{{"action":"통과","feedback":"","reason":"필요한 핵심 정보가 실제 대체 문구에 대응되거나 이미지가 고유 정보를 주지 않는다."}}

[Output]'''


def _frozen_visual_evidence_prompt(official):
    return f'''[Instruction]
alt_text와 이전 판정을 보지 말고 원본 이미지와 페이지 문맥에서 관찰 가능한 사실만 추출하라. 적절·부적절 결론을 내리지 않는다.

[Input]
- image: 상자나 주석 없는 원본 이미지.
- page_title, context_text, in_link, link_dest, heading_path, kogl_marker: 페이지 문맥.

[공식 판정 기준]
{official}

[추출 절차]
1. 현재 배치에서 이미지의 주된 역할 하나를 고른다.
2. 이미지에서 직접 확인되는 대상·행동·상태·문구·수치·기간·장소·단계·목적지를 최대 3개로 적는다.
3. 실제 context_text와 사람이 읽는 link_dest에서 직접 확인되는 대응 문구를 최대 3개로 적는다. page_title·heading_path·URL·경로는 대응 문구로 세지 않는다.
4. 이미지 역할상 대체 수단이 반드시 전달해야 할 정보를 최대 3개로 적는다. 순수 장식이라면 제거해도 사라지는 고유 정보가 없다는 조건을 적는다.

[Rule]
1. alt_text의 내용이나 적절성은 추측하지 않는다.
2. 적절·부적절, 충족·미충족 결론을 쓰지 않는다.
3. 작은 글자나 보이지 않는 사실을 만들어내지 않는다.
4. 아래 JSON 객체 하나만 반환한다.

[Output format]
{{"role":"정보 전달","reason":"역할을 정한 시각·문맥 근거","visible_facts":["이미지에서 직접 확인한 사실"],"context_quotes":["실제 인접·링크 문구"],"required_information":["대체 수단이 전달해야 할 정보"]}}

[Output]'''


def _frozen_coverage_audit_prompt(official):
    return f'''[Instruction]
앞 단계가 alt_text 없이 추출한 시각 증거 목록과 실제 alt_text·페이지 문맥을 항목별로 대조하여 적절·부적절을 판정하라.

[Input]
- image: 원본 이미지. 증거 목록의 명백한 오독만 확인한다.
- alt_text, page_title, context_text, in_link, link_dest, heading_path, kogl_marker: 평가 메타데이터.
- initial_assessment: 앞 증거 추출 단계의 JSON. 이전 이진판정이 아니라 시각 사실·필수 정보 목록이다.

[공식 판정 기준]
{official}

[대조 절차]
1. initial_assessment.required_information 각각에 대응하는 실제 표현을 alt_text, context_text, 사람이 읽는 link_dest에서 찾는다.
2. page_title·heading_path·URL·경로는 이미지 바로 옆에서 같은 정보를 읽어 주지 않는 한 대체 표현으로 세지 않는다.
3. 필수 정보가 하나라도 명확히 빠졌거나 이미지와 다른 내용이면 부적절로 판정한다.
4. 모든 필수 정보가 실제 표현에 대응하거나 이미지가 고유 정보를 주지 않는 순수 장식이면 적절로 판정한다.

[Rule]
1. 더 자세히 쓸 수 있다는 이유만으로 부적절로 판정하지 않는다. 누락·불일치한 항목을 구체적으로 특정할 수 있어야 한다.
2. 포스터·화면·도표·절차도는 핵심 문구·수치·단계가, 사진은 고유 대상·행동·상태가 전달되는지 확인한다.
3. 아래 JSON 객체 하나만 반환한다. 각 근거는 350자 이내다.

[Output format]
{{"verdict":"부적절","reason":"필수 정보와 실제 전달 표현의 대조 결과","image_evidence":"앞 단계에서 추출된 핵심 시각 사실","context_evidence":"alt와 실제 인접·링크 문구의 대응 결과"}}

[Output]'''


def frozen_coverage_feedback(coverage):
    """Turn a concrete V9 coverage failure into the existing fallback contract."""
    return {
        'action': '피드백 제공',
        'feedback': (
            'alt_text와 실제 context_text·사람이 읽는 link_dest가 앞 시각 증거 단계에서 '
            '추출한 필수 정보를 전달하는지 다시 대조하라. '
            f"대조 근거: {coverage['reason']}")[:350],
        'reason': '독립 증거 추출과 항목별 대조에서 구체적인 누락 또는 불일치가 확인됐다.',
    }


def _reflection_fallback_prompt(official, frozen=False):
    base = _guideline_binary_prompt(official)
    guideline_field = (
        '- current_guidelines: train에서 학습을 마친 뒤 고정한 Reflection 판단 가이드라인.\n'
        if frozen else '')
    review_sources = 'reflection_feedback과 current_guidelines의' if frozen else 'reflection_feedback의'
    return base.replace(
        '- kogl_marker: 수집 페이지의 공공누리 표시 여부.\n',
        '- kogl_marker: 수집 페이지의 공공누리 표시 여부.\n'
        '- initial_assessment: 같은 입력에 대한 최초 이진판정 JSON과 그 근거.\n'
        '- reflection_feedback: 별도 Reflection 모듈이 제공한 일반화된 추가 판단 기준.\n'
        + guideline_field,
    ).replace(
        '[판정 절차]\n',
        f'[판정 절차]\n0. initial_assessment를 기본 결론으로 두고 {review_sources} 재검토 조건을 원본 이미지·메타데이터와 다시 대조한다. reflection_feedback은 반대 결론을 지시하는 문장이 아니다. 피드백 조건이 원본에서 명확히 확인되고 최초 근거를 뒤집을 때만 결론을 변경하며, 그렇지 않으면 최초 결론을 유지한다.\n',
    )


def _reflection_document_prompt(official):
    return f'''[Instruction]
기존 Reflection 재검토 체크리스트를 새 피드백과 통합한 전체 Markdown 문서로 수정하라. 이 문서는 정답을 지시하는 규칙집이 아니라, 다음 사례에서 놓치기 쉬운 근거를 다시 대조하게 하는 짧은 체크리스트다.

[Input]
- current_guidelines: 기존 Markdown 문서 전체.
- new_feedback: 이번 문제에서 생성한 일반화된 피드백 JSON.

[공식 판정 기준]
{official}

[Rule]
1. 공식 기준은 충돌 검사용으로만 사용하며, 공식 가이드라인 원문이나 라벨 표를 복사하지 않는다.
2. 문서에는 new_feedback에서 뒷받침되는 일반화된 재검토 체크리스트만 둔다. 최초 판정을 지지하는 피드백을 새로운 결함·예외나 더 엄격한 충족 조건으로 바꾸지 않는다.
3. 각 항목은 판정 결론이나 라벨을 지시하지 않고, 반복적인 서론 없이 비교할 입력·시각 근거부터 바로 쓴다.
4. 기존 항목과 새 피드백이 겹치면 하나로 병합한다. 충돌하면 적용 조건을 더 좁혀 기존 항목을 교체하며, 서로 충돌하는 두 항목을 동시에 남기지 않는다.
5. 사례의 record ID·실제 라벨·고유명사·URL·날짜·필드의 고유 문자열을 남기지 않는다.
6. 제목과 `## 누적 재검토 체크리스트`만 유지하고 전체 3,000자 이내로 쓴다. 새 기준을 추가할 공간이 부족하면 중복 항목부터 통합한다.
7. 아래 JSON 객체 하나만 반환한다.

[Output format]
{{"markdown":"# Reflection 판단 가이드라인\\n\\n## 누적 재검토 체크리스트\\n\\n- 비교할 입력과 시각 근거를 다시 확인한다."}}

[Output]'''


def _reflection_document_compact_prompt(official):
    return f'''[Instruction]
5,000자를 초과한 기존 Reflection 재검토 체크리스트의 중복을 정리하면서 새 피드백을 통합한 전체 Markdown 문서를 반환하라. 문서를 짧게 뭉개는 요약이 아니라, 동일한 판단 기준의 반복만 제거하는 상세 가이드라인 편집 작업이다.

[Input]
- current_guidelines: 기존 Markdown 문서 전체.
- new_feedback: 이번 문제에서 생성한 일반화된 피드백 JSON.

[공식 판정 기준]
{official}

[Rule]
1. 공식 기준은 충돌 검사용으로만 사용하며, 공식 가이드라인 원문이나 라벨 표를 복사하지 않는다.
2. 역할·적용 조건·비교할 입력·시각 근거·판정 경계·예외가 모두 같은 항목만 중복으로 본다. 표현이 비슷해도 적용 조건이나 예외가 다르면 별도 기준으로 보존한다.
3. 정확히 중복되거나 의미가 거의 같은 항목은 하나로 병합한다. 병합한 항목에는 기존 항목이 가진 구체적인 적용 조건, 예외와 경계 사례를 빠뜨리지 않는다.
4. 특히 빈 alt, 링크 이미지, 텍스트 포함 이미지, 정보성 이미지, 장식 이미지의 조건과 예외를 서로 섞거나 일반 문장 하나로 축약하지 않는다.
5. 새 피드백은 기존 기준과 충돌하지 않게 적용 범위를 명시해 통합한다. 충돌하면 더 구체적인 조건으로 교체하고 충돌하는 두 항목을 함께 남기지 않는다.
6. 최초 판정을 지지하는 피드백을 새로운 결함·예외나 더 엄격한 충족 조건으로 바꾸지 않는다. 판정 결론이나 라벨을 직접 지시하지 않는다.
7. record ID·실제 라벨·고유명사·URL·날짜·필드의 고유 문자열을 남기지 않는다.
8. 제목과 `## 누적 재검토 체크리스트` 구조를 유지한다. 4,500~5,000자를 목표로 하되, 상세 조건과 예외 보존을 글자 수보다 우선한다.
9. 아래 JSON 객체 하나만 반환한다.

[Output format]
{{"markdown":"# Reflection 판단 가이드라인\\n\\n## 누적 재검토 체크리스트\\n\\n- 적용 조건과 비교할 근거를 구체적으로 확인한다."}}

[Output]'''


def _type_reflection_feedback_prompt(official, frozen=False):
    if frozen:
        instruction = ('원본 입력과 최초 오류 유형 후보를 공식 가이드라인 및 학습 완료된 오류 유형 후보 '
                       'Reflection 가이드라인과 대조하고 빠진 후보를 추가할 필요가 있는지 판단하라. '
                       '누락 가능성을 다시 검토할 구체적인 기준이 있으면 '
                       '"피드백 제공", 없으면 "통과"를 선택하라.')
        actual_input = ''
        rules = '''1. 최초 후보가 맞거나 틀렸다고 미리 전제하지 않는다.
2. 후보 Reflection은 후보를 거르는 단계가 아니다. 잘못 포함된 후보를 제거하거나 교체할 필요성은 판단하지 않는다.
3. current_guidelines의 구체적인 항목이 현재 사례에 적용되고, 최초 후보가 관련 유형을 빠뜨렸을 가능성이 명확할 때만 "피드백 제공"을 선택한다.
4. 추가할 후보가 없거나 적용할 누적 기준이 없으면 "통과"를 선택한다.
5. "피드백 제공"이면 최종 답을 지시하지 않고 빠진 유형을 발견하기 위한 한 가지 비교 조건만 "~인지 다시 확인하라" 형식으로 쓴다. "통과"이면 feedback은 빈 문자열로 쓴다. feedback과 reason은 각각 350자 이내다.'''
        output = '''{"action":"피드백 제공","feedback":"누락된 시각 정보가 특정 오류 유형의 정의를 충족하는지 다시 확인하라.","reason":"현재 사례에 적용되는 구체적인 누적 기준이 확인된다."}
또는
{"action":"통과","feedback":"","reason":"후보를 다시 검토할 구체적인 누적 기준이 확인되지 않는다."}'''
    else:
        instruction = ('학습 데이터의 최초 오류 유형 후보를 검토하고, 실제 유형이 후보에서 누락된 원인을 '
                       '찾아 빠진 후보를 추가할 필요가 있는지 재검토 모델이 원본 입력에서 다시 확인하도록 '
                       '일반화된 피드백 1~2문장으로 작성하라.')
        actual_input = '- actual_label: 학습 데이터의 실제 부적절 유형 라벨.\n'
        rules = '''1. actual_label의 공식 정의와 경계 규칙을 원본 입력에 대조하고, 최초 후보가 놓친 한 가지 비교 조건을 찾는다.
2. current_guidelines와 일관되게 작성하며 충돌하면 적용 조건이나 예외를 좁힌다.
3. 후보를 삭제하거나 교체하도록 유도하지 않고 누락 후보를 추가하기 위한 확인 기준만 쓴다.
4. actual_label이나 최종 후보를 직접 말하지 않고 "~인지 다시 확인하라" 또는 "~를 다시 고려하라" 형식으로 쓴다.
5. record ID·고유명사·URL·날짜를 암기하지 않고 다른 문제에도 적용할 수 있는 기준만 350자 이내로 쓴다.'''
        output = '{"feedback":"이미지와 alt_text의 불일치가 특정 오류 유형의 정의를 충족하는지 다시 확인하라."}'
    return f'''[Instruction]
{instruction}

[Input]
- image: 상자나 주석 없는 원본 이미지.
- alt_text, page_title, context_text, in_link, link_dest, heading_path, kogl_marker: 후보 생성과 같은 메타데이터.
- binary_assessment: 앞 단계의 최종 부적절 이진판정 JSON.
- initial_candidate_assessment: 최초 복수 오류 유형 후보 JSON.
{actual_input}- current_guidelines: 누적된 오류 유형 후보 Reflection 가이드라인.

[공식 판정 기준]
{official}

[Rule]
{rules}
6. 아래 JSON 객체 하나만 반환한다.

[Output format]
{output}

[Output]'''


def _type_reflection_fallback_prompt(official):
    return f'''[Instruction]
원본 이미지와 메타데이터를 공식 가이드라인에 대조하고, 최초 오류 유형 후보와 Reflection 피드백을 참고하여 빠진 부적절 유형 후보를 추가하라.

[Input]
- image: 상자나 주석 없는 원본 이미지.
- alt_text, page_title, context_text, in_link, link_dest, heading_path, kogl_marker: 평가 입력 메타데이터.
- binary_assessment: 앞 단계의 최종 부적절 이진판정 JSON.
- initial_candidate_assessment: 최초 복수 오류 유형 후보 JSON.
- reflection_feedback: 별도 Reflection 모듈이 제공한 추가 확인 기준.

[공식 판정 기준]
{official}

[판정 절차]
1. 6개 부적절 유형을 각각 독립적으로 원본 이미지·메타데이터와 대조한다.
2. reflection_feedback의 조건을 확인하되, 피드백을 정답이나 특정 라벨 지시로 해석하지 않는다.
3. initial_candidate_assessment의 후보를 모두 유지한다.
4. 관찰 근거가 있거나 명확히 배제할 수 없는 새 유형을 matched_types에 추가한다.

[Rule]
1. matched_types는 1개 이상 6개 이하이며 같은 라벨을 중복하지 않는다.
2. 최초 후보는 이미 확보한 후보이므로 후보를 삭제하거나 교체하지 않는다.
3. 아래 형식의 JSON 객체 하나만 반환한다. 각 근거는 한국어 350자 이내다.

[Output format]
{{"matched_types":[{{"label":"불충분형","reason":"후보 근거"}}],"reason":"후보 집합의 종합 근거","image_evidence":"이미지에서 확인한 내용","context_evidence":"메타데이터에서 확인한 내용"}}

[Output]'''


def _type_reflection_document_prompt(official, compact=False):
    task = ('5,000자를 초과한 기존 체크리스트에서 같은 판단 기준의 반복만 제거하고 새 피드백을 통합하라.'
            if compact else '기존 체크리스트에 새 피드백을 충돌 없이 통합하라.')
    length = ('4,500~5,000자를 목표로 하되 상세 조건과 예외 보존을 우선한다.'
              if compact else '전체 3,000자 이내로 쓰고 중복 항목은 하나로 병합한다.')
    return f'''[Instruction]
{task} 이 문서는 다음 사례에서 실제 부적절 유형을 후보 집합에서 빠뜨리지 않기 위한 재검토 체크리스트다.

[Input]
- current_guidelines: 기존 Markdown 문서 전체.
- new_feedback: 이번 문제에서 생성한 일반화된 피드백 JSON.

[공식 판정 기준]
{official}

[Rule]
1. 공식 기준은 충돌 검사용으로만 사용하며 원문이나 라벨 표를 복사하지 않는다.
2. 역할·적용 조건·비교할 입력·시각 근거·경계와 예외가 같은 항목만 병합한다.
3. 새 피드백이 기존 기준과 충돌하면 더 구체적인 적용 조건으로 교체한다.
4. 실제 라벨·record ID·고유명사·URL·날짜를 남기지 않는다.
5. 제목 `# 오류 유형 후보 Reflection 가이드라인`과 `## 누적 재검토 체크리스트` 구조를 유지한다.
6. {length}
7. 아래 JSON 객체 하나만 반환한다.

[Output format]
{{"markdown":"# 오류 유형 후보 Reflection 가이드라인\\n\\n## 누적 재검토 체크리스트\\n\\n- 후보에서 놓치기 쉬운 근거를 다시 확인한다."}}

[Output]'''


def _final_reflection_feedback_prompt(official, frozen=False):
    if frozen:
        instruction = ('원본 입력과 최초 최종 오류 유형 선택을 공식 가이드라인 및 학습 완료된 최종 선택 '
                       'Reflection 가이드라인과 대조하라. 다시 선택할 구체적인 기준이 있으면 '
                       '"피드백 제공", 없으면 "통과"를 선택하라.')
        actual_input = ''
        rules = '''1. 최초 선택이 맞거나 틀렸다고 미리 전제하지 않는다.
2. current_guidelines의 구체적인 후보 경계가 현재 사례에 적용되고, 최초 선택이 관련 이미지·alt·문맥 근거를 놓쳤을 가능성이 명확할 때만 "피드백 제공"을 선택한다.
3. 적용할 누적 기준이 없거나 최초 선택이 공식 기준 및 누적 기준과 일치하면 "통과"를 선택한다.
4. "피드백 제공"이면 특정 라벨이나 최종 답을 직접 지시하지 않고 후보 사이에서 다시 비교할 한 가지 조건만 쓴다.
5. "통과"이면 feedback은 빈 문자열로 쓴다. feedback과 reason은 각각 350자 이내다.'''
        output = '''{"action":"피드백 제공","feedback":"두 후보의 정의를 가르는 이미지와 alt_text 근거를 다시 확인하라.","reason":"현재 사례에 적용되는 구체적인 후보 경계가 확인된다."}
또는
{"action":"통과","feedback":"","reason":"최초 선택을 다시 검토할 구체적인 누적 기준이 확인되지 않는다."}'''
    else:
        instruction = ('학습 데이터의 최초 최종 오류 유형 선택을 검토하고, 같은 후보 집합 안에서 잘못 '
                       '선택한 원인을 재선택 모델이 다시 확인하도록 일반화된 피드백 1~2문장으로 작성하라.')
        actual_input = '- actual_label: 학습 데이터의 실제 부적절 유형 라벨.\n'
        rules = '''1. actual_label과 최초 선택이 다른 이유를 공식 정의와 경계 규칙에서 찾고, 같은 후보 집합 안에서 다시 비교할 한 가지 조건을 찾는다.
2. current_guidelines와 일관되게 작성하며 충돌하면 적용 조건이나 예외를 좁힌다.
3. actual_label이나 최종 답을 직접 말하지 않고 "~인지 다시 확인하라" 또는 "~를 다시 고려하라" 형식으로 쓴다.
4. record ID·고유명사·URL·날짜를 암기하지 않고 다른 문제에도 적용할 수 있는 기준만 350자 이내로 쓴다.'''
        output = '{"feedback":"후보들의 정의를 가르는 이미지와 alt_text의 핵심 차이를 다시 확인하라."}'
    return f'''[Instruction]
{instruction}

[Input]
- image: 상자나 주석 없는 원본 이미지.
- alt_text, page_title, context_text, in_link, link_dest, heading_path, kogl_marker: 최초 최종 선택과 같은 메타데이터.
- binary_assessment: 앞 단계의 최종 부적절 이진판정 JSON.
- candidate_assessment: 확정되어 변경할 수 없는 복수 오류 유형 후보 JSON.
- initial_final_assessment: 후보 안에서 고른 최초 최종 오류 유형 JSON.
{actual_input}- current_guidelines: 누적된 최종 오류 유형 선택 Reflection 가이드라인.

[공식 판정 기준]
{official}

[Rule]
{rules}
6. 아래 JSON 객체 하나만 반환한다.

[Output format]
{output}

[Output]'''


def _final_reflection_fallback_prompt(official):
    return f'''[Instruction]
최초 선택은 입력받지 않는다. 원본 이미지·alt_text·메타데이터와 확정 후보 집합을 독립적으로 대조하되, Reflection 피드백과 누적 최종 선택 가이드라인에서 현재 사례에 적용되는 경계를 다시 확인하여 최종 오류 유형 하나를 선택하라.

[Input]
- image: 상자나 주석 없는 원본 이미지.
- alt_text, page_title, context_text, in_link, link_dest, heading_path, kogl_marker: 평가 입력 메타데이터.
- binary_assessment: 앞 단계의 최종 부적절 이진판정 JSON.
- candidate_assessment: 확정되어 변경할 수 없는 복수 오류 유형 후보 JSON.
- reflection_feedback: 별도 Reflection 모듈이 현재 사례에서 다시 대조하라고 선별한 판단 경계.
- current_guidelines: train에서 학습을 마친 뒤 고정한 최종 오류 유형 선택 가이드라인.

[공식 판정 기준]
{official}

[판정 절차]
0. reflection_feedback과 current_guidelines는 정답 지시가 아니다. 원본 입력에서 그 조건이 실제로 확인되는 경우에만 적용한다.
1. 이미지의 주된 대상·역할·핵심 정보와 alt_text가 실제로 전달하는 내용을 각각 한 문장으로 확인한다. 이미지에 없는 기간·대상·방법을 추측하지 않는다.
2. candidate_assessment의 각 후보가 공식 정의의 전체 조건을 충족하는지 이미지·alt_text·문맥과 독립적으로 대조한다.
3. 다음 경계를 정의의 필수 조건으로 쓴다.
   - `무관형` 온라인 alt의 주된 대상이 이미지와 다른 경우다. 같은 대상·사건·기관을 설명하면서 세부 정보를 빠뜨리거나 일부 값이 틀린 경우는 `불충분형`을 우선 검토한다.
   - `파일명형` 은 확장자·경로·기계적 스템이 실제 파일명으로 읽히는 경우다. 자연스러운 한국어 제목에 괄호·번호·`_숫자`·`(배포용)`이 붙은 것만으로 파일명형이 되지 않는다.
   - `무의미형` 은 대상·목적·동작을 알 수 없는 상투어·기호다. `미리보기`·`더보기`·작성 지시문처럼 방향이나 의도는 있지만 대상과 핵심 정보가 빠진 표현은 `불충분형`을 우선 검토한다.
   - `중복형` 은 alt가 실제 인접 캡션·제목·같은 링크 문구를 실질적으로 반복해 정보 이득이 0인 경우다. 완전히 같은 사진 캡션·영상 제목이면 이미지에 부수적 문구가 보인다는 이유로 임의의 누락 정보를 만들지 않는다.
   - `불충분형` 은 alt의 방향이 이미지와 같지만, 인접 문맥이 대신하지 못하는 필수 목적지·동작·장면·절차·수치·기간·대상을 빠뜨린 경우다.
4. 각 후보의 정의가 독립적으로 성립한 뒤에만 우선순위를 적용한다. 후보에 있다는 사실이나 단일 단어·접미사만으로 우선순위 유형을 확정하지 않는다.
5. candidate_assessment에 이미 존재하는 라벨 하나만 선택한다.

[Rule]
1. 후보를 추가하거나 제거하지 않는다.
2. reason에 선택한 후보의 필수 조건과 가장 헷갈린 다른 후보를 배제한 실제 근거를 쓴다.
3. 아래 JSON 객체 하나만 반환한다. 각 근거는 한국어 350자 이내다.

[Output format]
{{"label":"불충분형","reason":"최종 선택 근거","image_evidence":"이미지에서 확인한 내용","context_evidence":"메타데이터에서 확인한 내용"}}

[Output]'''


def _final_reflection_document_prompt(official, compact=False):
    task = ('5,000자를 초과한 기존 체크리스트에서 같은 후보 경계의 반복만 제거하고 새 피드백을 통합하라.'
            if compact else '기존 체크리스트에 새 피드백을 충돌 없이 통합하라.')
    length = ('4,500~5,000자를 목표로 하되 상세 조건과 예외 보존을 우선한다.'
              if compact else '전체 3,000자 이내로 쓰고 중복 항목은 하나로 병합한다.')
    return f'''[Instruction]
{task} 이 문서는 다음 사례에서 확정 후보 집합 안의 최종 오류 유형을 정확히 선택하기 위한 재검토 체크리스트다.

[Input]
- current_guidelines: 기존 Markdown 문서 전체.
- new_feedback: 이번 문제에서 생성한 일반화된 피드백 JSON.

[공식 판정 기준]
{official}

[Rule]
1. 공식 기준은 충돌 검사용으로만 사용하며 원문이나 라벨 표를 복사하지 않는다.
2. 후보 간 경계·적용 조건·비교할 입력·시각 근거·예외가 같은 항목만 병합한다.
3. 새 피드백이 기존 기준과 충돌하면 더 구체적인 적용 조건으로 교체한다.
4. 실제 라벨·record ID·고유명사·URL·날짜를 남기지 않는다.
5. 제목 `# 최종 오류 유형 선택 Reflection 가이드라인`과 `## 누적 재검토 체크리스트` 구조를 유지한다.
6. {length}
7. 아래 JSON 객체 하나만 반환한다.

[Output format]
{{"markdown":"# 최종 오류 유형 선택 Reflection 가이드라인\\n\\n## 누적 재검토 체크리스트\\n\\n- 후보 사이에서 놓치기 쉬운 경계 근거를 다시 확인한다."}}

[Output]'''


def default_config(max_tokens=4096, context_size=16384, *, binary_prompt='role_split', recheck_appropriate=False,
                   empty_alt_prompt=False, review_policy=None, compare_review_role=False, role_observation=False,
                   model_variant=DEFAULT_MODEL_VARIANT, label_reflection=False,
                   reflection_mode=REFLECTION_TRAIN, reflection_guideline_source_run=None,
                   reflection_guideline_sha256=None, binary_only=False, candidate_only=False,
                   type_reflection=False,
                   type_reflection_guideline_sha256=None, final_reflection=False,
                   final_reflection_guideline_sha256=None, decorative_gate=False):
    if model_variant not in MODEL_VARIANTS:
        raise ValueError('알 수 없는 Gemma 모델: ' + str(model_variant))
    model = MODEL_VARIANTS[model_variant]
    guideline_direct = binary_prompt == 'guideline_direct'
    if label_reflection and not guideline_direct:
        raise ValueError('라벨 Reflection은 가이드라인 직접 판정에서만 사용합니다.')
    if label_reflection and recheck_appropriate:
        raise ValueError('라벨 Reflection과 기존 적절 재검토를 함께 사용할 수 없습니다.')
    if decorative_gate and (not label_reflection or binary_only or candidate_only):
        raise ValueError('장식오용형 전용 게이트는 전체 Train/Frozen Reflection 실행에서만 사용합니다.')
    if reflection_mode not in (REFLECTION_TRAIN, REFLECTION_FROZEN):
        raise ValueError('알 수 없는 Reflection 실행 모드입니다.')
    if reflection_mode == REFLECTION_FROZEN:
        if (not label_reflection or not isinstance(reflection_guideline_source_run, str)
                or not reflection_guideline_source_run.strip()
                or not isinstance(reflection_guideline_sha256, str)
                or not re.fullmatch(r'[0-9a-f]{64}', reflection_guideline_sha256)):
            raise ValueError('Frozen MD 실행에는 학습 실행 ID와 가이드라인 SHA-256이 필요합니다.')
    elif reflection_guideline_source_run is not None or reflection_guideline_sha256 is not None:
        raise ValueError('Train Reflection에는 Frozen MD 출처를 지정하지 않습니다.')
    if type_reflection and (not label_reflection or binary_only):
        raise ValueError('오류 유형 후보 Reflection은 전체 Train/Frozen Reflection 실행에서만 사용합니다.')
    if type_reflection and reflection_mode == REFLECTION_FROZEN:
        if (not isinstance(type_reflection_guideline_sha256, str)
                or not re.fullmatch(r'[0-9a-f]{64}', type_reflection_guideline_sha256)):
            raise ValueError('Frozen 후보 Reflection에는 후보 가이드라인 SHA-256이 필요합니다.')
    elif type_reflection_guideline_sha256 is not None:
        raise ValueError('후보 Reflection 가이드라인 SHA-256 설정이 잘못되었습니다.')
    if candidate_only and (binary_only or reflection_mode != REFLECTION_FROZEN
                           or not label_reflection or not type_reflection or final_reflection):
        raise ValueError('후보풀 전용 실행은 Frozen MD의 이진·후보 Reflection 조합에서만 사용합니다.')
    if final_reflection and (not type_reflection or not label_reflection or binary_only):
        raise ValueError('최종 선택 Reflection은 전체 Train/Frozen Reflection 실행에서만 사용합니다.')
    if final_reflection and reflection_mode == REFLECTION_FROZEN:
        if (not isinstance(final_reflection_guideline_sha256, str)
                or not re.fullmatch(r'[0-9a-f]{64}', final_reflection_guideline_sha256)):
            raise ValueError('Frozen 최종 선택 Reflection에는 최종 가이드라인 SHA-256이 필요합니다.')
    elif final_reflection_guideline_sha256 is not None:
        raise ValueError('최종 선택 Reflection 가이드라인 SHA-256 설정이 잘못되었습니다.')
    if guideline_direct and (empty_alt_prompt or compare_review_role or role_observation or review_policy is not None):
        raise ValueError('가이드라인 직접 판정은 역할·빈 alt·역할 비교 분기를 함께 사용하지 않습니다.')
    if empty_alt_prompt and binary_prompt == 'role_checks':
        raise ValueError('빈 alt 전용 판정은 v7/v8/v11에서 사용하세요. v9는 이전 출력 계약을 유지합니다.')
    options = {'basic': ('alt_binary_basic.md', 'v7-two-stage-basic'),
               'criteria': ('alt_binary.md', 'v8-two-stage-role-criteria'),
               'role_checks': ('alt_binary_roles.md', 'v9-two-stage-role-checks'),
               'role_split': ('alt_binary_selected_role.md', 'v11-short-role-split'),
               'evidence_split': ('alt_binary_evidence.md', 'v12-evidence-split'),
               'guideline_direct': (None, 'v16-guideline-v1.1-sections-1-3-5')}
    if binary_prompt not in options:
        raise ValueError('알 수 없는 1차 프롬프트: ' + str(binary_prompt))
    binary_file, version = options[binary_prompt]
    evidence_stage = binary_prompt == 'evidence_split'
    if review_policy is not None and not evidence_stage:
        raise ValueError('재검토 정책은 evidence_split에서만 선택할 수 있습니다.')
    policy = review_policy or 'evidence_uncertain_v1'
    if evidence_stage and policy not in ('evidence_uncertain_v1', 'appropriate_v1'):
        raise ValueError('알 수 없는 evidence 재검토 정책: ' + str(policy))
    if compare_review_role and not (evidence_stage and recheck_appropriate and empty_alt_prompt
                                    and policy == 'appropriate_v1'):
        raise ValueError('역할 포함 재검토 비교는 evidence_split의 appropriate_v1·재검토·빈 alt 전용 조합에서만 사용합니다.')
    role_observation = bool(role_observation or compare_review_role)
    if role_observation and not evidence_stage:
        raise ValueError('역할 관찰은 evidence_split에서만 선택할 수 있습니다.')
    if role_observation:
        binary_file = 'alt_binary_role_observation.md'
    def read_prompt(stage, name):
        prompt = (ROOT / 'prompts' / name).read_text(encoding='utf-8')
        if evidence_stage and stage == 'types':
            return prompt.replace('[설명]\n', '[설명]\nvisual_evidence는 앞 시각 근거이며 오류 가능성이 있어 원본과 다시 비교한다.\n\n', 1)
        return prompt
    official = _guideline_sections_1_to_3_and_5() if guideline_direct else None
    independent_types = bool(label_reflection)
    prompts = {
        'classify': _guideline_binary_prompt(official) if guideline_direct else read_prompt('classify', binary_file),
        'types': _guideline_types_prompt(official, independent_types) if guideline_direct else read_prompt('types', 'alt_types.md'),
    }
    if guideline_direct:
        prompts['final'] = _guideline_final_prompt(official, independent_types)
        prompts['empty_alt_link_visual_review'] = _empty_alt_link_visual_review_prompt(official)
    if label_reflection:
        prompts.update({
            'reflection_feedback': (_frozen_reflection_feedback_prompt(official)
                                    if reflection_mode == REFLECTION_FROZEN
                                    else _reflection_feedback_prompt(official)),
            'reflection_fallback': _reflection_fallback_prompt(
                official, reflection_mode == REFLECTION_FROZEN),
        })
        if reflection_mode == REFLECTION_TRAIN:
            prompts['reflection_document'] = _reflection_document_prompt(official)
            prompts['reflection_document_compact'] = _reflection_document_compact_prompt(official)
        else:
            prompts['reflection_visual_evidence'] = _frozen_visual_evidence_prompt(official)
            prompts['reflection_coverage_audit'] = _frozen_coverage_audit_prompt(official)
    if type_reflection:
        prompts.update({
            'type_reflection_feedback': _type_reflection_feedback_prompt(
                official, reflection_mode == REFLECTION_FROZEN),
            'type_reflection_fallback': _type_reflection_fallback_prompt(official),
        })
        if reflection_mode == REFLECTION_TRAIN:
            prompts['type_reflection_document'] = _type_reflection_document_prompt(official)
            prompts['type_reflection_document_compact'] = _type_reflection_document_prompt(
                official, compact=True)
    if final_reflection:
        prompts.update({
            'final_reflection_feedback': _final_reflection_feedback_prompt(
                official, reflection_mode == REFLECTION_FROZEN),
            'final_reflection_fallback': _final_reflection_fallback_prompt(official),
        })
        if reflection_mode == REFLECTION_TRAIN:
            prompts['final_reflection_document'] = _final_reflection_document_prompt(official)
            prompts['final_reflection_document_compact'] = _final_reflection_document_prompt(
                official, compact=True)
    if decorative_gate:
        prompts.update(decorative_router=DECORATIVE_ROUTER_PROMPT,
                       decorative_observer=DECORATIVE_OBSERVER_PROMPT)
    if evidence_stage:
        prompts['role'] = read_prompt('role', 'image_visual_evidence_guideline.md' if role_observation
                                      else 'image_visual_evidence.md')
    elif binary_prompt == 'role_split':
        prompts['role'] = read_prompt('role', 'image_role.md')
    if recheck_appropriate:
        prompts['recheck'] = (_guideline_binary_prompt(official, reflection=True) if guideline_direct else
                              read_prompt('recheck', 'alt_appropriate_review.md'
                                          if evidence_stage and policy == 'appropriate_v1'
                                          else 'alt_appropriate_review_evidence.md' if evidence_stage
                                          else 'alt_appropriate_review.md'))
    if empty_alt_prompt:
        prompts['empty_alt'] = read_prompt('empty_alt', 'alt_binary_empty_role_observation.md' if role_observation
                                           else 'alt_binary_empty_evidence.md' if evidence_stage
                                           else 'alt_binary_empty.md')
    return {'model_variant': model_variant, 'model_label': model['label'],
            'model_path': model['model_path'],
            'mmproj_path': model['mmproj_path'],
            'server_path': str(ROOT.parent / 'model/runtime/llama-server'),
            'max_tokens': int(max_tokens), 'context_size': int(context_size),
            'request_timeout': 240, 'startup_timeout': 180,
            'prompt_version': version + (('+appropriate-review-v2' if evidence_stage and policy == 'appropriate_v1'
                                           else '+appropriate-review-v1') if recheck_appropriate else '')
                              + ('+role-multi-description-v1' if role_observation else '')
                              + ('+review-role-comparison-v1' if compare_review_role else '')
                              + (('+empty-alt-v3-context-coverage-no-review' if role_observation
                                  else '+empty-alt-v2-no-review') if empty_alt_prompt else '')
                              + ('+link-structure-gate-v2+visual-exception-v1+candidate-recall-v2' if guideline_direct else '')
                              + (('+frozen-reflection-v3-feedback-gate-md-fallback' if reflection_mode == REFLECTION_FROZEN
                                  else '+label-reflection-v6-errors-only') if label_reflection else '')
                              + ('+type-candidate-recall-v2' if type_reflection else '')
                              + ('+final-selection-guided-v4' if final_reflection else '')
                              + ('+binary-only-v1' if binary_only else '')
                              + ('+candidate-only-v1' if candidate_only else '')
                              + ('+decorative-structured-gate-v2' if decorative_gate else '')
                              + '+image-metadata-v1',
            'input_mode': 'image_metadata',
            'binary_only': bool(binary_only),
            'candidate_only': bool(candidate_only),
            **({'decorative_gate': True} if decorative_gate else {}),
            **({'binary_gate': LINK_STRUCTURE_GATE} if guideline_direct else {}),
            **({'recheck_appropriate': True} if recheck_appropriate else {}),
            **({'label_reflection': True,
                'reflection_session': 'label_feedback_v1', 'session_slots': 2,
                'execution_mode': 'label_reflection_sequential_v1',
                'reflection_mode': reflection_mode,
                'reflection_gate_policy': ('deterministic_final_v1' if reflection_mode == REFLECTION_FROZEN
                                           else 'mismatch_reflection_v1'),
                'error_classification_input': 'image_metadata_candidates_v1',
                **({'reflection_document_compact_threshold': REFLECTION_DOCUMENT_COMPACT_THRESHOLD,
                    'reflection_document_max_tokens': REFLECTION_DOCUMENT_MAX_TOKENS,
                    'reflection_training_route': REFLECTION_ERRORS_ONLY}
                   if reflection_mode == REFLECTION_TRAIN else {}),
                **({'reflection_guideline_source_run': reflection_guideline_source_run,
                    'reflection_guideline_sha256': reflection_guideline_sha256,
                    'frozen_feedback_policy': FROZEN_FEEDBACK_GATE}
                   if reflection_mode == REFLECTION_FROZEN else {})} if label_reflection else {}),
            **({'type_reflection': True,
                **({'type_reflection_guideline_sha256': type_reflection_guideline_sha256}
                   if reflection_mode == REFLECTION_FROZEN else {})}
               if type_reflection else {}),
            **({'final_reflection': True,
                **({'final_reflection_guideline_sha256': final_reflection_guideline_sha256}
                   if reflection_mode == REFLECTION_FROZEN else {})}
               if final_reflection else {}),
            **({'empty_alt_prompt': True, 'skip_empty_alt_review': True} if empty_alt_prompt else {}),
            **({'binary_contract': 'role_verdict_v1'} if binary_prompt == 'role_checks' else {}),
            **({'binary_contract': 'evidence_verdict', 'evidence_stage': True,
                'role_stage': True, 'review_policy': policy} if evidence_stage else {}),
            **({'compare_review_role': True} if compare_review_role else {}),
            **({'role_observation_mode': 'multilabel_description_v1'} if role_observation else {}),
            **({'role_stage': True, 'role_criteria': dict(ROLE_CRITERIA)} if binary_prompt == 'role_split' else {}),
            **({'guideline_mode': 'sections_1_3_5_v1_1',
                'guideline_source': 'data/docs/라벨링-가이드라인-v1.1.md',
                'guideline_sha256': hashlib.sha256(official.encode()).hexdigest(),
                'type_candidate_policy': TYPE_CANDIDATE_RECALL,
                **({'reflection_session': 'separate_slot_v1', 'session_slots': 2}
                   if recheck_appropriate else
                   {} if label_reflection else
                   {'execution_mode': 'record_parallel_v1', 'parallel_slots': 2})}
               if guideline_direct else {}),
            'classification_mode': ('guideline_verdict_candidates_final_v1' if guideline_direct
                                    else 'two_stage_verdict_types_v1'),
            'label_priority': list(ERROR_PRIORITY),
            'prompts': prompts}


def validate_config(config):
    if type(config.get('binary_only', False)) is not bool:
        raise ValueError('이진분류 전용 설정은 true 또는 false여야 합니다.')
    if type(config.get('candidate_only', False)) is not bool:
        raise ValueError('후보풀 전용 설정은 true 또는 false여야 합니다.')
    decorative_gate = config.get('decorative_gate', False)
    if type(decorative_gate) is not bool:
        raise ValueError('장식오용형 전용 게이트 설정은 true 또는 false여야 합니다.')
    if decorative_gate and (not config.get('label_reflection')
                            or config.get('binary_only') or config.get('candidate_only')):
        raise ValueError('장식오용형 전용 게이트는 전체 Train/Frozen Reflection 실행에서만 사용합니다.')
    parallel_slots = config.get('parallel_slots')
    execution_mode = config.get('execution_mode')
    valid_execution = ((execution_mode, parallel_slots) in ((None, None), ('record_parallel_v1', 2))
                       or (execution_mode, parallel_slots) == ('label_reflection_sequential_v1', None))
    if (not valid_execution
            or (execution_mode and not config.get('guideline_mode'))
            or (execution_mode == 'record_parallel_v1' and config.get('recheck_appropriate'))):
        raise ValueError('저장된 병렬 실행 설정이 잘못되었습니다.')
    if config.get('binary_gate') not in (None, EMPTY_ALT_LINK_GATE, LINK_STRUCTURE_GATE):
        raise ValueError('저장된 이진 구조 게이트가 잘못되었습니다.')
    if config.get('binary_gate') and not config.get('guideline_mode'):
        raise ValueError('링크 구조 게이트는 가이드라인 직접 판정에서만 사용합니다.')
    if config.get('guideline_mode'):
        if (config.get('guideline_mode') not in (
                'sections_1_3_v1', 'sections_1_3_v1_0', 'sections_1_3_5_v1_1')
                or (config.get('recheck_appropriate') and
                    (config.get('reflection_session') != 'separate_slot_v1' or config.get('session_slots') != 2))
                or (not config.get('recheck_appropriate') and not config.get('label_reflection') and
                    (config.get('reflection_session') is not None or config.get('session_slots') is not None))
                or (config.get('label_reflection') and
                    (config.get('reflection_session') != 'label_feedback_v1'
                     or config.get('session_slots') != 2
                     or config.get('execution_mode') != 'label_reflection_sequential_v1'))
                or config.get('role_stage') or config.get('empty_alt_prompt')):
            raise ValueError('저장된 가이드라인 직접 판정 설정이 잘못되었습니다.')
    if config.get('role_observation_mode') and (config.get('role_observation_mode') != 'multilabel_description_v1'
                                                or not config.get('evidence_stage') or not config.get('role_stage')):
        raise ValueError('저장된 역할 관찰 형식이 잘못되었습니다.')
    if config.get('compare_review_role') and not uses_compare_review_role(config):
        raise ValueError('저장된 역할 포함 재검토 비교 조합이 잘못되었습니다.')
    error_input = config.get('error_classification_input')
    if (error_input not in (None, 'image_metadata_candidates_v1')
            or (error_input and not config.get('label_reflection'))):
        raise ValueError('저장된 오류 유형 입력 설정이 잘못되었습니다.')
    type_candidate_policy = config.get('type_candidate_policy')
    if (type_candidate_policy not in (None, TYPE_CANDIDATE_RECALL)
            or (type_candidate_policy and not config.get('guideline_mode'))):
        raise ValueError('저장된 후보 생성 정책이 잘못되었습니다.')
    gate_policy = config.get('reflection_gate_policy')
    if (gate_policy not in (None, 'deterministic_final_v1', 'mismatch_reflection_v1')
            or (gate_policy and not config.get('label_reflection'))):
        raise ValueError('저장된 Reflection 구조 게이트 정책이 잘못되었습니다.')
    reflection_mode = config.get('reflection_mode', REFLECTION_TRAIN)
    if reflection_mode not in (REFLECTION_TRAIN, REFLECTION_FROZEN):
        raise ValueError('저장된 Reflection 실행 모드가 잘못되었습니다.')
    source_run = config.get('reflection_guideline_source_run')
    source_sha = config.get('reflection_guideline_sha256')
    if reflection_mode == REFLECTION_FROZEN:
        if (not config.get('label_reflection') or not isinstance(source_run, str) or not source_run.strip()
                or not isinstance(source_sha, str) or not re.fullmatch(r'[0-9a-f]{64}', source_sha)):
            raise ValueError('저장된 Frozen MD 출처가 잘못되었습니다.')
    elif source_run is not None or source_sha is not None:
        raise ValueError('Train Reflection 설정에 Frozen MD 출처가 있습니다.')
    if config.get('candidate_only') and (
            config.get('binary_only') or reflection_mode != REFLECTION_FROZEN
            or not config.get('label_reflection') or not config.get('type_reflection')
            or config.get('final_reflection')):
        raise ValueError('저장된 후보풀 전용 실행 설정이 잘못되었습니다.')
    training_route = config.get('reflection_training_route')
    if (training_route not in (None, REFLECTION_ERRORS_ONLY)
            or (training_route and (not config.get('label_reflection')
                                    or reflection_mode != REFLECTION_TRAIN))
            or (gate_policy == 'mismatch_reflection_v1'
                and training_route != REFLECTION_ERRORS_ONLY)):
        raise ValueError('저장된 Train Reflection 적용 조건이 잘못되었습니다.')
    frozen_feedback_policy = config.get('frozen_feedback_policy')
    if (frozen_feedback_policy not in (None, FROZEN_FEEDBACK_GATE)
            or (frozen_feedback_policy and (not config.get('label_reflection')
                                             or reflection_mode != REFLECTION_FROZEN))):
        raise ValueError('저장된 Frozen Reflection 피드백 정책이 잘못되었습니다.')
    frozen_evidence_pipeline = config.get('frozen_evidence_pipeline')
    if (frozen_evidence_pipeline not in (None, FROZEN_EVIDENCE_PIPELINE)
            or (frozen_evidence_pipeline and (not config.get('label_reflection')
                                              or reflection_mode != REFLECTION_FROZEN))):
        raise ValueError('저장된 Frozen 시각 증거 검산 설정이 잘못되었습니다.')
    _response_contract('classify', config)
    if config.get('classification_mode') in ('two_stage_verdict_types_v1',
                                              'guideline_verdict_candidates_final_v1'):
        prompts = config.get('prompts', {})
        required = (('classify', 'types', 'final', 'empty_alt_link_visual_review')
                    if config.get('guideline_mode') in (
                        'sections_1_3_v1_0', 'sections_1_3_5_v1_1') else
                    ('classify', 'types', 'final')
                    if config.get('classification_mode') == 'guideline_verdict_candidates_final_v1' else
                    ('classify', 'types'))
        if any(not isinstance(prompts.get(stage), str) or not prompts[stage].strip()
               for stage in required):
            raise ValueError('Gemma 단계별 프롬프트가 저장되지 않았습니다.')
    if config.get('recheck_appropriate') and not config.get('prompts', {}).get('recheck', '').strip():
        raise ValueError('적절 재검토 프롬프트가 저장되지 않았습니다.')
    if config.get('label_reflection'):
        required_reflection = ['reflection_feedback', 'reflection_fallback']
        if reflection_mode == REFLECTION_TRAIN:
            required_reflection.append('reflection_document')
        elif frozen_evidence_pipeline:
            required_reflection += ['reflection_visual_evidence', 'reflection_coverage_audit']
        if (config.get('recheck_appropriate') or any(
                not config.get('prompts', {}).get(stage, '').strip()
                for stage in required_reflection)
                or (reflection_mode == REFLECTION_FROZEN
                    and config.get('prompts', {}).get('reflection_document'))):
            raise ValueError('라벨 Reflection 설정 또는 프롬프트가 잘못되었습니다.')
        if reflection_mode == REFLECTION_TRAIN:
            threshold = config.get('reflection_document_compact_threshold',
                                   REFLECTION_DOCUMENT_COMPACT_THRESHOLD)
            document_tokens = config.get('reflection_document_max_tokens', config['max_tokens'])
            if not isinstance(threshold, int) or threshold < 1000:
                raise ValueError('Reflection MD 정리 분기 글자 수가 잘못되었습니다.')
            if not isinstance(document_tokens, int) or not 128 <= document_tokens <= 8192:
                raise ValueError('Reflection MD 최대 생성 토큰이 잘못되었습니다.')
    if decorative_gate and any(
            not config.get('prompts', {}).get(stage, '').strip()
            for stage in ('decorative_router', 'decorative_observer')):
        raise ValueError('장식오용형 전용 게이트 프롬프트가 저장되지 않았습니다.')
    type_reflection = config.get('type_reflection')
    type_sha = config.get('type_reflection_guideline_sha256')
    if type_reflection:
        required_type_reflection = ['type_reflection_feedback', 'type_reflection_fallback']
        if reflection_mode == REFLECTION_TRAIN:
            required_type_reflection += [
                'type_reflection_document', 'type_reflection_document_compact']
        if (not config.get('label_reflection') or config.get('binary_only')
                or any(not config.get('prompts', {}).get(stage, '').strip()
                       for stage in required_type_reflection)):
            raise ValueError('오류 유형 후보 Reflection 설정 또는 프롬프트가 잘못되었습니다.')
        if reflection_mode == REFLECTION_FROZEN:
            if not isinstance(type_sha, str) or not re.fullmatch(r'[0-9a-f]{64}', type_sha):
                raise ValueError('Frozen 후보 Reflection MD 출처가 잘못되었습니다.')
        elif type_sha is not None:
            raise ValueError('Train 후보 Reflection에 Frozen MD 출처가 있습니다.')
    elif type_sha is not None:
        raise ValueError('후보 Reflection 비활성 설정에 MD 출처가 있습니다.')
    final_reflection = config.get('final_reflection')
    final_sha = config.get('final_reflection_guideline_sha256')
    if final_reflection:
        required_final_reflection = ['final_reflection_feedback', 'final_reflection_fallback']
        if reflection_mode == REFLECTION_TRAIN:
            required_final_reflection += [
                'final_reflection_document', 'final_reflection_document_compact']
        if (not type_reflection or not config.get('label_reflection') or config.get('binary_only')
                or any(not config.get('prompts', {}).get(stage, '').strip()
                       for stage in required_final_reflection)):
            raise ValueError('최종 오류 유형 선택 Reflection 설정 또는 프롬프트가 잘못되었습니다.')
        if reflection_mode == REFLECTION_FROZEN:
            if not isinstance(final_sha, str) or not re.fullmatch(r'[0-9a-f]{64}', final_sha):
                raise ValueError('Frozen 최종 선택 Reflection MD 출처가 잘못되었습니다.')
        elif final_sha is not None:
            raise ValueError('Train 최종 선택 Reflection에 Frozen MD 출처가 있습니다.')
    elif final_sha is not None:
        raise ValueError('최종 선택 Reflection 비활성 설정에 MD 출처가 있습니다.')
    if config.get('empty_alt_prompt') and not config.get('prompts', {}).get('empty_alt', '').strip():
        raise ValueError('빈 alt 전용 판정 프롬프트가 저장되지 않았습니다.')
    if config.get('role_stage'):
        if not config.get('prompts', {}).get('role', '').strip():
            raise ValueError('역할 판정 프롬프트가 없습니다.')
        if not config.get('evidence_stage'):
            for role in IMAGE_ROLES:
                binary_prompt_for_role(config, role)
    if not 128 <= config['max_tokens'] <= 8192:
        raise ValueError('Gemma 최대 생성 토큰: 128~8192')
    required_tokens = max(config['max_tokens'], config.get('reflection_document_max_tokens', 0))
    if config['context_size'] < required_tokens + 4096:
        raise ValueError('Gemma context는 최대 생성 토큰 + 4096 이상이어야 합니다.')
    for key in ('model_path', 'mmproj_path', 'server_path'):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(f'{key}: {config[key]}')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def uses_appropriate_v1_review(config):
    return bool((config or {}).get('evidence_stage')
                and config.get('review_policy', 'evidence_uncertain_v1') == 'appropriate_v1')


def uses_compare_review_role(config):
    return bool((config or {}).get('compare_review_role') and uses_appropriate_v1_review(config)
                and config.get('recheck_appropriate') and config.get('empty_alt_prompt'))


def uses_role_observation(config):
    return bool((config or {}).get('role_observation_mode') == 'multilabel_description_v1')


def _response_contract(stage, config):
    if stage == 'decorative_router' and (config or {}).get('decorative_gate'):
        return 'binary_verdict'
    if stage == 'decorative_observer' and (config or {}).get('decorative_gate'):
        return 'decorative_observation'
    if stage == 'empty_alt_link_visual_review':
        return 'binary_verdict'
    if (stage == 'reflection_visual_evidence'
            and (config or {}).get('frozen_evidence_pipeline') == FROZEN_EVIDENCE_PIPELINE):
        return 'visual_evidence'
    if (stage == 'reflection_coverage_audit'
            and (config or {}).get('frozen_evidence_pipeline') == FROZEN_EVIDENCE_PIPELINE):
        return 'binary_verdict'
    if stage == 'reflection_feedback' and (config or {}).get('label_reflection'):
        return ('reflection_feedback_gate'
                if config.get('frozen_feedback_policy') == FROZEN_FEEDBACK_GATE
                else 'reflection_feedback')
    if stage in ('reflection_document', 'reflection_document_compact') and (config or {}).get('label_reflection'):
        return 'reflection_document'
    if stage == 'reflection_fallback' and (config or {}).get('label_reflection'):
        return 'binary_verdict'
    if stage == 'type_reflection_feedback' and (config or {}).get('type_reflection'):
        return ('reflection_feedback_gate'
                if config.get('reflection_mode') == REFLECTION_FROZEN
                else 'reflection_feedback')
    if stage == 'type_reflection_fallback' and (config or {}).get('type_reflection'):
        return 'error_types'
    if (stage in ('type_reflection_document', 'type_reflection_document_compact')
            and (config or {}).get('type_reflection')):
        return 'reflection_document'
    if stage == 'final_reflection_feedback' and (config or {}).get('final_reflection'):
        return ('reflection_feedback_gate'
                if config.get('reflection_mode') == REFLECTION_FROZEN
                else 'reflection_feedback')
    if stage == 'final_reflection_fallback' and (config or {}).get('final_reflection'):
        return 'final_error_type'
    if (stage in ('final_reflection_document', 'final_reflection_document_compact')
            and (config or {}).get('final_reflection')):
        return 'reflection_document'
    if stage in ('recheck_role', 'recheck_role_no_ocr'):
        if not uses_compare_review_role(config):
            raise ValueError(f'알 수 없는 역할 포함 재검토 단계: {stage}')
        return 'binary_verdict'
    if stage == 'role' and (config or {}).get('role_stage'):
        if uses_role_observation(config):
            return 'image_role_observation'
        return 'visual_evidence' if config.get('evidence_stage') else 'image_role'
    mode = (config or {}).get('classification_mode', 'single')
    if mode in ('two_stage_verdict_types_v1', 'guideline_verdict_candidates_final_v1'):
        priority = config.get('label_priority')
        if (not isinstance(priority, list) or len(priority) != 6 or any(not isinstance(label, str) for label in priority)
                or set(priority) != set(ERROR_PRIORITY)):
            raise ValueError('저장된 라벨 우선순위가 잘못되었습니다.')
        if stage in ('classify', 'classify_no_ocr'):
            contract = config.get('binary_contract', 'binary_verdict')
            if contract not in ('binary_verdict', 'role_verdict_v1', 'evidence_verdict'):
                raise ValueError('저장된 이진 판정 출력 형식이 잘못되었습니다.')
            return contract
        if stage in ('types', 'types_no_ocr'):
            return 'error_types'
        if stage in ('final', 'final_no_ocr') and mode == 'guideline_verdict_candidates_final_v1':
            return 'final_error_type'
        if stage in ('recheck', 'recheck_no_ocr') and config.get('recheck_appropriate'):
            return 'binary_verdict' if uses_appropriate_v1_review(config) or not config.get('evidence_stage') else 'evidence_verdict'
        raise ValueError(f'알 수 없는 2단계 판정 단계: {stage}')
    if mode == 'single':
        return 'classify'  # Historical manifests keep their single-label contract.
    priority = config.get('label_priority')
    if (mode not in ('multilabel_priority_v1', 'explicit_verdict_multilabel_v1') or not isinstance(priority, list)
            or len(priority) != 6 or any(not isinstance(label, str) for label in priority)
            or set(priority) != set(ERROR_PRIORITY)):
        raise ValueError('복수 유형 판정 방식 또는 저장된 라벨 우선순위가 잘못되었습니다.')
    return 'classify_verdict_types' if mode == 'explicit_verdict_multilabel_v1' else 'classify_types'


def binary_prompt_for_role(config, role):
    criteria = config.get('role_criteria', {})
    if not isinstance(role, str) or role not in IMAGE_ROLES or not isinstance(criteria.get(role), str) or not criteria[role].strip():
        raise ValueError('선택된 역할 또는 저장된 역할 기준이 잘못되었습니다.')
    prompt = config['prompts']['classify']
    if '{{role}}' not in prompt or '{{criterion}}' not in prompt:
        raise ValueError('역할별 이진 판정 프롬프트의 치환 위치가 없습니다.')
    return prompt.replace('{{role}}', role).replace('{{criterion}}', criteria[role])


def uses_empty_alt_prompt(config, data):
    alt = data.get('alt_text')
    return bool(config.get('empty_alt_prompt') and isinstance(alt, str) and not alt.strip())


def needs_appropriate_review(config, data, verdict, decision=None):
    # Old manifests retain their original review policy.
    if config.get('evidence_stage'):
        if uses_appropriate_v1_review(config):
            return bool(config.get('recheck_appropriate') and verdict == '적절'
                        and not uses_empty_alt_prompt(config, data))
        return bool(config.get('recheck_appropriate') and not uses_empty_alt_prompt(config, data)
                    and isinstance(decision, dict) and decision.get('needs_review') is True)
    return bool(config.get('recheck_appropriate') and verdict == '적절'
                and not (config.get('skip_empty_alt_review') and uses_empty_alt_prompt(config, data)))


def binary_prompt_for_input(config, data):
    if uses_empty_alt_prompt(config, data):
        return config['prompts']['empty_alt']
    if config.get('role_stage') and not config.get('evidence_stage'):
        return binary_prompt_for_role(config, data.get('_image_role'))
    return config['prompts']['classify']


def reflection_document_stage(config, document):
    """Choose the MD editor mechanically from Python character count."""
    threshold = config.get('reflection_document_compact_threshold',
                           REFLECTION_DOCUMENT_COMPACT_THRESHOLD)
    if (isinstance(document, str) and len(document) > threshold
            and config.get('prompts', {}).get('reflection_document_compact')):
        return 'reflection_document_compact'
    return 'reflection_document'


def type_reflection_document_stage(config, document):
    threshold = config.get('reflection_document_compact_threshold',
                           REFLECTION_DOCUMENT_COMPACT_THRESHOLD)
    if (isinstance(document, str) and len(document) > threshold
            and config.get('prompts', {}).get('type_reflection_document_compact')):
        return 'type_reflection_document_compact'
    return 'type_reflection_document'


def final_reflection_document_stage(config, document):
    threshold = config.get('reflection_document_compact_threshold',
                           REFLECTION_DOCUMENT_COMPACT_THRESHOLD)
    if (isinstance(document, str) and len(document) > threshold
            and config.get('prompts', {}).get('final_reflection_document_compact')):
        return 'final_reflection_document_compact'
    return 'final_reflection_document'


def request_payload(stage, image_path, data, config):
    contract = {'classify_no_ocr': 'classify', 'types_no_ocr': 'types', 'recheck_no_ocr': 'recheck',
                'final_no_ocr': 'final', 'recheck_role_no_ocr': 'recheck_role'}.get(stage, stage)
    output_contract = _response_contract(stage, config)
    supplied = {key: data.get(key) for key in METADATA_FIELDS}
    # Preserve the evaluated alt verbatim; normalize only context whitespace.
    for key, value in supplied.items():
        if stage == 'decorative_observer' and key in (
                'page_title', 'context_text', 'link_dest', 'heading_path') and isinstance(value, str):
            supplied[key] = value[:500]
            continue
        if key != 'alt_text' and isinstance(value, str):
            supplied[key] = re.sub(r'\s+', ' ', value).strip()
    if stage in ('decorative_router', 'reflection_visual_evidence'):
        supplied.pop('alt_text')
    elif stage == 'role':
        supplied.pop('alt_text')
    elif config.get('guideline_mode'):
        if contract == 'final_reflection_feedback':
            candidates = data.get('_candidate_assessment')
            binary = data.get('_binary_assessment')
            initial = data.get('_initial_final_assessment')
            document = data.get('_final_reflection_document')
            if (not isinstance(candidates, dict) or not isinstance(binary, dict)
                    or not isinstance(initial, dict) or not isinstance(document, str)):
                raise ValueError('최종 선택 Reflection 피드백 입력이 없습니다.')
            supplied.update(binary_assessment=binary, candidate_assessment=candidates,
                            initial_final_assessment=initial, current_guidelines=document)
            if config.get('reflection_mode', REFLECTION_TRAIN) == REFLECTION_TRAIN:
                actual = data.get('_actual_label')
                if actual not in ERROR_PRIORITY:
                    raise ValueError('최종 선택 Reflection 실제 오류 유형 입력이 없습니다.')
                supplied['actual_label'] = actual
        elif contract == 'final_reflection_fallback':
            candidates = data.get('_candidate_assessment')
            binary = data.get('_binary_assessment')
            feedback = data.get('_final_reflection_feedback')
            document = data.get('_final_reflection_document')
            if (not isinstance(candidates, dict) or not isinstance(binary, dict)
                    or not isinstance(feedback, dict)
                    or not isinstance(feedback.get('feedback'), str)
                    or not isinstance(document, str)):
                raise ValueError('최종 오류 유형 재선택 입력이 없습니다.')
            supplied.update(binary_assessment=binary, candidate_assessment=candidates,
                            reflection_feedback=feedback, current_guidelines=document)
        elif contract in ('final_reflection_document', 'final_reflection_document_compact'):
            document = data.get('_final_reflection_document')
            feedback = data.get('_final_reflection_feedback')
            if not isinstance(document, str) or not isinstance(feedback, dict):
                raise ValueError('최종 선택 Reflection 문서 수정 입력이 없습니다.')
            supplied = {'current_guidelines': document, 'new_feedback': feedback}
        elif contract == 'type_reflection_feedback':
            candidates = data.get('_candidate_assessment')
            binary = data.get('_binary_assessment')
            document = data.get('_type_reflection_document')
            if not isinstance(candidates, dict) or not isinstance(binary, dict) or not isinstance(document, str):
                raise ValueError('오류 유형 후보 Reflection 피드백 입력이 없습니다.')
            supplied.update(binary_assessment=binary,
                            initial_candidate_assessment=candidates,
                            current_guidelines=document)
            if config.get('reflection_mode', REFLECTION_TRAIN) == REFLECTION_TRAIN:
                actual = data.get('_actual_label')
                if actual not in ERROR_PRIORITY:
                    raise ValueError('후보 Reflection 실제 오류 유형 입력이 없습니다.')
                supplied['actual_label'] = actual
        elif contract == 'type_reflection_fallback':
            candidates = data.get('_candidate_assessment')
            binary = data.get('_binary_assessment')
            feedback = data.get('_type_reflection_feedback')
            if not isinstance(candidates, dict) or not isinstance(binary, dict) or not isinstance(feedback, dict):
                raise ValueError('오류 유형 후보 재생성 입력이 없습니다.')
            supplied.update(binary_assessment=binary,
                            initial_candidate_assessment=candidates,
                            reflection_feedback=feedback)
        elif contract in ('type_reflection_document', 'type_reflection_document_compact'):
            document = data.get('_type_reflection_document')
            feedback = data.get('_type_reflection_feedback')
            if not isinstance(document, str) or not isinstance(feedback, dict):
                raise ValueError('오류 유형 후보 Reflection 문서 수정 입력이 없습니다.')
            supplied = {'current_guidelines': document, 'new_feedback': feedback}
        elif stage == 'reflection_coverage_audit':
            initial = data.get('_initial_assessment')
            if not isinstance(initial, dict):
                raise ValueError('Frozen 시각 증거 대조 입력이 없습니다.')
            supplied['initial_assessment'] = initial
        elif contract == 'reflection_feedback':
            initial = data.get('_initial_assessment')
            document = data.get('_reflection_document')
            if not isinstance(initial, dict) or not isinstance(document, str):
                raise ValueError('Reflection 피드백 입력이 없습니다.')
            supplied.update(initial_assessment=initial, current_guidelines=document)
            if config.get('reflection_mode', REFLECTION_TRAIN) == REFLECTION_TRAIN:
                actual = data.get('_actual_label')
                if actual not in LABELS:
                    raise ValueError('Train Reflection 실제 라벨 입력이 없습니다.')
                supplied['actual_label'] = actual
        elif contract == 'reflection_fallback':
            feedback = data.get('_reflection_feedback')
            initial = data.get('_initial_assessment')
            if not isinstance(feedback, dict) or not isinstance(initial, dict):
                raise ValueError('Reflection 재판정의 최초 판정 또는 피드백이 없습니다.')
            supplied.update(initial_assessment=initial, reflection_feedback=feedback)
            if config.get('reflection_mode', REFLECTION_TRAIN) == REFLECTION_FROZEN:
                document = data.get('_reflection_document')
                if not isinstance(document, str):
                    raise ValueError('Frozen Reflection 재판정의 고정 가이드라인이 없습니다.')
                supplied['current_guidelines'] = document
        elif contract in ('reflection_document', 'reflection_document_compact'):
            document = data.get('_reflection_document')
            feedback = data.get('_reflection_feedback')
            if not isinstance(document, str) or not isinstance(feedback, dict):
                raise ValueError('Reflection 문서 수정 입력이 없습니다.')
            supplied = {'current_guidelines': document, 'new_feedback': feedback}
        elif contract == 'recheck':
            initial = data.get('_initial_assessment')
            if not isinstance(initial, dict):
                raise ValueError('가이드라인 재검토에 최초 판정 JSON이 없습니다.')
            supplied['initial_assessment'] = initial
        elif contract in ('types', 'final'):
            independent = config.get('error_classification_input') == 'image_metadata_candidates_v1'
            if not independent:
                binary = data.get('_binary_assessment')
                if not isinstance(binary, dict):
                    raise ValueError('가이드라인 오류 유형 판정에 최종 이진판정 JSON이 없습니다.')
                supplied['binary_assessment'] = binary
            if contract == 'final':
                candidates = data.get('_candidate_assessment')
                if not isinstance(candidates, dict):
                    raise ValueError('가이드라인 최종판정에 후보 집합 JSON이 없습니다.')
                supplied['candidate_assessment'] = candidates
    elif config.get('evidence_stage'):
        if stage in ('recheck_role', 'recheck_role_no_ocr'):
            if not uses_compare_review_role(config):
                raise ValueError('역할 포함 재검토 비교 설정이 없습니다.')
            supplied['visual_evidence'] = data.get('_visual_evidence')
        elif not (stage in ('recheck', 'recheck_no_ocr') and uses_appropriate_v1_review(config)):
            supplied['visual_evidence'] = data.get('_visual_evidence')
            if stage in ('recheck', 'recheck_no_ocr'):
                supplied['initial_assessment'] = data.get('_initial_assessment')
    elif config.get('input_mode') != 'image_metadata' and not stage.endswith('_no_ocr'):
        # Historical request reproduction for stored experiment audits only.
        supplied['cleaned_ocr_text'] = data.get('cleaned_ocr_text', '')
    prompt_key = (stage if stage in ('reflection_visual_evidence', 'reflection_coverage_audit')
                  else 'empty_alt_link_visual_review' if stage == 'empty_alt_link_visual_review'
                  else 'recheck' if contract == 'recheck_role' else contract)
    prompt = config['prompts'][prompt_key]
    if stage in ('classify', 'classify_no_ocr'):
        prompt = binary_prompt_for_input(config, data)
    image_url = 'data:image/png;base64,' + base64.b64encode(Path(image_path).read_bytes()).decode('ascii')
    content = [{'type': 'text', 'text': json.dumps(supplied, ensure_ascii=False)}]
    if contract not in ('reflection_document', 'reflection_document_compact',
                        'type_reflection_document', 'type_reflection_document_compact',
                        'final_reflection_document', 'final_reflection_document_compact'):
        content.insert(0, {'type': 'image_url', 'image_url': {'url': image_url}})
    max_tokens = (512 if stage in ('decorative_router', 'decorative_observer') else
                  config.get('reflection_document_max_tokens', config['max_tokens'])
                  if contract in ('reflection_document', 'reflection_document_compact',
                                  'type_reflection_document', 'type_reflection_document_compact',
                                  'final_reflection_document', 'final_reflection_document_compact')
                  else config['max_tokens'])
    response_schema = SCHEMAS[output_contract]
    if output_contract == 'final_error_type' and isinstance(supplied.get('candidate_assessment'), dict):
        # llama.cpp can enforce the current candidate pool in its grammar.  A
        # static six-label enum only asked the model to obey the prompt and
        # allowed internally valid JSON with an out-of-pool label.
        response_schema = json.loads(json.dumps(response_schema, ensure_ascii=False))
        response_schema['properties']['label']['enum'] = final_candidate_labels(
            supplied['candidate_assessment'])
    payload = {'model': 'cv-pbl-gemma', 'messages': [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': content}],
        'temperature': 0, 'seed': 42, 'max_tokens': max_tokens, 'cache_prompt': False,
        'chat_template_kwargs': {'enable_thinking': False},
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': output_contract, 'strict': True, 'schema': response_schema}}}
    if config.get('reflection_session') == 'separate_slot_v1':
        payload['id_slot'] = 1 if contract == 'recheck' else 0
    elif config.get('reflection_session') == 'label_feedback_v1':
        payload['id_slot'] = 1 if stage in (
            'decorative_router', 'decorative_observer',
            'reflection_visual_evidence', 'reflection_coverage_audit') or contract in (
            'reflection_feedback', 'reflection_document', 'reflection_document_compact',
            'type_reflection_feedback', 'type_reflection_document',
            'type_reflection_document_compact', 'final_reflection_feedback',
            'final_reflection_document', 'final_reflection_document_compact') else 0
    return payload


def parse_response(raw, stage, config=None):
    contract = _response_contract(stage, config)
    choice = raw['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError(f"Gemma 출력 미완료: finish_reason={choice.get('finish_reason')}; 토큰 한도를 확인하세요.")
    value = json.loads(choice['message']['content'])
    if not isinstance(value, dict) or set(value) != set(SCHEMAS[contract]['required']):
        raise ValueError('Gemma JSON 필드가 출력 형식과 다릅니다.')
    if contract == 'decorative_observation':
        if (value['visual_kind'] not in SCHEMAS[contract]['properties']['visual_kind']['enum']
                or value['other_error'] not in SCHEMAS[contract]['properties']['other_error']['enum']
                or type(value['pure_decorative_role']) is not bool
                or value['decorative_pattern'] not in SCHEMAS[contract]['properties']['decorative_pattern']['enum']
                or value['confidence'] not in SCHEMAS[contract]['properties']['confidence']['enum']
                or not isinstance(value['reason'], str) or len(value['reason']) > 350):
            raise ValueError('Gemma 장식오용형 구조화 관찰 형식이 잘못되었습니다.')
    elif contract == 'image_role':
        if (not isinstance(value['role'], str) or value['role'] not in IMAGE_ROLES
                or not isinstance(value['reason'], str) or not value['reason'].strip() or len(value['reason']) > 150):
            raise ValueError('Gemma 이미지 역할 형식이 잘못되었습니다.')
    elif contract == 'visual_evidence':
        if (not isinstance(value['role'], str) or value['role'] not in IMAGE_ROLES
                or not isinstance(value['reason'], str) or not value['reason'].strip() or len(value['reason']) > 150):
            raise ValueError('Gemma 시각 근거 역할 형식이 잘못되었습니다.')
        for key in ('visible_facts', 'context_quotes', 'required_information'):
            facts = value[key]
            if (not isinstance(facts, list) or len(facts) > 3
                    or any(not isinstance(fact, str) or not fact.strip() or len(fact) > 150 for fact in facts)):
                raise ValueError('Gemma 시각 근거 목록 형식이 잘못되었습니다.')
    elif contract == 'image_role_observation':
        roles = value['roles']
        if (not isinstance(roles, list) or not 1 <= len(roles) <= 4
                or any(not isinstance(role, str) or role not in OBSERVATION_ROLES for role in roles)
                or not isinstance(value['description'], str) or not value['description'].strip()
                or len(value['description']) > 350):
            raise ValueError('Gemma 역할 관찰 형식이 잘못되었습니다.')
        if len(set(roles)) != len(roles):
            raise ValueError('Gemma 역할 관찰 역할이 중복되었습니다.')
    elif contract == 'classify':
        if value['label'] not in LABELS or any(not isinstance(value[key], str) or not value[key].strip()
                                             or len(value[key]) > 350 for key in SCHEMAS[contract]['required']):
            raise ValueError('Gemma 라벨 또는 근거 형식이 잘못되었습니다.')
    elif contract == 'reflection_feedback':
        if not isinstance(value['feedback'], str) or not value['feedback'].strip() or len(value['feedback']) > 350:
            raise ValueError('Reflection 피드백 형식이 잘못되었습니다.')
    elif contract == 'reflection_feedback_gate':
        if (value['action'] not in ('피드백 제공', '통과')
                or not isinstance(value['feedback'], str) or len(value['feedback']) > 350
                or not isinstance(value['reason'], str) or not value['reason'].strip()
                or len(value['reason']) > 350
                or (value['action'] == '피드백 제공' and not value['feedback'].strip())
                or (value['action'] == '통과' and value['feedback'] != '')):
            raise ValueError('Frozen Reflection 피드백 선택 형식이 잘못되었습니다.')
    elif contract == 'reflection_document':
        markdown = value['markdown']
        if (not isinstance(markdown, str) or not markdown.strip()
                or not markdown.lstrip().startswith('#')):
            raise ValueError('Reflection Markdown 형식이 잘못되었습니다.')
    elif contract in ('binary_verdict', 'evidence_verdict', 'role_verdict_v1', 'error_types',
                      'final_error_type', 'classify_types', 'classify_verdict_types'):
        if any(not isinstance(value[key], str) or not value[key].strip() or len(value[key]) > 350
               for key in ('reason', 'image_evidence', 'context_evidence')):
            raise ValueError('Gemma 판정 근거 형식이 잘못되었습니다.')
        if contract == 'final_error_type':
            if value['label'] not in ERROR_PRIORITY:
                raise ValueError('Gemma 최종 오류 유형 형식이 잘못되었습니다.')
            return value
        if contract in ('binary_verdict', 'evidence_verdict', 'role_verdict_v1'):
            if value['verdict'] not in ('적절', '부적절'):
                raise ValueError('Gemma 이진 판정 형식이 잘못되었습니다.')
            if contract == 'evidence_verdict':
                checks = value['checks']
                if not isinstance(value['needs_review'], bool):
                    raise ValueError('Gemma 재검토 필요 여부 형식이 잘못되었습니다.')
                if not isinstance(checks, list) or len(checks) > 3:
                    raise ValueError('Gemma 정보 충족 검사 목록이 너무 깁니다.')
                for check in checks:
                    if (not isinstance(check, dict) or set(check) != {'information', 'alt_quote', 'context_quote', 'status'}
                            or check['status'] not in ('충족', '미충족', '확인불가')
                            or any(not isinstance(check[key], str) or len(check[key]) > 350
                                   for key in ('information', 'alt_quote', 'context_quote'))
                            or not check['information'].strip()):
                        raise ValueError('Gemma 정보 충족 검사 형식이 잘못되었습니다.')
            if contract == 'role_verdict_v1':
                roles, checks = value['image_roles'], value['criteria_checks']
                if (not isinstance(roles, list) or not 1 <= len(roles) <= 5
                        or any(not isinstance(role, str) or role not in IMAGE_ROLES for role in roles)
                        or len(set(roles)) != len(roles)):
                    raise ValueError('Gemma 이미지 역할 형식이 잘못되었습니다.')
                if not isinstance(checks, list) or not 1 <= len(checks) <= 6:
                    raise ValueError('Gemma 역할별 기준 검사 목록이 비었거나 너무 깁니다.')
                for check in checks:
                    if (not isinstance(check, dict) or set(check) != {'criterion', 'alt_evidence', 'context_evidence', 'status'}
                            or check['status'] not in ('충족', '미충족', '확인불가')
                            or any(not isinstance(check[key], str) or not check[key].strip() or len(check[key]) > 350
                                   for key in ('criterion', 'alt_evidence', 'context_evidence'))):
                        raise ValueError('Gemma 역할별 기준 검사 형식이 잘못되었습니다.')
                statuses = {check['status'] for check in checks}
                if (value['verdict'] == '적절' and '미충족' in statuses
                        or value['verdict'] == '부적절' and statuses == {'충족'}):
                    raise ValueError('Gemma 기준 충족 여부와 이진 판정이 모순됩니다.')
            return value
        matches = value['matched_types']
        if not isinstance(matches, list) or len(matches) > 6 or (contract == 'error_types' and not matches):
            raise ValueError('matched_types는 오류 유형 목록이어야 합니다.')
        reasons = {}
        normalized_matches = {}
        for item in matches:
            if (not isinstance(item, dict) or set(item) != {'label', 'reason'}
                    or not isinstance(item['label'], str) or item['label'] not in ERROR_PRIORITY
                    or not isinstance(item['reason'], str)
                    or not item['reason'].strip() or len(item['reason']) > 350):
                raise ValueError('오류 유형별 근거 형식이 잘못되었습니다.')
            if item['label'] in reasons and stage != 'type_reflection_fallback':
                raise ValueError('오류 유형이 중복되었습니다.')
            reasons[item['label']] = item['reason']
            normalized_matches[item['label']] = item
        value['matched_types'] = list(normalized_matches.values())
        if contract == 'error_types':
            return value
        # New runs require an explicit verdict; preserve historical v4 decoding.
        if contract == 'classify_verdict_types':
            verdict = value['verdict']
            if verdict not in ('적절', '부적절') or (verdict == '적절') != (not matches):
                raise ValueError('적절은 빈 유형 목록, 부적절은 하나 이상의 유형이 필요합니다.')
            winner = '적절' if verdict == '적절' else next(label for label in config['label_priority'] if label in reasons)
        else:
            winner = next((label for label in config['label_priority'] if label in reasons), '적절')
        # Selection happens only here; priority is never added to the model request.
        value.update(label=winner, assessment_reason=value['reason'],
                     reason=reasons.get(winner, value['reason']),
                     decision_rule={'name': 'explicit-verdict-priority-v1' if contract == 'classify_verdict_types' else 'error-priority-v1',
                                    'priority': list(config['label_priority'])})
    return value


def audit_request(payload, image_path):
    content = payload['messages'][1]['content']
    if content and content[0].get('type') == 'image_url':
        content = [{'type': 'image_file', 'path': str(image_path),
                    'sha256': hashlib.sha256(Path(image_path).read_bytes()).hexdigest()}, *content[1:]]
    return {**payload, 'messages': [payload['messages'][0], {'role': 'user', 'content': content}]}


def server_command(config, port):
    slots = int(config.get('parallel_slots', config.get('session_slots', 1)))
    return [config['server_path'], '--model', config['model_path'],
            '--mmproj', config['mmproj_path'], '--alias', 'cv-pbl-gemma',
            '--host', '127.0.0.1', '--port', str(port), '--n-gpu-layers', 'all',
            '--split-mode', 'none', '--main-gpu', '0', '--mmproj-offload',
            '--ctx-size', str(config['context_size'] * slots), '--parallel', str(slots),
            '--reasoning', 'off', '--no-context-shift', '--fit', 'off', '--log-verbosity', '4']


class GemmaServer:
    def __init__(self, config, device, run_dir):
        self.config, self.device, self.run_dir = config, device, Path(run_dir)
        self.process = self.log = None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def http(self, path, payload=None, timeout=3):
        request = urllib.request.Request(self.url + path, headers={'Content-Type': 'application/json'},
                                         data=None if payload is None else json.dumps(payload).encode())
        try:
            with self.opener.open(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'Gemma HTTP {exc.code}: {exc.read().decode(errors="replace")[:1500]}') from exc

    def start(self):
        validate_config(self.config)
        if not re.fullmatch(r'gpu:\d+', self.device):
            raise ValueError('Gemma도 GPU 장치만 지원합니다.')
        gpu = subprocess.run(['nvidia-smi', '-i', self.device.split(':')[1], '--query-gpu=uuid,name',
                              '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout.strip()
        gpu_uuid, gpu_name = gpu.split(',', 1)
        env = os.environ.copy()
        # Avoid inherited llama.cpp switches silently changing model/device/backend.
        env = {key: value for key, value in env.items() if not key.startswith('LLAMA_ARG_')}
        env['CUDA_VISIBLE_DEVICES'] = gpu_uuid.strip()
        env.pop('CUDA_LAUNCH_BLOCKING', None)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.url = f'http://127.0.0.1:{port}'
        command = server_command(self.config, port)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log = (self.run_dir / 'gemma-server.log').open('a')
        offset = self.log.tell()
        self.process = owned_process(command, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        self.runtime = {'pid': self.process.pid, 'requested_device': self.device, 'gpu_uuid': gpu_uuid.strip(),
                        'gpu_name': gpu_name.strip(), 'command': command, 'started_at': now()}
        write_json(self.run_dir / 'gemma-runtime.json', self.runtime)
        deadline = time.monotonic() + self.config['startup_timeout']
        while time.monotonic() < deadline:
            if (self.run_dir / 'STOP').exists():
                raise InterruptedError('Gemma 로딩 중 중지 요청')
            if self.process.poll() is not None:
                raise RuntimeError('Gemma 서버 로드 실패. gemma-server.log를 확인하세요.')
            try:
                ready = self.http('/health').get('status') == 'ok'
            except (OSError, RuntimeError):
                ready = False
            if ready:
                with (self.run_dir / 'gemma-server.log').open() as log:
                    log.seek(offset)
                    startup = log.read()
                offload = re.search(r'offloaded (\d+)/(\d+) layers to GPU', startup)
                if (not offload or int(offload[1]) == 0 or offload[1] != offload[2]
                        or 'CLIP using CUDA0 backend' not in startup):
                    raise RuntimeError('전체 레이어 GPU 로드를 확인하지 못했습니다. CPU 대체를 허용하지 않습니다.')
                self.runtime['gpu_layers'] = offload.group(0)
                self.runtime['vision_backend'] = 'CUDA0'
                write_json(self.run_dir / 'gemma-runtime.json', self.runtime)
                return self
            time.sleep(0.5)
        raise TimeoutError('Gemma 로딩 시간 초과')

    def __enter__(self):
        try:
            return self.start()
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log:
            self.log.close()
        if hasattr(self, 'runtime'):
            self.runtime['stopped_at'] = now()
            write_json(self.run_dir / 'gemma-runtime.json', self.runtime)

    def __exit__(self, *args):
        self.close()

    def _infer_once(self, stage, image_path, data, audit_path, slot_id=None):
        payload = request_payload(stage, image_path, data, self.config)
        if slot_id is not None:
            payload['id_slot'] = int(slot_id)
        started = time.perf_counter()
        # Preserve the exact text input, settings, and raw response without duplicating PNG bytes.
        audit = audit_request(payload, image_path)
        write_json(audit_path.with_suffix('.request.json'), audit)
        raw = self.http('/v1/chat/completions', payload, self.config['request_timeout'])
        write_json(audit_path, raw)
        value = parse_response(raw, stage, self.config)
        relative = str(audit_path.relative_to(self.run_dir))
        return value, {'stage': stage, 'usage': raw.get('usage', {}),
                       'elapsed_seconds': round(time.perf_counter() - started, 3), 'raw_path': relative}

    def _shared_role(self, image_path, data, audit_path):
        path = self.run_dir / 'raw' / 'gemma_role' / audit_path.name
        expected = audit_request(request_payload('role', image_path, data, self.config), image_path)
        if read_json(path.with_suffix('.request.json')) == expected:
            try:
                role = parse_response(read_json(path), 'role', self.config)
                return role, {'stage': 'role', 'usage': {}, 'elapsed_seconds': 0, 'reused': True,
                              'raw_path': str(path.relative_to(self.run_dir))}
            except (KeyError, TypeError, IndexError, ValueError):
                pass
        return self._infer_once('role', image_path, data, path)

    def infer(self, stage, image_path, data, audit_path):
        if self.config.get('role_stage') and stage in ('classify', 'classify_no_ocr'):
            role, role_stats = self._shared_role(image_path, data, audit_path)
            if (self.run_dir / 'STOP').exists():
                raise InterruptedError('역할 판정 후 중지 요청')
            evidence = {**data, '_visual_evidence': role}
            if 'role' in role:
                evidence['_image_role'] = role['role']
            value, stats = self._infer_verdict(stage, image_path, evidence, audit_path)
            value['role_decision'] = role
            stats['calls'].insert(0, role_stats)
            stats['raw_paths'].insert(0, role_stats['raw_path'])
            stats['elapsed_seconds'] = round(stats['elapsed_seconds'] + role_stats['elapsed_seconds'], 3)
            stats['usage'] = {key: stats['usage'].get(key, 0) + role_stats['usage'].get(key, 0)
                              for key in ('prompt_tokens', 'completion_tokens')}
            return value, stats
        return self._infer_verdict(stage, image_path, data, audit_path)

    def infer_binary(self, stage, image_path, data, audit_path, slot_id):
        return self._infer_once(stage, image_path, data, audit_path, slot_id)

    def infer_decorative_router(self, image_path, data, audit_path, slot_id=1):
        return self._infer_once('decorative_router', image_path, data, audit_path, slot_id)

    def infer_decorative_observer(self, image_path, data, audit_path, slot_id=1):
        return self._infer_once('decorative_observer', image_path, data, audit_path, slot_id)

    def infer_reflection_feedback(self, image_path, data, audit_path):
        return self._infer_once('reflection_feedback', image_path, data, audit_path)

    def infer_reflection_visual_evidence(self, image_path, data, audit_path):
        return self._infer_once('reflection_visual_evidence', image_path, data, audit_path)

    def infer_reflection_coverage_audit(self, image_path, data, audit_path):
        return self._infer_once('reflection_coverage_audit', image_path, data, audit_path)

    def infer_reflection_fallback(self, image_path, data, audit_path):
        return self._infer_once('reflection_fallback', image_path, data, audit_path)

    def infer_reflection_document(self, image_path, data, audit_path):
        document = data.get('_reflection_document')
        stage = reflection_document_stage(self.config, document)
        return self._infer_once(stage, image_path, data, audit_path)

    def infer_type_reflection_feedback(self, image_path, data, audit_path):
        return self._infer_once('type_reflection_feedback', image_path, data, audit_path)

    def infer_type_reflection_fallback(self, image_path, data, audit_path):
        return self._infer_once('type_reflection_fallback', image_path, data, audit_path)

    def infer_type_reflection_document(self, image_path, data, audit_path):
        stage = type_reflection_document_stage(
            self.config, data.get('_type_reflection_document'))
        return self._infer_once(stage, image_path, data, audit_path)

    def infer_final_reflection_feedback(self, image_path, data, audit_path):
        return self._infer_once('final_reflection_feedback', image_path, data, audit_path)

    def infer_final_reflection_fallback(self, image_path, data, audit_path):
        return self._infer_once('final_reflection_fallback', image_path, data, audit_path)

    def infer_final_reflection_document(self, image_path, data, audit_path):
        stage = final_reflection_document_stage(
            self.config, data.get('_final_reflection_document'))
        return self._infer_once(stage, image_path, data, audit_path)

    def infer_after_binary(self, stage, image_path, data, audit_path, binary, first,
                           binary_source=None, slot_id=None, type_reflector=None,
                           final_reflector=None):
        return self._infer_verdict(stage, image_path, data, audit_path, binary, first,
                                   binary_source, slot_id, type_reflector, final_reflector)

    def infer_from_binary(self, stage, image_path, data, audit_path, binary, gate_stats,
                          binary_source=EMPTY_ALT_LINK_GATE):
        return self.infer_after_binary(stage, image_path, data, audit_path, binary, gate_stats,
                                       binary_source)

    def _infer_verdict(self, stage, image_path, data, audit_path, binary=None, first=None,
                       binary_source=None, slot_id=None, type_reflector=None,
                       final_reflector=None):
        if binary is None:
            binary, first = self._infer_once(stage, image_path, data, audit_path, slot_id)
        mode = self.config.get('classification_mode')
        if mode not in ('two_stage_verdict_types_v1', 'guideline_verdict_candidates_final_v1') or stage not in ('classify', 'classify_no_ocr'):
            return binary, {'usage': first['usage'], 'elapsed_seconds': first['elapsed_seconds'],
                            'calls': [first], 'raw_paths': [first['raw_path']]}
        calls = [first]
        review = None
        role_review = None
        if needs_appropriate_review(self.config, data, binary['verdict'], binary):
            if (self.run_dir / 'STOP').exists():
                raise InterruptedError('적절 재검토 전 중지 요청')
            review_stage = 'recheck_no_ocr' if stage == 'classify_no_ocr' else 'recheck'
            review_path = audit_path.with_name(audit_path.stem + '.recheck.json')
            review_data = data if uses_appropriate_v1_review(self.config) else {**data, '_initial_assessment': binary}
            review, stats = self._infer_once(review_stage, image_path, review_data, review_path, slot_id)
            calls.append(stats)
            if uses_compare_review_role(self.config):
                if (self.run_dir / 'STOP').exists():
                    raise InterruptedError('역할 포함 재검토 전 중지 요청')
                role_stage = 'recheck_role_no_ocr' if stage == 'classify_no_ocr' else 'recheck_role'
                role_path = audit_path.with_name(audit_path.stem + '.recheck_role.json')
                role_review, stats = self._infer_once(role_stage, image_path, data, role_path, slot_id)
                calls.append(stats)
        decision = review or binary
        if self.config.get('binary_only'):
            evidence = decision
            return {'verdict': decision['verdict'], 'label': None, 'matched_types': [],
                    'assessment_reason': evidence['reason'], 'reason': evidence['reason'],
                    'image_evidence': evidence['image_evidence'],
                    'context_evidence': evidence['context_evidence'],
                    'decision_rule': {'name': 'binary-only-v1'},
                    'binary_decision': binary, 'type_classification': None,
                    'final_decision': None, 'binary_only': True,
                    **({'binary_decision_source': binary_source} if binary_source else {}),
                    **({'binary_route': 'empty_alt' if uses_empty_alt_prompt(self.config, data) else 'standard'}
                       if self.config.get('empty_alt_prompt') else {}),
                    **({'review_decision': review} if self.config.get('recheck_appropriate') else {}),
                    **({'role_review_decision': role_review} if uses_compare_review_role(self.config) else {})}, {
                        'usage': {key: sum(item['usage'].get(key, 0) for item in calls)
                                  for key in ('prompt_tokens', 'completion_tokens')},
                        'elapsed_seconds': round(sum(item['elapsed_seconds'] for item in calls), 3),
                        'calls': calls, 'raw_paths': [item['raw_path'] for item in calls]}
        types = initial_types = final = initial_final = None
        final_selection_source = None
        final_contract_recovery = None
        type_reflection_fields = {}
        final_reflection_fields = {}
        if decision['verdict'] == '부적절':
            if (self.run_dir / 'STOP').exists():
                raise InterruptedError('유형 판정 전 중지 요청')
            type_stage = 'types_no_ocr' if stage == 'classify_no_ocr' else 'types'
            types_path = audit_path.with_name(audit_path.stem + '.types.json')
            types, stats = self._infer_once(type_stage, image_path,
                                            {**data, '_binary_assessment': decision}, types_path, slot_id)
            calls.append(stats)
            if type_reflector is not None:
                initial_types = types
                types, reflection_calls, type_reflection_fields = type_reflector(types, decision)
                calls.extend(reflection_calls)
            if mode == 'guideline_verdict_candidates_final_v1' and not self.config.get('candidate_only'):
                if (self.run_dir / 'STOP').exists():
                    raise InterruptedError('최종 오류 유형 판정 전 중지 요청')
                candidates = final_candidate_labels(types)
                if len(candidates) == 1:
                    final = candidate_priority_decision(types, self.config['label_priority'])
                    final_selection_source = 'single_candidate_v1'
                else:
                    final_stage = 'final_no_ocr' if stage == 'classify_no_ocr' else 'final'
                    final_path = audit_path.with_name(audit_path.stem + '.final.json')
                    attempted, stats = self._infer_once(final_stage, image_path, {
                        **data, '_binary_assessment': decision,
                        '_candidate_assessment': types}, final_path, slot_id)
                    calls.append(stats)
                    final, final_contract_recovery = enforce_final_candidate_pool(
                        attempted, types, self.config['label_priority'])
                    final_selection_source = ('candidate_priority_contract_fallback_v1'
                                              if final_contract_recovery
                                              else 'candidate_schema_v1')
                if final_reflector is not None:
                    initial_final = final
                    final, reflection_calls, final_reflection_fields = final_reflector(
                        final, types, decision)
                    calls.extend(reflection_calls)
                    if final['label'] not in set(candidates):
                        raise ValueError('Gemma 최종 Reflection 결과가 고정 후보 집합에 없습니다.')
        elif binary['verdict'] == '부적절' and review and review['verdict'] == '적절':
            if (self.run_dir / 'STOP').exists():
                raise InterruptedError('초기 유형 판정 전 중지 요청')
            type_stage = 'types_no_ocr' if stage == 'classify_no_ocr' else 'types'
            types_path = audit_path.with_name(audit_path.stem + '.initial_types.json')
            initial_types, stats = self._infer_once(type_stage, image_path, data, types_path, slot_id)
            calls.append(stats)
        reasons = {item['label']: item['reason'] for item in types['matched_types']} if types else {}
        priority_winner = next(label for label in self.config['label_priority'] if label in reasons) if types else '적절'
        candidate_only = bool(self.config.get('candidate_only'))
        winner = None if candidate_only else final['label'] if final else priority_winner
        evidence = final or types or decision
        return {'verdict': decision['verdict'], 'label': winner,
                'matched_types': types['matched_types'] if types else [],
                'assessment_reason': evidence['reason'],
                'reason': evidence['reason'] if final else reasons.get(winner, decision['reason']),
                'image_evidence': evidence['image_evidence'], 'context_evidence': evidence['context_evidence'],
                'decision_rule': {'name': ('candidate-only-v1' if candidate_only
                                           else 'guideline-candidate-final-v1' if final
                                           else 'two-stage-verdict-types-v1'),
                                  'priority': list(self.config['label_priority'])},
                'binary_decision': binary, 'type_classification': types,
                **({'candidate_only': True, 'final_decision': None}
                   if candidate_only else {}),
                **({'binary_decision_source': binary_source} if binary_source else {}),
                **({'final_decision': final, 'priority_label': priority_winner}
                   if mode == 'guideline_verdict_candidates_final_v1' and not candidate_only else {}),
                **({'final_selection_source': final_selection_source}
                   if final_selection_source else {}),
                **({'final_contract_recovery': final_contract_recovery}
                   if final_contract_recovery else {}),
                **({'initial_final_decision': initial_final}
                   if initial_final and not final_reflection_fields else {}),
                **({'initial_type_classification': initial_types,
                    'initial_label': next(label for label in self.config['label_priority']
                                          if label in {item['label'] for item in initial_types['matched_types']})}
                   if initial_types and not type_reflection_fields else {}),
                **type_reflection_fields,
                **final_reflection_fields,
                **({'binary_route': 'empty_alt' if uses_empty_alt_prompt(self.config, data) else 'standard'}
                   if self.config.get('empty_alt_prompt') else {}),
                **({'review_decision': review} if self.config.get('recheck_appropriate') else {}),
                **({'role_review_decision': role_review} if uses_compare_review_role(self.config) else {})}, {
                    'usage': {key: sum(item['usage'].get(key, 0) for item in calls)
                              for key in ('prompt_tokens', 'completion_tokens')},
                    'elapsed_seconds': round(sum(item['elapsed_seconds'] for item in calls), 3),
                    'calls': calls, 'raw_paths': [item['raw_path'] for item in calls]}
