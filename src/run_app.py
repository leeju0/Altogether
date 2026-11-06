"""Local launcher for image + metadata Gemma classification."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import streamlit as st

from common import (
    DEFAULT_DATASET,
    create_run,
    get_status,
    launch_run,
    load_manifest,
    load_records,
    list_runs,
    request_stop,
    load_result,
    read_json,
    run_elapsed_seconds,
    write_json,
)
from gemma import (LABELS, METADATA_FIELDS, MODEL_VARIANTS, REFLECTION_FROZEN,
                   REFLECTION_TRAIN, default_config)
from benchmark import (build_benchmark, show_benchmark, candidate_labels, label_csv,
                       read_label_csv, score_labels)
from reflection import (FILENAME as REFLECTION_FILENAME, FINAL_FILENAME as FINAL_REFLECTION_FILENAME,
                        TYPE_FILENAME as TYPE_REFLECTION_FILENAME, write_document,
                        write_final_document, write_type_document)

ERROR_TYPES = ('파일명형', '무관형', '무의미형', '장식오용형', '중복형', '불충분형')
LIVE_REFRESH_SECONDS = 10
PUBLIC_109_PRESET = '공개 Test 109 · 현재 MD에서 이어서 Train'
COMBINED_FROZEN_PRESET = 'Train 2,400 + 공개 Test 109 · Frozen 분리 평가'
DATASET_PRESETS = {
    'Train · 정답 및 성능 평가': DEFAULT_DATASET,
    'Test · Frozen MD 제출 예측': DEFAULT_DATASET.parent / 'test',
    PUBLIC_109_PRESET: DEFAULT_DATASET.parent / 'test',
    COMBINED_FROZEN_PRESET: DEFAULT_DATASET,
}
BOUNDARY_CASES_CSV = DEFAULT_DATASET.parent / 'docs' / '경계사례_20261013.csv'
TEST_BASELINE_DOC = DEFAULT_DATASET.parent / 'docs' / '베이스라인_20261026.md'
TEST_BASELINE = {
    'count': 584, 'macro_f1': .631, 'binary_f1': .747,
    'combined_score': .689, 'accuracy': .764,
}

REFLECTION_MODE_LABELS = {
    REFLECTION_TRAIN: 'Train · 최초 이진 오답만 피드백 생성 및 MD 갱신',
    REFLECTION_FROZEN: 'Frozen MD · 학습된 MD로 추론',
}


def _reflection_sources(require_type=False, require_final=False) -> list[Path]:
    """Runs whose train Reflection document can be snapshotted for inference."""
    sources = []
    for run_dir in list_runs():
        try:
            config = (load_manifest(run_dir).get('gemma') or {})
        except (FileNotFoundError, TypeError, ValueError):
            continue
        if (config.get('label_reflection')
                and config.get('reflection_mode', REFLECTION_TRAIN) == REFLECTION_TRAIN
                and (Path(run_dir) / REFLECTION_FILENAME).is_file()
                and (not require_type or (Path(run_dir) / TYPE_REFLECTION_FILENAME).is_file())
                and (not require_final or (Path(run_dir) / FINAL_REFLECTION_FILENAME).is_file())):
            sources.append(Path(run_dir))
    return sources


def _is_priority_run(config: dict | None) -> bool:
    return bool(config and config.get('classification_mode') in (
        'multilabel_priority_v1', 'explicit_verdict_multilabel_v1', 'two_stage_verdict_types_v1',
        'guideline_verdict_candidates_final_v1'))


def _matched_type_summary(result: dict, config: dict) -> str:
    if result.get('verdict') == '적절':
        if result.get('review_decision'):
            return '생략 (재검토 후 적절)'
        if config.get('role_observation_mode'):
            return '생략 (1차 판정 적절)'
        return '생략 (1단계 적절)' if config.get('classification_mode') == 'two_stage_verdict_types_v1' else '적절 (모델 명시)'
    matches = [item.get('label') for item in result.get('matched_types', [])
               if isinstance(item, dict) and item.get('label') in ERROR_TYPES]
    priority = config.get('label_priority')
    if isinstance(priority, list):
        found = set(matches)
        matches = [label for label in priority if label in found]
    return ' · '.join(matches) if matches else '오류 유형 없음 → 적절'


def _record_id(row: dict) -> str:
    return str(row.get("record_id", row.get("id", "")))


def _selection(records: list[dict]) -> list[str]:
    all_option = f'전체 {len(records):,}개'
    mode = st.radio("실행 대상", ("처음 N개", "ID 범위", "직접 입력", all_option), horizontal=True)
    ids = [_record_id(row) for row in records]
    if mode == all_option:
        return ids
    if mode == "처음 N개":
        count = st.number_input("개수", 1, len(ids), min(20, len(ids)), step=1)
        return ids[:count]
    if mode == "ID 범위":
        start, end = st.select_slider("ID 범위", ids, value=(ids[0], ids[-1]))
        return ids[ids.index(start) : ids.index(end) + 1]
    raw = st.text_area("ID 목록 (쉼표 또는 줄바꿈으로 구분)")
    requested = {part.strip() for part in raw.replace(",", "\n").splitlines() if part.strip()}
    return [record_id for record_id in ids if record_id in requested]


def _use_dataset_preset() -> None:
    st.session_state['dataset_path'] = str(DATASET_PRESETS[st.session_state['dataset_preset']])


def _public_test_labels(cases_csv) -> dict[str, str]:
    with Path(cases_csv).open(encoding='utf-8-sig', newline='') as source:
        cases = [row for row in csv.DictReader(source) if row.get('분할') == 'test']
    labels = {}
    for case in cases:
        rid, label = case.get('record_id'), case.get('데이터셋_라벨')
        if not rid or rid in labels or label not in LABELS:
            raise ValueError('공개 평가 라벨에 중복 ID 또는 알 수 없는 라벨이 있습니다.')
        labels[rid] = label
    return labels


def _public_test_records(test_dir=DEFAULT_DATASET.parent / 'test', cases_csv=BOUNDARY_CASES_CSV) -> list[dict]:
    test_dir = Path(test_dir).resolve()
    labels = _public_test_labels(cases_csv)
    rows = load_records(test_dir)
    available = {row['record_id'] for row in rows}
    if len(labels) != 109 or not set(labels) <= available:
        raise ValueError('공개 평가 라벨은 Test에 존재하는 서로 다른 109건이어야 합니다.')
    return [{**row, 'label': labels[row['record_id']], '_dataset_dir': str(test_dir)}
            for row in rows if row['record_id'] in labels]


def _combined_frozen_records(train_dir=DEFAULT_DATASET, test_dir=DEFAULT_DATASET.parent / 'test',
                             cases_csv=BOUNDARY_CASES_CSV) -> list[dict]:
    train_dir = Path(train_dir).resolve()
    train = [{**row, '_dataset_dir': str(train_dir), '_evaluation_group': 'train_2400'}
             for row in load_records(train_dir)]
    public = [{**row, '_evaluation_group': 'public_test_109'}
              for row in _public_test_records(test_dir, cases_csv)]
    if len(train) != 2400 or set(row['record_id'] for row in train) & set(row['record_id'] for row in public):
        raise ValueError('Frozen 분리 평가는 서로 다른 Train 2,400건과 공개 Test 109건이어야 합니다.')
    return train + public


def _submission_labels(run_dir: Path, record_ids: list[str],
                       cases_csv=BOUNDARY_CASES_CSV) -> dict[str, str]:
    labels = {}
    for record_id in record_ids:
        result = load_result(run_dir, 'gemma_no_ocr', record_id)
        if result and result.get('status') == 'ok' and result.get('label') in LABELS:
            labels[record_id] = result['label']
    if cases_csv is not None and Path(cases_csv).is_file():
        labels.update({rid: label for rid, label in _public_test_labels(cases_csv).items()
                       if rid in labels})
    return labels




def _show_test_predictions(run_dir: Path, dataset_dir: Path) -> None:
    expected_ids = [_record_id(row) for row in load_records(dataset_dir)]
    cases_csv = dataset_dir.parent / 'docs' / BOUNDARY_CASES_CSV.name
    raw_predictions = _submission_labels(run_dir, expected_ids, None)
    predictions = _submission_labels(run_dir, expected_ids, cases_csv)
    st.info('Test 전체 정답은 공개되지 않았으므로 예측 분포와 제출 CSV를 표시합니다. 공개 정답 적용은 별도로 안내합니다.')
    corrected = sum(predictions.get(record_id) != label
                    for record_id, label in raw_predictions.items())
    if corrected:
        st.caption(f'공개 정답 109건을 제출에 적용하여 모델 예측 {corrected:,}건을 보정했습니다. '
                   '데이터셋_라벨을 사용하며 원본 모델 결과는 보존합니다.')
    st.dataframe([{'record_id': record_id, 'label': label}
                  for record_id, label in predictions.items()], hide_index=True, height=250)
    count, total = len(predictions), len(expected_ids)
    if predictions:
        st.markdown('**레이블별 예측 분포**')
        st.caption(f'성능 점수가 아닌 완료된 예측 {count:,}개 기준 출력 비율입니다.')
        counts = {label: sum(predicted == label for predicted in predictions.values())
                  for label in LABELS}
        st.dataframe([
            {'라벨': label,
             '예측 수': counts[label],
             '예측 비율': f'{counts[label] / count:.1%}'}
            for label in LABELS
        ], hide_index=True)
        complete = count == total
        st.download_button(
            f'{"제출" if complete else "부분 예측"} CSV 다운로드 · {count:,}/{total:,}개',
            label_csv(predictions), file_name=f'{run_dir.name}_submission.csv', mime='text/csv')
        if not complete:
            st.warning('아직 모든 Test 문제의 예측이 끝나지 않았습니다. 이 CSV는 대회 제출용으로 사용할 수 없습니다.')
    else:
        st.caption('저장된 예측을 기다리는 중입니다.')


def _prompt_template(config: dict) -> str:
    prompts = config.get('prompts') or {}
    evidence_review = config.get('evidence_stage') and config.get('review_policy') != 'appropriate_v1'
    stages = [('clean', 'OCR 정제')]
    if config.get('decorative_gate'):
        stages.extend([
            ('decorative_router', '장식오용형 전용 · alt 비공개 후보 라우터'),
            ('decorative_observer', '장식오용형 전용 · 구조화 관찰·결정 규칙'),
        ])
    if config.get('role_stage'):
        title = ('사전 단계 · 이미지 역할 복수 선택 + 이미지 설명' if config.get('role_observation_mode') else
                 '사전 단계 · 이미지·문맥 관찰 근거' if config.get('evidence_stage') else '사전 단계 · 이미지 역할 판정')
        stages.append(('role', title))
    stages.append(('classify', '1단계 · 적절/부적절' if config.get('classification_mode') in (
                       'two_stage_verdict_types_v1', 'guideline_verdict_candidates_final_v1')
                   else 'alt 품질 판정'))
    if config.get('label_reflection'):
        if config.get('frozen_evidence_pipeline'):
            stages.extend([
                ('reflection_visual_evidence', '2단계 · alt 비공개 시각 근거 추출'),
                ('reflection_coverage_audit', '3단계 · 시각 근거와 alt·문맥 대조'),
            ])
        stages.extend([
            ('reflection_feedback', '2단계 · 최초 이진 오답만 Reflection 피드백 생성'),
            ('reflection_fallback', '3단계 · 피드백 생성 문제만 이진 재판정'),
        ])
        if config.get('reflection_mode', REFLECTION_TRAIN) == REFLECTION_TRAIN:
            stages.extend([
                ('reflection_document', '4A단계 · 누적 MD 일반 갱신 (5,000자 이하)'),
                ('reflection_document_compact', '4B단계 · 누적 MD 중복 정리·갱신 (5,000자 초과)'),
            ])
    if config.get('empty_alt_prompt'):
        stages[-1] = ('classify', '1단계 · 적절/부적절 · 일반 alt (비어 있지 않을 때)')
        stages.append(('empty_alt', '1단계 · 빈 alt 전용 (위 일반 판정 대신 호출)'))
    if config.get('recheck_appropriate'):
        condition = '빈 alt 전용 제외, 최초 적절일 때만 호출' if config.get('skip_empty_alt_review') else '최초 적절일 때만 호출'
        stages.append(('recheck', '근거 재검토 · 일반 alt에서 needs_review=true일 때 호출'
                       if evidence_review else f'적절 재검토 · 적절 기준 + 부적절 6유형 ({condition})'))
        if config.get('compare_review_role'):
            stages[-1] = ('recheck', '적절 재검토 A · 역할·관찰 정보 미포함')
            stages.append(('recheck_role', '적절 재검토 B · 역할·관찰 정보 포함'))
    if config.get('classification_mode') in ('two_stage_verdict_types_v1',
                                              'guideline_verdict_candidates_final_v1'):
        stages.append(('types', '2단계 · 오류 유형 (재검토를 반영한 판정이 부적절일 때만 호출)'))
    if config.get('type_reflection'):
        frozen = config.get('reflection_mode', REFLECTION_TRAIN) == REFLECTION_FROZEN
        stages.extend([
            ('type_reflection_feedback', '후보 Reflection · 고정 MD 기반 피드백 제공/통과 선택'
             if frozen else '후보 Reflection · 정답 유형 미포함 문제만 피드백 생성'),
            ('type_reflection_fallback', '후보 Reflection · 피드백 제공 문제만 후보 재생성'),
        ])
        if not frozen:
            stages.extend([
                ('type_reflection_document', '후보 Reflection MD · 일반 갱신'),
                ('type_reflection_document_compact', '후보 Reflection MD · 중복 정리·갱신'),
            ])
    if (config.get('classification_mode') == 'guideline_verdict_candidates_final_v1'
            and not config.get('candidate_only')):
        stages.append(('final', '4단계 · 가이드라인 최종 오류 유형 선택'))
    if config.get('final_reflection'):
        frozen = config.get('reflection_mode', REFLECTION_TRAIN) == REFLECTION_FROZEN
        stages.extend([
            ('final_reflection_feedback', '최종 선택 Reflection · 고정 MD 기반 피드백 제공/통과 선택'
             if frozen else '최종 선택 Reflection · 최초 선택 오답만 피드백 생성'),
            ('final_reflection_fallback', '최종 선택 Reflection · 같은 후보 안에서 재선택'),
        ])
        if not frozen:
            stages.extend([
                ('final_reflection_document', '최종 선택 Reflection MD · 일반 갱신'),
                ('final_reflection_document_compact', '최종 선택 Reflection MD · 중복 정리·갱신'),
            ])
    if config.get('guideline_mode'):
        offset = 1 if config.get('recheck_appropriate') else 0
        stages = [(stage, {'classify': '1단계 · 가이드라인 적절/부적절',
                           'recheck': '2단계 · 별도 세션 독립 재검토 (최초 적절일 때만 호출)',
                           'types': f'{2 + offset}단계 · 오류 유형 복수 선택 (부적절일 때만 호출)',
                           'final': f'{3 + offset}단계 · 후보 중 최종 오류 유형 선택'}.get(stage, title))
                  for stage, title in stages]
    if config.get('label_reflection'):
        frozen = config.get('reflection_mode', REFLECTION_TRAIN) == REFLECTION_FROZEN
        frozen_v9 = frozen and bool(config.get('frozen_evidence_pipeline'))
        reflection_titles = {
            'classify': '1단계 · 최초 적절/부적절 판정',
            'reflection_visual_evidence': '2단계 · alt 비공개 시각 근거 추출',
            'reflection_coverage_audit': '3단계 · 시각 근거와 alt·문맥 대조',
            'reflection_feedback': (('4단계' if frozen_v9 else '2단계') +
                                    ' · 고정 MD 기반 피드백 제공/통과 선택' if frozen
                                    else '2단계 · 최초 이진 오답만 Reflection 피드백 생성'),
            'reflection_fallback': (('5단계' if frozen_v9 else '3단계') +
                                    ' · 피드백 제공일 때만 이진 재판정' if frozen
                                    else '3단계 · 피드백 생성 문제만 이진 재판정'),
            'reflection_document': '4A단계 · 기존 MD + 새 피드백 최소 수정 (5,000자 이하)',
            'reflection_document_compact': '4B단계 · 기존 MD 중복 정리 + 새 피드백 통합 (5,000자 초과)',
            'types': (('6단계' if frozen_v9 else '4단계') +
                      ' · 오류 유형 복수 선택 (최종 이진판정이 부적절일 때)' if frozen
                      else '5단계 · 오류 유형 복수 선택 (재판정이 부적절일 때)'),
            'final': (('7단계' if frozen_v9 else '5단계') if frozen else '6단계') +
                     ' · 후보 중 최종 오류 유형 선택',
        }
        stages = [(stage, reflection_titles.get(stage, title)) for stage, title in stages]
    if config.get('binary_only'):
        stages = [(stage, title) for stage, title in stages if stage not in ('types', 'final')]
    sections = []
    for stage, title in stages:
        if config.get('role_observation_mode'):
            if stage == 'role':
                title = title.replace('사전 단계', '1단계', 1)
            elif stage in ('classify', 'empty_alt'):
                title = title.replace('1단계', '2단계', 1)
            elif stage in ('recheck', 'recheck_role'):
                title = '3단계 · ' + title
            elif stage == 'types':
                title = title.replace('2단계', '4단계', 1)
        prompt_stage = 'recheck' if stage == 'recheck_role' else stage
        if not prompts.get(prompt_stage):
            continue
        system = prompts[prompt_stage]
        if stage == 'classify' and config.get('role_stage'):
            system = system.replace('{{role}}', '{image_role}').replace('{{criterion}}', '{role_criterion}')
        fields = ['candidates'] if stage == 'clean' else [
            key for key in METADATA_FIELDS
            if stage not in ('role', 'decorative_router', 'reflection_visual_evidence')
            or key != 'alt_text']
        if stage == 'reflection_coverage_audit':
            fields.append('initial_assessment')
        elif stage == 'reflection_feedback':
            fields += ['initial_assessment', 'current_guidelines']
            if config.get('reflection_mode', REFLECTION_TRAIN) == REFLECTION_TRAIN:
                fields.append('actual_label')
        elif stage == 'reflection_fallback':
            fields += ['initial_assessment', 'reflection_feedback']
            if config.get('reflection_mode', REFLECTION_TRAIN) == REFLECTION_FROZEN:
                fields.append('current_guidelines')
        elif stage in ('reflection_document', 'reflection_document_compact'):
            fields = ['current_guidelines', 'new_feedback']
        elif stage == 'type_reflection_feedback':
            fields += ['binary_assessment', 'initial_candidate_assessment', 'current_guidelines']
            if config.get('reflection_mode', REFLECTION_TRAIN) == REFLECTION_TRAIN:
                fields.append('actual_label')
        elif stage == 'type_reflection_fallback':
            fields += ['binary_assessment', 'initial_candidate_assessment', 'reflection_feedback']
        elif stage in ('type_reflection_document', 'type_reflection_document_compact'):
            fields = ['current_guidelines', 'new_feedback']
        elif stage == 'final_reflection_feedback':
            fields += ['binary_assessment', 'candidate_assessment',
                       'initial_final_assessment', 'current_guidelines']
            if config.get('reflection_mode', REFLECTION_TRAIN) == REFLECTION_TRAIN:
                fields.append('actual_label')
        elif stage == 'final_reflection_fallback':
            fields += ['binary_assessment', 'candidate_assessment',
                       'initial_final_assessment', 'reflection_feedback']
        elif stage in ('final_reflection_document', 'final_reflection_document_compact'):
            fields = ['current_guidelines', 'new_feedback']
        if stage == 'recheck' and config.get('guideline_mode'):
            fields.append('initial_assessment')
        if stage in ('types', 'final') and config.get('guideline_mode'):
            if config.get('error_classification_input') != 'image_metadata_candidates_v1':
                fields.append('binary_assessment')
            if stage == 'final':
                fields.append('candidate_assessment')
        if config.get('evidence_stage') and stage != 'role' and (stage != 'recheck' or evidence_review):
            fields.append('visual_evidence')
            if stage == 'recheck' and evidence_review:
                fields.append('initial_assessment')
        def user_json(keys):
            return '{\n' + ',\n'.join('  "' + key + '": {' +
                ('ocr_candidates' if key == 'candidates' else key) + '}' for key in keys) + '\n}'
        section = f'[{title}]\n\n[System]\n{system.rstrip()}\n\n'
        if stage not in ('reflection_document', 'reflection_document_compact',
                         'type_reflection_document', 'type_reflection_document_compact',
                         'final_reflection_document', 'final_reflection_document_compact'):
            section += '[User · 이미지]\n{image}\n\n'
        if stage in ('classify', 'empty_alt', 'types', 'recheck') and config.get('input_mode') != 'image_metadata':
            section += '[User · 텍스트 JSON · OCR 사용]\n' + user_json(fields + ['cleaned_ocr_text'])
            section += '\n\n[User · 텍스트 JSON · OCR 미사용]\n' + user_json(fields)
        else:
            section += '[User · 텍스트 JSON]\n' + user_json(fields)
        sections.append(section)
    return ('\n\n' + '─' * 40 + '\n\n').join(sections)


def _prompt_details(config: dict, *, saved: bool = False) -> None:
    title = '이 실행의 프롬프트 · 레이블 설명' if saved else '새 실행에 사용할 프롬프트 · 레이블 설명'
    with st.expander(title):
        prompts = config.get('prompts') or {}
        if not prompts:
            st.info('저장된 프롬프트가 없습니다. 이 실행에 사용한 설명을 확인할 수 없습니다.')
            return
        st.caption('선택한 실행의 manifest.json에 저장된 지시문으로 만든 입력 템플릿입니다.' if saved
                   else '새 실행의 전체 입력 템플릿입니다. 아래 단계들은 각각 별도의 모델 호출입니다.')
        st.caption(f"프롬프트 버전: {config.get('prompt_version', '미기록')} · "
                   f"모델: {config.get('model_label', Path(config.get('model_path', '미기록')).name)} · "
                   f"최대 생성 토큰: {config.get('max_tokens', '미기록')} · "
                   f"슬롯당 문맥 토큰: {config.get('context_size', '미기록')} · "
                   f"병렬 슬롯: {config.get('parallel_slots', config.get('session_slots', 1))}")
        st.caption('{image}는 별도 첨부하는 이미지입니다. User JSON의 {필드명}은 실제 데이터가 들어가는 자리이며, '
                   '문자열은 JSON 문자열, in_link·kogl_marker는 true/false 또는 null 등 실제 값의 형식으로 들어갑니다. '
                   'User JSON에는 선택한 단계의 입력만 보냅니다.')
        if config.get('input_mode') != 'image_metadata':
            st.caption('{ocr_candidates}는 [{"id": 1, "block_id": 2, "text": "OCR 원문"}, …] 형식의 후보 목록입니다.')
        if config.get('guideline_mode'):
            stages = ('이진 판정' if config.get('binary_only') else
                      '이진 판정·후보 생성' if config.get('candidate_only') else
                      '이진 판정·후보 생성·최종 선택' if not config.get('recheck_appropriate')
                      else '이진 판정·재검토·후보 생성·최종 선택')
            detail = ('오류 유형 후보와 최종 7범주 선택은 실행하지 않습니다.'
                      if config.get('binary_only') else
                      '후보 Reflection 결과를 저장한 뒤 종료하며 최종 1개 오류 유형은 선택하지 않습니다.'
                      if config.get('candidate_only') else
                      '후보 생성은 원본 입력만, 최종 선택은 원본 입력과 후보 집합만 사용합니다. 이진 Reflection의 피드백·MD·재판정 JSON은 5·6단계에 전달하지 않습니다.'
                      if config.get('error_classification_input') == 'image_metadata_candidates_v1' else
                      '후보 생성은 이진판정을, 최종 선택은 이진판정과 후보 집합을 입력으로 받습니다.')
            st.caption(f'공식 가이드라인 v1.1의 판정 관련 1~3절과 판정 범위 5절을 {stages}에 포함합니다. {detail}')
        if config.get('label_reflection'):
            if config.get('reflection_mode', REFLECTION_TRAIN) == REFLECTION_FROZEN:
                st.caption('Frozen MD 추론: 원본 이미지·메타데이터·최초 판정과 공식 가이드라인·고정 MD를 대조해 피드백 제공 또는 통과를 선택하며 실제 레이블은 전달하지 않습니다. '
                           '장식오용형 게이트가 확정하지 않은 문제는 alt를 숨긴 시각 근거 추출과 alt·문맥 대조를 먼저 수행합니다. '
                           + ('통과이면 최초 판정을 최종 이진판정으로 저장합니다. ' if config.get('binary_only')
                              else '통과이면 최초 적절은 종료하고 최초 부적절은 유형 분류로 이동합니다. ')
                           + '피드백 제공일 때만 동일한 이미지·메타데이터와 최초 판정 JSON·피드백으로 재판정합니다. MD는 수정하지 않습니다. '
                           '실제 레이블은 모델 추론이 모두 끝난 뒤 성능 계산에만 사용합니다.')
            else:
                st.caption('Train Reflection: 피드백 생성에는 최초 판정·train 실제 레이블·현재 MD가 들어갑니다. '
                           '실제 라벨과 최초 판정의 이진 방향이 다른 문제만 피드백·재판정·MD 갱신을 수행합니다. '
                           '피드백 문장에는 최초 판정이 틀렸다는 사실이나 정답 방향을 직접 쓰지 않습니다. '
                           '재판정에는 동일한 이미지·메타데이터와 최초 판정 JSON·새 피드백을 전달하며, 실제 레이블과 MD는 전달하지 않습니다. '
                           '문서 수정에는 기존 MD와 새 피드백만 들어갑니다.')
            st.caption('빈 alt 링크에서 URL·경로형 또는 빈 link_dest는 결정적 구조 게이트를 유지합니다. '
                       '사람이 읽을 수 있는 link_dest가 있으면 v1.1 시각 예외 검사로 이미지에만 있는 실질 정보를 확인합니다. '
                       '비어 있지 않은 alt는 이미지·아이콘 접미사를 제거한 문구가 같은 링크 표시 문구에 포함되면 '
                       '최초 이진판정과 Reflection 없이 부적절로 확정합니다.')
        if config.get('decorative_gate'):
            st.caption('비어 있지 않은 alt는 최초 이진판정과 alt를 숨긴 장식 후보 라우터를 병렬 실행합니다. '
                       '라우터 통과 문제만 구조화 관찰하며, 검증된 결정 규칙이 확정한 경우 장식오용형으로 종료합니다. '
                       '확정하지 않은 문제는 기존 Reflection·후보·최종 선택 파이프라인을 그대로 진행합니다.')
        if config.get('empty_alt_prompt'):
            st.caption('빈 문자열·공백뿐인 alt는 전용 1차 프롬프트로 분기합니다. 일반 판정과 둘 중 하나만 호출합니다. ' +
                       ('빈 alt도 역할·관찰 결과를 입력으로 참고합니다. ' if config.get('evidence_stage')
                        else '빈 alt 판정에는 사전 역할을 강제하지 않습니다. ') +
                       ('빈 alt는 폴백을 생략하고, 부적절일 때만 유형을 분류합니다.' if config.get('skip_empty_alt_review')
                        else '이후 적절 재검토·유형 판정은 같은 설정을 유지합니다.'))
        separator = '\n\n' + '─' * 40 + '\n\n'
        for section in _prompt_template(config).split(separator):
            if not section.strip():
                continue
            with st.container(border=True):
                st.code(section.strip(), language='text', wrap_lines=True)
        if config.get('evidence_stage'):
            if config.get('role_observation_mode'):
                st.caption('{visual_evidence} = {"roles": ["기능성", "텍스트 포함 이미지"], "description": "이미지 설명"} 형식입니다. '
                           '역할은 기능성·정보성·텍스트 포함 이미지·장식성 중 복수 선택하며, 이 단계에는 alt가 없습니다. '
                           '역할·이미지 설명만 추출하고 alt 품질은 다음 단계에서 판정합니다. 빈 alt는 폴백을 생략합니다.')
            else:
                st.caption('{visual_evidence}에는 사전 단계의 JSON 전체가 들어갑니다. 역할은 참고값이며, '
                           '필요한 정보와 실제 문맥 문구를 대조합니다. 빈 alt는 폴백을 생략합니다.')
            if config.get('review_policy') == 'appropriate_v1':
                st.caption('일반 alt의 최초 적절은 needs_review 값과 관계없이 기존 적절 기준·6유형으로 재검토합니다. '
                           '이 재검토 입력은 원본 이미지와 메타데이터이며 최초 관찰·충족표는 포함하지 않습니다.')
                if config.get('compare_review_role'):
                    st.caption('비교 조건 B는 같은 재검토 지시문에 {visual_evidence}만 추가합니다. '
                               '역할·관찰 JSON 전체를 전달하며 역할을 다시 추론하지 않습니다. '
                               'A·B 모두 같은 1차 판정을 공유하고, 빈 alt와 최초 부적절은 두 폴백을 생략합니다.')
            else:
                st.caption('{initial_assessment}는 재검토에만 제공되는 최초 판정 JSON입니다. '
                           '재검토로 부적절→적절이 된 경우, 폴백 없음 비교에 필요한 최초 오류 유형도 별도로 저장합니다.')
        elif config.get('role_stage'):
            st.caption('{image_role}에는 사전 단계가 선택한 역할 하나, {role_criterion}에는 그 역할의 기준 한 문장이 들어갑니다. '
                       '아래 표는 치환 규칙이며, 표 전체를 모델에 보내지 않습니다.')
            st.table([{'image_role': role, 'role_criterion': criterion}
                      for role, criterion in config['role_criteria'].items()])
        if _is_priority_run(config) and not config.get('candidate_only'):
            priority = config.get('label_priority')
            reflection_step = '6단계 · '
            st.caption((reflection_step if config.get('label_reflection') else
                        '4단계 · ' if config.get('guideline_mode') else
                        '5단계 · ' if config.get('role_observation_mode') else '') +
                       ('비교용 코드 우선순위: ' if config.get('classification_mode') == 'guideline_verdict_candidates_final_v1'
                        else '코드 우선순위 (모델 프롬프트와 별도): ') +
                       (' > '.join(priority) if isinstance(priority, list) else '미기록'))


def _saved_requests(run_dir: Path, manifest: dict) -> None:
    with st.expander('실제 전송된 프롬프트 · 문제별'):
        if not manifest.get('record_ids'):
            st.info('선택된 문제가 없습니다.')
            return
        rid = st.selectbox('프롬프트를 확인할 문제 ID', manifest['record_ids'], key=f'prompt_record:{run_dir}')
        st.caption('저장된 요청에서 System·User 텍스트를 읽습니다. 이미지 바이트는 파일 경로와 SHA256으로 기록되어 있습니다.')
        paths = [(f'OCR 정제 · {path.stem}', path)
                 for path in sorted((run_dir / 'raw/cleaned').glob(f'{rid}-*.request.json'))]
        direct = manifest.get('pipeline_mode') == 'gemma_only'
        evidence = (manifest.get('gemma') or {}).get('evidence_stage')
        evidence_review = evidence and (manifest.get('gemma') or {}).get('review_policy') != 'appropriate_v1'
        paths.append(('사전 관찰 근거' if evidence else '사전 역할 판정' if direct else '사전 역할 판정 · OCR 조건 공통', run_dir / 'raw/gemma_role' / f'{rid}.request.json'))
        conditions = [('gemma_no_ocr', 'Gemma')] if direct else [('gemma', '정제 OCR 사용'), ('gemma_no_ocr', 'OCR 미사용')]
        for model, condition in conditions:
            paths.extend([(f'링크 구조 게이트 · {condition}', run_dir / 'raw' / model / f'{rid}.gate.request.json'),
                          (f'장식 후보 라우터 · {condition}', run_dir / 'raw' / model / f'{rid}.decorative_router.request.json'),
                          (f'장식 구조화 관찰 · {condition}', run_dir / 'raw' / model / f'{rid}.decorative_observer.request.json'),
                          (f'품질 판정 · {condition}', run_dir / 'raw' / model / f'{rid}.request.json'),
                          (f'Reflection 피드백 생성 · {condition}', run_dir / 'raw' / model / f'{rid}.reflection_feedback.request.json'),
                          (f'피드백 기반 이진 재판정 · {condition}', run_dir / 'raw' / model / f'{rid}.reflection_fallback.request.json'),
                          (f'Reflection MD 수정 · {condition}', run_dir / 'raw' / model / f'{rid}.reflection_document.request.json'),
                          (f'{"근거 재검토" if evidence_review else "적절 재검토"} · {condition}', run_dir / 'raw' / model / f'{rid}.recheck.request.json'),
                          (f'오류 유형 판정 · {condition}', run_dir / 'raw' / model / f'{rid}.types.request.json'),
                          (f'오류 유형 후보 Reflection 피드백 · {condition}', run_dir / 'raw' / model / f'{rid}.type_reflection_feedback.request.json'),
                          (f'오류 유형 후보 재생성 · {condition}', run_dir / 'raw' / model / f'{rid}.type_reflection_fallback.request.json'),
                          (f'오류 유형 후보 Reflection MD 수정 · {condition}', run_dir / 'raw' / model / f'{rid}.type_reflection_document.request.json'),
                          (f'최종 오류 유형 선택 · {condition}', run_dir / 'raw' / model / f'{rid}.final.request.json'),
                          (f'최종 선택 Reflection 피드백 · {condition}', run_dir / 'raw' / model / f'{rid}.final_reflection_feedback.request.json'),
                          (f'최종 선택 Reflection 재선택 · {condition}', run_dir / 'raw' / model / f'{rid}.final_reflection_fallback.request.json'),
                          (f'최종 선택 Reflection MD 수정 · {condition}', run_dir / 'raw' / model / f'{rid}.final_reflection_document.request.json'),
                          (f'폴백 없음 비교용 최초 유형 · {condition}', run_dir / 'raw' / model / f'{rid}.initial_types.request.json')])
            if (manifest.get('gemma') or {}).get('compare_review_role'):
                paths.append((f'적절 재검토 B · 역할·관찰 정보 포함 · {condition}',
                              run_dir / 'raw' / model / f'{rid}.recheck_role.request.json'))
        audits = []
        for title, path in paths:
            try:
                audit = read_json(path)
                if audit is None:
                    continue
                if audit.get('decision_source'):
                    text = json.dumps(audit, ensure_ascii=False, indent=2)
                else:
                    system, user = audit['messages']
                    content = user['content']
                    data = content[-1]
                    text = f"[System]\n{system['content']}\n\n"
                    if len(content) == 2:
                        text += (f"[User · 이미지]\n{json.dumps(content[0], ensure_ascii=False, indent=2)}\n\n")
                    text += f"[User · 텍스트 JSON]\n{data['text']}"
            except (ValueError, KeyError, TypeError):
                st.warning(f'{title}: 요청 기록을 읽을 수 없습니다.')
                continue
            st.markdown(f'**{title}**')
            st.code(text, language='text', wrap_lines=True)
            audits.append({'단계': title, '요청': audit})
        if audits:
            with st.expander('요청 JSON · 생성 설정 · 출력 스키마'):
                st.json(audits, expanded=False)
        else:
            st.info('이 문제에는 아직 저장된 요청이 없습니다. 현재 템플릿을 실제 요청처럼 대신 표시하지 않습니다.')


def _new_run(dataset_dir: Path, records: list[dict], preset: str) -> None:
    st.subheader("새 실행")
    special = preset in (PUBLIC_109_PRESET, COMBINED_FROZEN_PRESET)
    selected = [_record_id(row) for row in records] if special else _selection(records)
    labeled = all(row.get('label') in LABELS for row in records)
    seed_train = preset == PUBLIC_109_PRESET
    if seed_train:
        reflection_mode = REFLECTION_TRAIN
        st.info('선택한 기존 누적 MD를 초기값으로 복사해 공개 Test 109건만 이어서 Train합니다.')
    elif preset == COMBINED_FROZEN_PRESET:
        reflection_mode = REFLECTION_FROZEN
        st.info('한 실행에서 Train 2,400건 다음 공개 Test 109건을 처리하고 성능은 분리해 표시합니다.')
    elif labeled:
        reflection_mode = st.radio(
            'Reflection 실행 모드', tuple(REFLECTION_MODE_LABELS), horizontal=True,
            format_func=REFLECTION_MODE_LABELS.get,
        )
    else:
        reflection_mode = REFLECTION_FROZEN
        st.info('정답이 없는 Test 데이터입니다. Train에서 만든 MD를 사용하는 Frozen MD 추론만 실행합니다.')
    scopes = (('전체 파이프라인', '후보풀까지만 실행', '이진분류만 실행')
              if reflection_mode == REFLECTION_FROZEN
              else ('전체 파이프라인', '이진분류만 실행'))
    execution_scope = st.radio(
        '실행 범위', scopes, horizontal=True,
        help='후보풀까지만 실행하면 Frozen MD 후보 Reflection 뒤에 종료합니다. 이진분류만 실행하면 오류 유형 후보 생성도 호출하지 않습니다.',
    )
    binary_only = execution_scope == '이진분류만 실행'
    candidate_only = execution_scope == '후보풀까지만 실행'
    source_run = None
    source_document = None
    source_sha = None
    source_type_document = None
    source_type_sha = None
    source_final_document = None
    source_final_sha = None
    if reflection_mode == REFLECTION_FROZEN or seed_train:
        sources = _reflection_sources(
            require_type=not binary_only,
            require_final=not binary_only and not candidate_only)
        if sources:
            source_run = st.selectbox(
                '이어갈 Reflection MD' if seed_train else '학습된 Reflection MD', sources,
                format_func=lambda path: f'{path.name} · {(get_status(path) or {}).get("state", "미기록")}',
            )
            source_document = (source_run / REFLECTION_FILENAME).read_text(encoding='utf-8')
            source_sha = hashlib.sha256(source_document.encode()).hexdigest()
            if not binary_only:
                source_type_document = (source_run / TYPE_REFLECTION_FILENAME).read_text(encoding='utf-8')
                source_type_sha = hashlib.sha256(source_type_document.encode()).hexdigest()
            if not binary_only and not candidate_only:
                source_final_document = (source_run / FINAL_REFLECTION_FILENAME).read_text(encoding='utf-8')
                source_final_sha = hashlib.sha256(source_final_document.encode()).hexdigest()
            st.caption(f'출처 실행: {source_run.name} · 이진 MD SHA-256: {source_sha}'
                       + (f' · 후보 MD SHA-256: {source_type_sha}' if source_type_sha else '')
                       + (f' · 최종 선택 MD SHA-256: {source_final_sha}' if source_final_sha else '')
                       + (' · 복사 후 공개 109건으로 계속 갱신합니다.' if seed_train
                          else ' · 새 실행에서는 수정하지 않습니다.'))
        else:
            st.warning('사용할 Train Reflection MD가 없습니다. 먼저 Train 모드에서 가이드라인을 생성하세요.')
    st.caption('링크 구조 게이트·빈 alt v1.1 시각 예외 검사 / Gemma 최초 이진판정'
               + (' → V9 시각 근거·alt 대조' if reflection_mode == REFLECTION_FROZEN else '')
               + ' → Reflection 피드백·보수적 재판정'
               + ('' if binary_only else ' → 오류 유형 후보'
                  + ('' if candidate_only else '·최종 선택')))
    st.caption(('Train 모드는 최초 이진 오답에만 피드백을 만들고 MD를 갱신합니다. '
                if labeled else '')
               + 'Frozen MD 모드는 실제 레이블 없이 선택한 MD를 고정 사용합니다.')
    st.caption('공식 가이드라인 v1.1의 1~3절·5절 사용 · 빈 alt 링크 시각 예외·같은 링크 alt 중복 분기 · 흰 전경은 밝은 회색 배경')
    model_variant = st.selectbox('Gemma 모델', tuple(MODEL_VARIANTS), index=0,
                                 format_func=lambda key: MODEL_VARIANTS[key]['label'])
    device = st.selectbox('GPU', tuple(f'gpu:{i}' for i in range(8)), index=0,
                          help='31B·24K 문맥·2슬롯 실측 VRAM은 약 24.6GB입니다. 한 GPU 안에서 두 요청을 병렬 처리합니다.')
    config = None
    if (reflection_mode == REFLECTION_TRAIN and not seed_train) or source_run is not None:
        frozen = reflection_mode == REFLECTION_FROZEN
        config = default_config(
            4096, 24576, binary_prompt='guideline_direct', model_variant=model_variant,
            label_reflection=True, reflection_mode=reflection_mode,
            reflection_guideline_source_run=source_run.name if frozen else None,
            reflection_guideline_sha256=source_sha if frozen else None,
            binary_only=binary_only,
            candidate_only=candidate_only,
            type_reflection=not binary_only,
            type_reflection_guideline_sha256=source_type_sha if frozen else None,
            final_reflection=not binary_only and not candidate_only,
            final_reflection_guideline_sha256=source_final_sha if frozen else None,
            decorative_gate=not binary_only and not candidate_only,
        )
        _prompt_details(config)
    st.caption(f'선택됨: {len(selected):,} / 전체: {len(records):,}')
    if st.button('실행', type='primary', disabled=not selected or config is None):
        try:
            evaluation_groups = None
            if preset == COMBINED_FROZEN_PRESET:
                evaluation_groups = {
                    'train_2400': {
                        'label': 'Train 2,400',
                        'record_ids': [row['record_id'] for row in records
                                       if row.get('_evaluation_group') == 'train_2400'],
                    },
                    'public_test_109': {
                        'label': '공개 Test 109',
                        'record_ids': [row['record_id'] for row in records
                                       if row.get('_evaluation_group') == 'public_test_109'],
                    },
                }
            run_dir = create_run(dataset_dir, selected, [], 'auto_gray', device,
                                 gemma_config=config, pipeline_mode='gemma_only', records=records,
                                 evaluation_groups=evaluation_groups)
            if source_run is not None:
                write_document(run_dir, source_document)
                if source_type_document is not None:
                    write_type_document(run_dir, source_type_document)
                if source_final_document is not None:
                    write_final_document(run_dir, source_final_document)
            if seed_train:
                manifest = load_manifest(run_dir)
                manifest.update(seed_reflection_run=source_run.name,
                                seed_reflection_sha256=source_sha,
                                seed_type_reflection_sha256=source_type_sha,
                                seed_final_reflection_sha256=source_final_sha)
                write_json(run_dir / 'manifest.json', manifest)
            launch_run(run_dir)
            st.session_state['active_run'] = str(run_dir)
            st.session_state['run_selector'] = str(run_dir)
            st.success(f'작업을 시작했습니다: {Path(run_dir).name}')
        except Exception as exc:
            st.error(f'작업을 시작할 수 없습니다: {exc}')


def _active_run(run_dir: Path, refresh_was_active: bool = False) -> None:
    if run_dir.name == 'reviewed_guidelines':
        st.info('최종 선정 가이드라인 사본입니다. 새 실행에서 Frozen MD를 선택하면 사용합니다.')
        for filename in (REFLECTION_FILENAME, TYPE_REFLECTION_FILENAME, FINAL_REFLECTION_FILENAME):
            with st.expander(filename):
                st.markdown((run_dir / filename).read_text(encoding='utf-8'))
        return
    status = get_status(run_dir) or {}
    if refresh_was_active and status.get('state') not in {'running', 'stopping'}:
        # Rebuild the page once without a timer when a live run finishes.
        st.rerun()
    manifest = load_manifest(run_dir) or {}
    direct = manifest.get('pipeline_mode') == 'gemma_only'
    live = status.get('state') in {'running', 'stopping'}
    # Full request-by-request integrity auditing can take minutes on a large
    # stopped run. The dashboard only needs saved prediction metrics; the
    # scoring/export path keeps the complete audit.
    report = build_benchmark(run_dir, audit_requests=False) if manifest.get('gemma') else None
    if report is not None:
        variants = ['without_ocr'] if direct else (['with_ocr', 'without_ocr'] if manifest.get('compare_without_ocr') else ['with_ocr'])
        problem_states = [[row[variant + '_status'] for variant in variants] for row in report['predictions']]
    else:
        problem_states = [[(load_result(run_dir, model, rid) or {}).get('status', 'pending')
                           for model in manifest['models']] for rid in manifest['record_ids']]
    finished = [states for states in problem_states if states and all(s in ('ok', 'error') for s in states)]
    completed, total = len(finished), len(manifest['record_ids'])
    failed = sum('error' in states for states in finished)
    st.progress(completed / max(1, total), text=f"문제 처리 완료 {completed:,} / {total:,}개")
    st.caption(f'성공 {completed - failed:,}개 · 실패 {failed:,}개 · 미완료 {total - completed:,}개')
    if status.get('execution_mode') == 'record_parallel_v1':
        st.caption(f"현재 단계: {status.get('phase', '미기록')} · 문제 병렬 슬롯 "
                   f"{status.get('parallel_slots', 2)}개")
    elapsed = int(run_elapsed_seconds(run_dir, status))
    hours, rest = divmod(elapsed, 3600)
    minutes, seconds = divmod(rest, 60)
    timing = read_json(run_dir / 'timing.json', {})
    estimated = timing.get('estimated', not bool(timing)) and elapsed > 0
    st.metric('총 경과 시간', ('약 ' if estimated else '') + f'{hours:02d}:{minutes:02d}:{seconds:02d}',
              help='시:분:초 · 모델 로딩부터 판정 완료까지, 이어서 실행한 시간도 합산합니다. '
                   '새 시간 기록은 중지 기간을 제외합니다. 기존 실행의 추정 시간에는 대기·중지 기간이 포함될 수 있습니다.')
    st.caption(f"상태: {status.get('state', '미기록')}")
    if status.get('error'):
        st.error(status['error'])
    if report is not None:
        st.caption('OCR 사용·미사용 판정이 모두 끝나면 문제 1개로 집계합니다.' if manifest.get('compare_without_ocr')
                   else 'Gemma 최종 판정이 끝나면 문제 1개로 집계합니다.')
        if report.get('request_audit') == 'skipped_for_live_dashboard':
            st.caption('대시보드 빠른 집계 · 저장 요청의 무결성 감사는 성능 평가·내보내기에서 수행합니다.')
    if status.get('pipeline_mode') == 'stream':
        ocr = read_json(run_dir / 'ocr_status.json', {})
        if ocr:
            st.caption(f"Paddle OCR: {ocr.get('completed', 0):,} / {ocr.get('total', 0):,} · "
                       f"{ocr.get('record_id') or ocr.get('phase') or ocr.get('state', '')}")
    cache_status = read_json(run_dir / 'ocr_status.json', status)
    if cache_status.get('cache_reused') is not None:
        st.caption(f"공유 OCR 재사용: {cache_status['cache_reused']:,}개 · 재추출 없이 저장된 결과 사용")
    first, second, third = st.columns(3)
    with first:
        if st.button("진행 상황 새로고침"):
            st.rerun()
    with second:
        if st.button("중지 요청", disabled=status.get("state") not in {"running", "stopping"}):
            request_stop(run_dir)
            st.warning("중지 요청을 기록했습니다. 진행 중인 요청이 끝나면 중지하며 저장된 결과는 유지됩니다.")
    with third:
        if st.button("이어서 실행", disabled=not direct or status.get("state") in {"running", "stopping"}):
            try:
                launch_run(run_dir)
                st.rerun()
            except Exception as exc:
                st.error(f"이어서 실행할 수 없습니다: {exc}")
    if not direct:
        st.caption('과거 실험은 결과 열람만 지원합니다. 변경된 파이프라인은 새 실행으로 시작하세요.')
    if manifest.get('gemma'):
        gemma_config = manifest['gemma']
        priority_run = _is_priority_run(gemma_config)
        _prompt_details(gemma_config, saved=True)
        _saved_requests(run_dir, manifest)
        reflection_document = run_dir / 'reflection_guidelines.md'
        if gemma_config.get('label_reflection'):
            frozen = gemma_config.get('reflection_mode', REFLECTION_TRAIN) == REFLECTION_FROZEN
            title = ('고정 Reflection 판단 가이드라인 · MD' if frozen
                     else '누적 Reflection 판단 가이드라인 · MD')
            with st.expander(title, expanded=False):
                if reflection_document.is_file():
                    markdown = reflection_document.read_text(encoding='utf-8')
                    if frozen:
                        st.caption(f"출처 실행: {gemma_config.get('reflection_guideline_source_run', '미기록')} · "
                                   f"SHA-256: {gemma_config.get('reflection_guideline_sha256', '미기록')} · 읽기 전용 스냅샷")
                    elif manifest.get('seed_reflection_run'):
                        st.caption(f"이어받은 출처 실행: {manifest['seed_reflection_run']} · "
                                   f"초기 SHA-256: {manifest.get('seed_reflection_sha256', '미기록')} · 이후 계속 갱신")
                    st.markdown(markdown)
                    st.download_button('Reflection 가이드라인 다운로드', markdown,
                                       file_name='reflection_guidelines.md', mime='text/markdown')
                else:
                    st.info('고정 Reflection MD 스냅샷이 없습니다.' if frozen
                            else '첫 Reflection 문서 수정을 기다리는 중입니다.')
        type_reflection_document = run_dir / TYPE_REFLECTION_FILENAME
        if gemma_config.get('type_reflection'):
            frozen = gemma_config.get('reflection_mode', REFLECTION_TRAIN) == REFLECTION_FROZEN
            title = ('고정 오류 유형 후보 Reflection 가이드라인 · MD' if frozen
                     else '누적 오류 유형 후보 Reflection 가이드라인 · MD')
            with st.expander(title, expanded=False):
                if type_reflection_document.is_file():
                    markdown = type_reflection_document.read_text(encoding='utf-8')
                    if frozen:
                        st.caption(f"출처 실행: {gemma_config.get('reflection_guideline_source_run', '미기록')} · "
                                   f"SHA-256: {gemma_config.get('type_reflection_guideline_sha256', '미기록')} · 읽기 전용 스냅샷")
                    st.markdown(markdown)
                    st.download_button('오류 유형 후보 Reflection 가이드라인 다운로드', markdown,
                                       file_name=TYPE_REFLECTION_FILENAME, mime='text/markdown')
                else:
                    st.info('고정 후보 Reflection MD 스냅샷이 없습니다.' if frozen
                            else '첫 후보 Reflection 문서 수정을 기다리는 중입니다.')
        final_reflection_document = run_dir / FINAL_REFLECTION_FILENAME
        if gemma_config.get('final_reflection'):
            frozen = gemma_config.get('reflection_mode', REFLECTION_TRAIN) == REFLECTION_FROZEN
            title = ('고정 최종 선택 Reflection 가이드라인 · MD' if frozen
                     else '누적 최종 선택 Reflection 가이드라인 · MD')
            with st.expander(title, expanded=False):
                if final_reflection_document.is_file():
                    markdown = final_reflection_document.read_text(encoding='utf-8')
                    if frozen:
                        st.caption(f"출처 실행: {gemma_config.get('reflection_guideline_source_run', '미기록')} · "
                                   f"SHA-256: {gemma_config.get('final_reflection_guideline_sha256', '미기록')} · 읽기 전용 스냅샷")
                    st.markdown(markdown)
                    st.download_button('최종 선택 Reflection 가이드라인 다운로드', markdown,
                                       file_name=FINAL_REFLECTION_FILENAME, mime='text/markdown')
                else:
                    st.info('고정 최종 선택 Reflection MD 스냅샷이 없습니다.' if frozen
                            else '첫 최종 선택 Reflection 문서 수정을 기다리는 중입니다.')
        comparison = run_dir / 'prompt_comparison.md'
        if manifest.get('prompt_comparison') and comparison.is_file():
            with st.expander(manifest['prompt_comparison'].get('title', 'v7 · v8 동일 문제 비교'), expanded=False):
                st.markdown(comparison.read_text(encoding='utf-8'))
                st.download_button('프롬프트 비교 보고서', comparison.read_bytes(),
                                   file_name='prompt_comparison.md', mime='text/markdown')
        rows = {row['record_id']: row for row in read_json(run_dir / 'records.json', [])}
        labeled = all(rows.get(rid, {}).get('label') in LABELS for rid in manifest['record_ids'])
        if not labeled:
            _show_test_predictions(run_dir, Path(manifest['dataset_dir']))
            return
        groups = manifest.get('evaluation_groups')
        if groups:
            st.subheader('Frozen MD · 데이터셋별 성능')
            st.caption('두 데이터셋은 한 번에 연속 추론하지만 점수는 합치지 않습니다. 공개 Test 109는 공식 공개 라벨로 '
                       '학습에도 사용할 수 있는 표본이므로 일반화 점수가 아니라 재확인 점수입니다.')
            group_reports = {
                key: build_benchmark(
                    run_dir, audit_requests=False, record_ids=group['record_ids'],
                    evaluation_name=group['label'])
                for key, group in groups.items()
            }
            summary = []
            for key, group in groups.items():
                group_report = group_reports[key]
                metrics = group_report['paired'].get('without_ocr')
                binary = (group_report.get('binary_stage_comparison') or {}).get('paired', {}).get('without_ocr')
                coverage = group_report['coverage']['without_ocr']
                summary.append({
                    '데이터셋': group['label'],
                    '완료/전체': f"{coverage['labeled_ok']:,}/{group_report['selected_count']:,}",
                    '7범주 macro-F1': metrics['macro_f1'] if metrics else None,
                    '후보 전 이진 F1': binary['pre_candidate']['binary_f1'] if binary else None,
                    '최종 라벨 환산 이진 F1': binary['final_label']['binary_f1'] if binary else None,
                    '결합 점수': metrics['combined_score'] if metrics else None,
                    '정확도': metrics['accuracy'] if metrics else None,
                })
            st.dataframe(summary, hide_index=True)
            for key, group in groups.items():
                with st.expander(f"{group['label']} · 단계별 상세 성능"):
                    show_benchmark(run_dir, group_reports[key])
        else:
            show_benchmark(run_dir, report)
        predictions = []
        for rid in manifest['record_ids']:
            saved = load_result(run_dir, 'gemma', rid)
            baseline = load_result(run_dir, 'gemma_no_ocr', rid)
            if direct:
                saved = baseline
            if saved and saved.get('status') == 'ok':
                truth = rows.get(rid, {}).get('label')
                prediction = {'ID': rid, '실제 alt': rows.get(rid, {}).get('alt_text', ''),
                              '실제 레이블 (데이터 정답)': truth or '미공개'}
                if gemma_config.get('binary_only'):
                    binary = saved.get('binary_decision') or {}
                    reflected = saved.get('reflection_decision') or {}
                    truth_binary = ('적절' if truth == '적절' else '부적절') if truth else '미공개'
                    prediction.update({
                        '실제 이진 정답': truth_binary,
                        '최초 이진판정': binary.get('verdict', ''),
                        '최종 이진판정': saved.get('verdict', ''),
                        '최종 정답 여부': (saved.get('verdict') == truth_binary) if truth else None,
                        'Reflection 피드백': (saved.get('reflection_feedback') or {}).get('feedback', ''),
                        'Reflection 재판정': reflected.get('verdict', ''),
                        '판정 근거': saved.get('reason', ''),
                    })
                    predictions.append(prediction)
                    continue
                if gemma_config.get('candidate_only'):
                    binary = saved.get('binary_decision') or {}
                    reflected = saved.get('reflection_decision') or {}
                    candidates = [item.get('label') for item in saved.get('matched_types', [])
                                  if isinstance(item, dict)]
                    initial_candidates = [
                        item.get('label') for item in
                        (saved.get('initial_type_classification') or {}).get('matched_types', [])
                        if isinstance(item, dict)]
                    truth_binary = ('적절' if truth == '적절' else '부적절') if truth else '미공개'
                    prediction.update({
                        '실제 이진 정답': truth_binary,
                        '최초 이진판정': binary.get('verdict', ''),
                        '최종 이진판정': saved.get('verdict', ''),
                        '이진 정답 여부': (saved.get('verdict') == truth_binary) if truth else None,
                        'Reflection 재판정': reflected.get('verdict', ''),
                        '최초 오류 유형 후보': ', '.join(initial_candidates),
                        'Frozen Reflection 후 후보': ', '.join(candidates),
                        '후보에 실제 유형 포함': (truth in candidates) if truth in LABELS[1:] else None,
                        '후보 판정 근거': saved.get('reason', ''),
                    })
                    predictions.append(prediction)
                    continue
                if priority_run:
                    conditions = [('Gemma', saved)] if direct else [('정제 OCR 사용', saved), ('OCR 미사용', baseline)]
                    for condition, result in conditions:
                        ok = result and result.get('status') == 'ok'
                        verdict = '판정 실패' if result else '결과 없음'
                        if ok:
                            verdict = ('✅ 정답' if result['label'] == truth else '❌ 오답') if truth else '정답 미공개'
                        two_stage = gemma_config.get('classification_mode') in (
                            'two_stage_verdict_types_v1', 'guideline_verdict_candidates_final_v1')
                        binary_title = ('2단계 판정' if gemma_config.get('role_observation_mode') else '1단계 판정') if two_stage else 'Gemma 적절/부적절'
                        reflection_types_step = ('5단계 후보' if gemma_config.get('reflection_mode') == REFLECTION_FROZEN
                                                 else '5단계 후보')
                        types_title = (reflection_types_step if gemma_config.get('label_reflection') else
                                       '4단계 후보' if gemma_config.get('role_observation_mode') else
                                       '3단계 후보' if gemma_config.get('guideline_mode') else '2단계 후보') if two_stage else 'Gemma 복수 라벨'
                        prediction.update({
                            f'{binary_title} ({condition})': result.get('verdict', '기존 형식 · 미기록') if ok else '',
                            f'{types_title} ({condition})': _matched_type_summary(result, gemma_config) if ok else '',
                            f'{"가이드라인 최종 판정" if gemma_config.get("classification_mode") == "guideline_verdict_candidates_final_v1" else "기계적 최종 판정"} ({condition})': result['label'] if ok else '',
                            f'정답 여부 ({condition})': verdict,
                        })
                        if two_stage:
                            binary = result.get('binary_decision') or {} if ok else {}
                            prediction[f'{binary_title} ({condition})'] = binary.get('verdict', '')
                            if gemma_config.get('empty_alt_prompt'):
                                prediction[f'1차 판정 경로 ({condition})'] = (
                                    '빈 alt 전용' if result.get('binary_route') == 'empty_alt' else '일반 alt') if ok else ''
                            if gemma_config.get('recheck_appropriate'):
                                review = (result.get('review_decision') or {}) if ok else {}
                                skip = ('빈 alt 전용 · 폴백 생략' if ok and gemma_config.get('skip_empty_alt_review')
                                        and result.get('binary_route') == 'empty_alt' else '대상 아님')
                                prediction[f'판정 재검토 ({condition})'] = review.get('verdict', skip if ok else '')
                                prediction[f'재검토 근거 ({condition})'] = review.get('reason', '')
                                prediction[f'재검토 후 이진 판정 ({condition})'] = result.get('verdict', '') if ok else ''
                                prediction[f'폴백 없음 · 최종 라벨 ({condition})'] = (
                                    '적절' if binary.get('verdict') == '적절' else result.get('initial_label', result['label'])) if ok else ''
                                prediction[f'폴백 있음 · 최종 라벨 ({condition})'] = result['label'] if ok else ''
                                if gemma_config.get('compare_review_role'):
                                    role_review = (result.get('role_review_decision') or {}) if ok else {}
                                    prediction['폴백 역할 정보 미포함 · 이진'] = result.get('verdict', '') if ok else ''
                                    prediction['폴백 역할 정보 포함 · 이진'] = role_review.get('verdict', binary.get('verdict', ''))
                                    prediction['폴백 역할 정보 포함 · 근거'] = role_review.get('reason', skip if ok else '')
                            if gemma_config.get('label_reflection'):
                                reflected = result.get('reflection_decision') or {}
                                feedback = result.get('reflection_feedback') or {}
                                visual = result.get('reflection_visual_evidence') or {}
                                coverage = result.get('reflection_coverage_audit') or {}
                                prediction[f'V9 시각 근거 ({condition})'] = visual.get('reason', '')
                                prediction[f'V9 alt·문맥 대조 ({condition})'] = coverage.get('verdict', '')
                                prediction[f'V9 대조 근거 ({condition})'] = coverage.get('reason', '')
                                feedback_stage = '4단계' if gemma_config.get('frozen_evidence_pipeline') else '2단계'
                                fallback_stage = '5단계' if gemma_config.get('frozen_evidence_pipeline') else '3단계'
                                prediction[f'{feedback_stage} Reflection 피드백 ({condition})'] = feedback.get('feedback', '')
                                prediction[f'{fallback_stage} 피드백 재판정 ({condition})'] = reflected.get('verdict', '')
                                prediction[f'{fallback_stage} 재판정 근거 ({condition})'] = reflected.get('reason', '')
                            if gemma_config.get('decorative_gate'):
                                router = result.get('decorative_router_decision') or {}
                                observation = result.get('decorative_observation') or {}
                                prediction[f'장식 후보 라우터 ({condition})'] = router.get('verdict', '')
                                prediction[f'장식 구조화 관찰 ({condition})'] = observation.get('reason', '')
                                prediction[f'장식오용형 전용 확정 ({condition})'] = (
                                    '예' if result.get('decorative_gate_selected') else '아니오')
                            if gemma_config.get('type_reflection'):
                                initial_candidates = result.get('initial_type_classification') or {}
                                type_feedback = result.get('type_reflection_feedback') or {}
                                prediction[f'후보 Reflection 전 ({condition})'] = ' · '.join(
                                    item.get('label', '') for item in initial_candidates.get('matched_types', []))
                                prediction[f'후보 Reflection 전 정답 포함 ({condition})'] = (
                                    truth in {item.get('label') for item in initial_candidates.get('matched_types', [])}
                                    if truth in ERROR_TYPES else None)
                                prediction[f'후보 Reflection 피드백 ({condition})'] = type_feedback.get('feedback', '')
                                prediction[f'후보 Reflection 후 ({condition})'] = _matched_type_summary(
                                    result, gemma_config)
                            if gemma_config.get('final_reflection'):
                                initial_final = result.get('initial_final_decision') or {}
                                final_feedback = result.get('final_reflection_feedback') or {}
                                reflected_final = result.get('final_reflection_decision') or {}
                                prediction[f'최종 선택 Reflection 전 ({condition})'] = initial_final.get('label', '')
                                prediction[f'최종 선택 Reflection 피드백 ({condition})'] = (
                                    final_feedback.get('feedback') or final_feedback.get('action', ''))
                                prediction[f'최종 선택 Reflection 후 ({condition})'] = reflected_final.get('label', '')
                            if ok and result.get('role_decision'):
                                role = result['role_decision']
                                prediction[f'이미지 역할 ({condition})'] = ' · '.join(role['roles']) if 'roles' in role else role['role']
                                if 'description' in role:
                                    prediction[f'이미지 설명 ({condition})'] = role['description']
                            if binary.get('checks'):
                                if result.get('binary_route') == 'empty_alt':
                                    prediction[f'빈 alt · 핵심 정보와 문맥 대체 ({condition})'] = '\n'.join(
                                        f"{check['status']}: {check['information']} | 문구: {check['context_quote'] or '(없음)'}"
                                        for check in binary['checks'])
                                else:
                                    prediction[f'필요 정보 대조 ({condition})'] = '\n'.join(
                                        f"{check['status']}: {check['information']}" for check in binary['checks'])
                            if binary.get('image_roles'):
                                prediction[f'이미지 역할 ({condition})'] = ' · '.join(binary['image_roles'])
                                prediction[f'기준별 검사 ({condition})'] = '\n'.join(
                                    f"{check['status']}: {check['criterion']}" for check in binary['criteria_checks'])
                            prediction[f'1단계 근거 ({condition})'] = binary.get('reason', '')
                            prediction[f'1단계 정답 여부 ({condition})'] = (
                                (binary.get('verdict') == '적절') == (truth == '적절') if ok and truth else None)
                        candidates = candidate_labels(result) if ok else None
                        prediction[f'후보에 정답 포함 ({condition})'] = (
                            truth in candidates if truth and candidates is not None else None)
                else:
                    prediction.update({
                        'Gemma 예측 (정제 OCR 사용)': saved['label'],
                        'Gemma 예측 (OCR 미사용)': baseline['label'] if baseline and baseline.get('status') == 'ok' else '',
                        '정답 일치 (정제 OCR 사용)': saved['label'] == truth if truth else None,
                    })
                prediction['예측 이유' if direct else '예측 이유 (정제 OCR 사용)'] = saved['reason']
                predictions.append(prediction)
        st.caption(f"Gemma 판정 완료: {len(predictions):,} / {len(manifest['record_ids']):,}")
        if predictions:
            with st.expander('문제별 판정 결과'):
                st.dataframe(predictions, hide_index=True, height=250)
    log_path = run_dir / "worker.log"
    if log_path.exists():
        with st.expander("작업 로그"):
            st.text_area("작업 로그", log_path.read_text(encoding="utf-8", errors="replace")[-8000:], height=220, disabled=True)


def main() -> None:
    st.set_page_config(page_title="ALTogether", page_icon="🔎", layout="wide")
    st.title("ALTogether")
    st.caption("원본 이미지 + 메타데이터 → Gemma alt 품질 판정")
    st.sidebar.link_button("② 개별 결과 평가", "http://localhost:8502")
    if 'dataset_path' not in st.session_state:
        st.session_state['dataset_path'] = str(DEFAULT_DATASET)
    st.radio('실행 데이터', tuple(DATASET_PRESETS), key='dataset_preset', horizontal=True,
             on_change=_use_dataset_preset)
    dataset_text = st.sidebar.text_input("데이터셋 경로", key='dataset_path')
    dataset_dir = Path(dataset_text).expanduser()
    preset = st.session_state['dataset_preset']
    try:
        if preset == PUBLIC_109_PRESET:
            records = _public_test_records(dataset_dir, dataset_dir.parent / 'docs' / BOUNDARY_CASES_CSV.name)
        elif preset == COMBINED_FROZEN_PRESET:
            records = _combined_frozen_records(
                dataset_dir, dataset_dir.parent / 'test', dataset_dir.parent / 'docs' / BOUNDARY_CASES_CSV.name)
        else:
            records = load_records(dataset_dir)
    except Exception as exc:
        st.error(f"데이터셋을 읽을 수 없습니다: {exc}")
        return
    st.sidebar.metric("레코드", f"{len(records):,}")
    _new_run(dataset_dir, records, preset)
    st.subheader("실행 결과")
    runs = list_runs()
    if not runs:
        st.info("저장된 실행이 없습니다.")
        return
    labels = {str(run): Path(run).name for run in runs}
    active = st.session_state.get("active_run")
    index = list(labels).index(active) if active in labels else 0
    chosen = st.selectbox("실행", list(labels), index=index, format_func=labels.get, key="run_selector")
    st.session_state["active_run"] = chosen
    run_dir = Path(chosen)
    live = (get_status(run_dir) or {}).get('state') in {'running', 'stopping'}
    st.fragment(_active_run, run_every=LIVE_REFRESH_SECONDS if live else None)(run_dir, live)


if __name__ == "__main__":
    main()
