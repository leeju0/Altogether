"""Saved image/classification viewer and human-review dashboard."""
from __future__ import annotations

import base64
import json
from pathlib import Path

from PIL import Image, ImageDraw
import streamlit as st
from benchmark import candidate_labels

from common import (
    load_manifest,
    load_rating,
    load_records,
    load_result,
    list_runs,
    get_status,
    prepare_image,
    read_json,
    ratings_csv,
    resolve_image,
    save_rating,
)

ERROR_TYPES = ('파일명형', '무관형', '무의미형', '장식오용형', '중복형', '불충분형')


def _binary_review_comparison(result, truth_label):
    binary = result['binary_decision']
    empty = result.get('binary_route') == 'empty_alt'
    conditions = [('빈 alt 전용 · 폴백 없음' if empty else '1차 · 폴백 없음', binary)]
    if not empty:
        conditions.extend([('폴백 · 역할 정보 미포함', result.get('review_decision') or binary),
                           ('폴백 · 역할 정보 포함', result.get('role_review_decision') or binary)])
    truth = ('적절' if truth_label == '적절' else '부적절') if truth_label else None
    with st.container(border=True):
        columns = st.columns(1 + len(conditions))
        columns[0].metric('실제 이진 정답', truth or '미공개')
        for column, (name, decision) in zip(columns[1:], conditions):
            column.metric(name, decision['verdict'])
            column.caption(('✅ 정답' if decision['verdict'] == truth else '❌ 오답') if truth else '정답 미공개')
        st.caption('빈 alt는 역할·관찰 후 전용 판정만 수행하며 두 폴백을 모두 생략합니다.' if empty else
                   '역할·관찰과 1차 판정은 공유합니다. 최초 부적절이면 두 폴백을 생략하고 1차 판정을 유지합니다.')
        with st.expander('조건별 이진 판정 근거'):
            for name, decision in conditions:
                st.markdown(f'**{name}**')
                st.write(decision['reason'])


def _is_priority_run(config: dict | None) -> bool:
    return bool(config and config.get('classification_mode') in (
        'multilabel_priority_v1', 'explicit_verdict_multilabel_v1', 'two_stage_verdict_types_v1',
        'guideline_verdict_candidates_final_v1'))


def _matched_types(result: dict, config: dict | None) -> list[dict]:
    items = [item for item in result.get('matched_types', [])
             if isinstance(item, dict) and item.get('label') in ERROR_TYPES]
    priority = (config or {}).get('label_priority')
    if not isinstance(priority, list):
        return items
    matches = {item['label']: item for item in items}
    return [matches[label] for label in priority if label in matches]


def _matched_type_summary(result: dict, config: dict | None) -> str:
    if result.get('verdict') == '적절':
        return '적절 (모델 명시)'
    matches = _matched_types(result, config)
    return ' · '.join(item['label'] for item in matches) if matches else '오류 유형 없음 → 적절'


def _binary_only_result_details(result: dict, truth_label: str | None) -> None:
    initial = result.get('binary_decision') or result
    st.markdown(f"**1단계 · 최초 이진판정: {initial['verdict']}**")
    st.write(initial.get('reason', ''))
    feedback = result.get('reflection_feedback')
    if feedback:
        action = feedback.get('action')
        st.markdown(f"**Reflection: {action or '피드백 제공'}**")
        st.write(feedback.get('reason') or feedback.get('feedback') or '')
    reflected = result.get('reflection_decision')
    if reflected and not result.get('reflection_skipped'):
        st.markdown(f"**재판정: {reflected['verdict']}**")
        st.write(reflected.get('reason', ''))
    elif result.get('reflection_skipped'):
        st.caption('재판정 생략 · ' + result['reflection_skipped'])
    if truth_label:
        truth = '적절' if truth_label == '적절' else '부적절'
        st.caption(('✅ 이진 정답' if result.get('verdict') == truth else '❌ 이진 오답')
                   + f' · 실제 이진 정답: {truth}')


def _priority_result_details(result: dict, config: dict, truth_label, condition: str) -> None:
    matches = _matched_types(result, config)
    current = bool(config.get('role_observation_mode'))
    guideline = bool(config.get('guideline_mode'))
    model_final = config.get('classification_mode') == 'guideline_verdict_candidates_final_v1'
    binary_step, types_step = (('2', '4') if current else
                               ('1', '3' if config.get('recheck_appropriate') else '2') if guideline else
                               ('1', '2'))
    evidence_review = config.get('evidence_stage') and config.get('review_policy') != 'appropriate_v1'
    binary = result.get('binary_decision')
    if binary:
        if result.get('role_decision'):
            role = result['role_decision']
            role_text = ' · '.join(role['roles']) if 'roles' in role else role['role']
            st.markdown(f"**{'1단계 · 역할 파악' if current else '사전 역할 판정'}: {role_text}**")
            st.write(role.get('description', role.get('reason', '')))
            if config.get('role_observation_mode'):
                st.caption('alt 없이 이미지·문맥으로 역할을 복수 선택하고 이미지 설명만 추출한 결과입니다.')
            elif config.get('evidence_stage'):
                st.caption('역할은 참고값입니다. 관찰한 정보와 문맥을 바탕으로 충분성을 대조합니다.')
                st.json({key: role.get(key, []) for key in ('visible_facts', 'context_quotes', 'required_information')})
            else:
                st.caption('역할은 관찰 기록이며, 1차 판정에는 빈 alt 전용 기준을 적용합니다.'
                       if result.get('binary_route') == 'empty_alt'
                       else '적용 기준: ' + config.get('role_criteria', {}).get(role['role'], '미기록'))
        if config.get('evidence_stage'):
            if result.get('binary_route') == 'empty_alt':
                st.markdown('**빈 alt · 핵심 정보와 문맥 대체 여부**')
                st.dataframe([{'생략 시 놓치는 핵심 정보': check['information'],
                               '주변·링크 실제 문구': check['context_quote'], '문맥 대체 여부': check['status']}
                              for check in binary.get('checks', [])], hide_index=True)
            else:
                st.markdown('**필요한 정보 · alt/문맥 충족 대조**')
                st.dataframe([{'필요한 정보': check['information'], 'alt 원문 근거': check['alt_quote'],
                               '문맥 원문 근거': check['context_quote'], '충족 여부': check['status']}
                              for check in binary.get('checks', [])], hide_index=True)
            if evidence_review:
                st.caption('모델의 근거 재검토 요청: ' + ('있음' if binary.get('needs_review') else '없음'))
            else:
                st.caption('폴백 정책: 일반 alt의 최초 적절 전체에 기존 적절 기준·6유형 적용')
        if binary.get('image_roles'):
            st.markdown('**판단한 이미지 역할:** ' + ' · '.join(binary['image_roles']))
            with st.expander('역할별 충족 기준 · 이미지·문맥 근거', expanded=True):
                st.write(binary.get('image_evidence', ''))
                st.write(binary.get('context_evidence', ''))
                st.dataframe([{'판정 기준': check['criterion'], 'alt 근거': check['alt_evidence'],
                               '문맥 근거': check['context_evidence'], '충족 여부': check['status']}
                              for check in binary.get('criteria_checks', [])], hide_index=True)
        if result.get('binary_route') == 'empty_alt':
            st.caption('1차 판정 경로: 빈 alt 전용 · 주변·링크 텍스트의 정보 대체 여부 확인')
        if result.get('binary_decision_source') == 'empty_alt_link_v1':
            st.caption('1차 판정 경로: 빈 alt 링크 구조 게이트 · Gemma 이진 호출 생략')
        if result.get('binary_decision_source') == 'empty_alt_link_visual_review_v1':
            st.caption('1차 판정 경로: 빈 alt 링크 시각 예외 검사 · 이미지의 추가 실질 정보 확인')
        st.markdown(f"**{binary_step}단계 · 적절/부적절: {binary['verdict']}**")
        st.write(binary['reason'])
        if truth_label:
            correct = (binary['verdict'] == '적절') == (truth_label == '적절')
            st.caption(f'✅ {binary_step}단계 이진 판정 정답' if correct else f'❌ {binary_step}단계 이진 판정 오답')
    elif result.get('verdict'):
        st.write(f"**모델 적절/부적절:** {result['verdict']}")
    review = result.get('review_decision')
    if config.get('recheck_appropriate') and binary:
        before_label = '적절' if binary['verdict'] == '적절' else result.get('initial_label', result['label'])
        st.markdown(f"**폴백 없음: {before_label} → 폴백 있음: {result['label']}**")
    if review:
        review_prefix = '3단계 · ' if current else '2단계 · ' if guideline else ''
        st.markdown(f"**{review_prefix}{'근거 재검토' if evidence_review else '적절 재검토'}: {binary['verdict']} → {review['verdict']}**")
        st.write(review['reason'])
        if evidence_review:
            with st.expander('재검토 후 충족 대조'):
                st.json(review.get('checks', []))
        if truth_label:
            correct = (review['verdict'] == '적절') == (truth_label == '적절')
            st.caption('✅ 재검토 후 이진 판정 정답' if correct else '❌ 재검토 후 이진 판정 오답')
    elif config.get('recheck_appropriate') and binary:
        st.caption('적절 재검토 생략 · 빈 alt 전용 판정 유지'
                   if config.get('skip_empty_alt_review') and result.get('binary_route') == 'empty_alt'
                   else '근거 재검토 생략 · 추가 확인 요청 없음' if evidence_review
                   else '적절 재검토 생략 · 최초 판정이 부적절')
    skipped = binary and result.get('verdict') == '적절'
    if skipped:
        st.caption(f'{types_step}단계 생략 · 재검토 후 적절로 판정' if review else f'{types_step}단계 생략 · {binary_step}단계에서 적절로 판정')
    else:
        if config.get('type_reflection') and result.get('initial_type_classification'):
            initial_items = result['initial_type_classification'].get('matched_types', [])
            st.markdown(f'**{types_step}단계 · 후보 Reflection 전 · {condition}**')
            st.write(' · '.join(item.get('label', '') for item in initial_items) or '(후보 없음)')
            type_feedback = result.get('type_reflection_feedback') or {}
            if type_feedback.get('feedback'):
                st.caption('후보 Reflection 피드백: ' + type_feedback['feedback'])
            elif result.get('type_reflection_skipped'):
                st.caption('후보 Reflection 생략: ' + result['type_reflection_skipped'])
        title = f'{types_step}단계 · 오류 유형 후보' if binary else 'Gemma 복수 라벨'
        st.markdown(f'**{title}{" · Reflection 후" if config.get("type_reflection") else ""} · {condition}**')
        st.write(_matched_type_summary(result, config))
    if len(matches) > 1:
        priority = result.get('decision_rule', {}).get('priority') or config.get('label_priority')
        labels = {item['label'] for item in matches}
        if priority:
            st.caption(('↓ 가이드라인 우선순위 참고: ' if model_final else '↓ 적용 우선순위: ') +
                       ' > '.join(label for label in priority if label in labels))
    elif not skipped:
        st.caption('↓ 후보가 하나이므로 최종 후보 재확인' if model_final else '↓ 코드 규칙 적용')
    final_prefix = ('5단계 · ' if current else
                    f"{'4' if config.get('recheck_appropriate') else '3'}단계 · " if guideline else '')
    if model_final and config.get('final_reflection') and result.get('initial_final_decision'):
        initial_final = result['initial_final_decision']
        st.markdown(f"**최종 선택 Reflection 전: {initial_final['label']}**")
        st.write(initial_final.get('reason', ''))
        final_feedback = result.get('final_reflection_feedback') or {}
        if final_feedback.get('action') == '통과':
            st.markdown('**최종 선택 Reflection: 통과**')
            st.write(final_feedback.get('reason', ''))
        elif final_feedback.get('feedback'):
            st.markdown('**최종 선택 Reflection 피드백**')
            st.write(final_feedback['feedback'])
        elif result.get('final_reflection_skipped'):
            st.caption('최종 선택 Reflection 생략: ' + result['final_reflection_skipped'])
        reflected_final = result.get('final_reflection_decision') or result.get('final_decision')
        if reflected_final:
            st.markdown(f"**최종 선택 Reflection 후: {reflected_final['label']}**")
            st.write(reflected_final.get('reason', ''))
    st.metric(f"{final_prefix}{'가이드라인 최종 판정' if model_final else '기계적 최종 판정'} · {condition}", result['label'])
    if model_final and result.get('final_decision'):
        st.write(result['final_decision']['reason'])
        st.caption(f"비교용 기계적 우선순위 결과: {result.get('priority_label', '미기록')}")
    if truth_label:
        verdict = '✅ **정답**' if result['label'] == truth_label else '❌ **오답**'
        st.markdown(f'{verdict} · 실제 레이블: **{truth_label}**')
        candidates = candidate_labels(result)
        if candidates is None:
            st.caption('후보 포함 여부 확인 불가 · 기존 빈 목록을 적절로 추정하지 않습니다.')
        elif truth_label in candidates:
            st.caption('후보에 정답 포함 · 최종 선택도 정답' if result['label'] == truth_label
                       else '후보에 정답 포함 · 최종 선택은 오답')
        else:
            st.caption('후보에 정답 없음')
    else:
        st.caption('정답 미공개 · 정답 여부 판정 불가')
    with st.expander('유형별 근거 · 종합 판단'):
        for item in matches:
            st.markdown(f"- {item['label']}: {item.get('reason', '')}")
        st.write(result.get('assessment_reason') or result['reason'])


def _record_id(row: dict) -> str:
    return str(row.get("record_id", row.get("id", "")))


def _polygon(points: object) -> list[tuple[float, float]]:
    if not isinstance(points, list):
        return []
    if len(points) == 4 and all(isinstance(point, (int, float)) for point in points):
        x1, y1, x2, y2 = points
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return [tuple(point[:2]) for point in points if isinstance(point, (list, tuple)) and len(point) >= 2]


def overlay(image, result: dict | None):
    """Return an image with numbered detection/item polygons in canonical coordinates."""
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    entities = (result or {}).get("detections") or (result or {}).get("items") or []
    for number, entity in enumerate(entities, start=1):
        points = _polygon(entity.get("polygon", []))
        if len(points) < 2:
            continue
        draw.line(points + [points[0]], fill="red", width=max(2, image.width // 500))
        x, y = points[0]
        # Put a small outlined number above the box instead of covering text
        # with a solid badge (particularly important for tiny web logos).
        draw.text((max(0, x), max(0, y - 12)), str(entity.get("id", number)),
                  fill="red", stroke_width=1, stroke_fill="white")
    return marked, entities


def _preview_width(image) -> int:
    # Limit only the display size; preserve original pixels for OCR and overlays.
    return max(1, min(image.width, 720, int(image.width * 360 / image.height)))


def _result_panel(preview, result: dict | None, run_dir: Path) -> None:
    st.markdown("#### PaddleOCR-VL-1.6", width="stretch", text_alignment="center")
    if not result:
        st.caption("아직 결과가 없습니다.")
        return
    image = preview
    input_path = result.get("input_path")
    if input_path:
        candidate = run_dir / input_path
        if candidate.is_file():
            with Image.open(candidate) as cached:
                image = cached.convert("RGB")
    status = result.get("status", "unknown")
    if status == "error":
        st.error(f"처리 상태: {status} — {result.get('error', '세부 정보 없음')}")
        if image is not None:
            with st.container(horizontal_alignment="center"):
                st.image(image, caption="저장된 입력", width=_preview_width(image))
        return
    if image is None:
        st.error("원본과 저장된 입력을 모두 열 수 없습니다.")
        return
    if status != "ok":
        st.warning(f"처리 상태: {status} — 부분 결과를 표시합니다.")
    marked, entities = overlay(image, result)
    with st.container(horizontal_alignment="center"):
        st.image(marked, caption=f"검출 {len(entities)}개", width=_preview_width(image))
    items = result.get("items") or []
    image_ocr = load_manifest(run_dir).get("use_ocr_for_image_block", False)
    detection_scores = {box['id']: box.get('confidence') for box in result.get("detections", [])}
    with st.container(height=220, border=False):
        if not items:
            st.caption("인식된 텍스트 없음")
        for item in items:
            text = item.get("text") or ""
            if not text.strip():
                skipped = not image_ocr and item.get("label") in ("image", "header_image", "footer_image")
                text = "그림 영역 · OCR 생략 (이 실행은 그림 영역 OCR이 꺼져 있음)" if skipped else "인식된 텍스트 없음"
            score = detection_scores.get(item.get("id"))
            suffix = f"  ·  검출 점수 {float(score):.3f}" if score is not None else ""
            st.write(f"{item.get('id', '?')}. {text}{suffix}")


def _navigate(ids: list[str], run_dir: Path) -> tuple[int, str]:
    state_key = f"review_index:{run_dir.name}"
    record_key = f"record:{run_dir.name}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0
    st.session_state[state_key] = min(max(st.session_state[state_key], 0), len(ids) - 1)
    def move(amount: int) -> None:
        st.session_state[state_key] = min(max(st.session_state[state_key] + amount, 0), len(ids) - 1)
        st.session_state[record_key] = ids[st.session_state[state_key]]
    before, current, after = st.columns((1, 3, 1))
    with before:
        st.button("← 이전", key=f"previous:{run_dir.name}", disabled=st.session_state[state_key] == 0,
                  on_click=move, args=(-1,), width="stretch")
    with after:
        st.button("다음 →", key=f"next:{run_dir.name}", disabled=st.session_state[state_key] == len(ids) - 1,
                  on_click=move, args=(1,), width="stretch")
    with current:
        labels = {record_id: f"{number:,} / {len(ids):,} · {record_id}"
                  for number, record_id in enumerate(ids, start=1)}
        if record_key not in st.session_state:
            st.session_state[record_key] = ids[st.session_state[state_key]]
        selected = st.selectbox("레코드", ids, format_func=labels.get, key=record_key, label_visibility="collapsed")
        st.session_state[state_key] = ids.index(selected)
    return st.session_state[state_key], ids[st.session_state[state_key]]


def _ratings(run_dir: Path, record_id: str, reviewer: str, prediction=None, truth_label=None, gemma_config=None) -> None:
    direct = load_manifest(run_dir).get('pipeline_mode') == 'gemma_only'
    existing = load_rating(run_dir, record_id, reviewer) if reviewer else None
    identity = f"quality:{run_dir.name}:{record_id}:{reviewer}"
    with st.container(border=True):
        labels = {1: "매우 나쁨", 2: "나쁨", 3: "보통", 4: "좋음", 5: "매우 좋음"}
        choices = tuple(labels)
        detection = text_score = None
        detection_reason = text_reason = ''
        if not direct:
            detection_value = (existing or {}).get("detection_score")
            text_value = (existing or {}).get("text_score")
            # Scores rerun immediately so low-score reason fields appear on selection.
            detection_col, text_col = st.columns(2)
            with detection_col:
                detection = st.radio("영역 검출 품질", choices,
                                     index=choices.index(detection_value) if detection_value in choices else None,
                                     format_func=lambda score: f"{score} · {labels[score]}",
                                     horizontal=True, key=f"detection:{identity}")
            with text_col:
                text_score = st.radio("텍스트 품질", choices,
                                      index=choices.index(text_value) if text_value in choices else None,
                                      format_func=lambda score: f"{score} · {labels[score]}",
                                      horizontal=True, key=f"text:{identity}")
            detection_reason = st.text_area("영역 검출 평가 이유 (선택)", (existing or {}).get("detection_reason", ""),
                                            key=f"detection_reason:{identity}", height=80) if detection in (1, 2) else ""
            text_reason = st.text_area("텍스트 평가 이유 (선택)", (existing or {}).get("text_reason", ""),
                                      key=f"text_reason:{identity}", height=80) if text_score in (1, 2) else ""
        classification, classification_reason = None, ''
        if prediction and prediction.get('status') == 'ok':
            if direct:
                if (gemma_config or {}).get('binary_only') or prediction.get('binary_only'):
                    truth = ('적절' if truth_label == '적절' else '부적절') if truth_label else '미공개'
                    st.caption(f"실제 이진 정답: {truth} · Gemma 이진판정: {prediction['verdict']}")
                else:
                    st.caption(f"실제 레이블: {truth_label or '미공개'} · Gemma 판정: {prediction['label']}")
            elif _is_priority_run(gemma_config):
                label = ('가이드라인 최종 판정' if gemma_config.get('classification_mode') == 'guideline_verdict_candidates_final_v1'
                         else '기계적 최종 판정')
                st.caption(f"실제 레이블: {truth_label or '미공개'} · {label} (정제 OCR 사용): {prediction['label']}")
            else:
                st.caption(f"실제 레이블: {truth_label or '미공개'} · Gemma 예측 (정제 OCR 사용): {prediction['label']}")
            value = (existing or {}).get('classification_score')
            classification = st.radio('모델 판정 품질' if direct else '모델 판정 품질 (정제 OCR 사용)', choices,
                                      index=choices.index(value) if value in choices else None,
                                      format_func=lambda score: f'{score} · {labels[score]}',
                                      horizontal=True, key=f'classification:{identity}')
            if classification in (1, 2):
                classification_reason = st.text_area('모델 판정 평가 이유 (선택)',
                                                     (existing or {}).get('classification_reason', ''),
                                                     key=f'classification_reason:{identity}', height=80)
        submitted = st.button("평가 저장", type="primary", key=f"save:{identity}")
        if submitted:
            if not reviewer.strip():
                st.error("검토자 이름을 입력하세요.")
            elif direct and classification is None:
                st.error('모델 판정 품질 점수를 선택하세요.')
            elif not direct and (detection is None or text_score is None):
                st.error("두 항목의 점수를 모두 선택하세요.")
            else:
                save_rating(run_dir, record_id, reviewer.strip(), detection, text_score, detection_reason, text_reason,
                            classification_score=classification, classification_reason=classification_reason)
                st.session_state.reviewer = reviewer.strip()
                csv_data = ratings_csv(run_dir)
                filename = f"CV_PBL_{run_dir.name}_quality_ratings.csv"
                encoded_data = base64.b64encode(csv_data).decode("ascii")
                encoded_name = base64.b64encode(filename.encode()).decode("ascii")
                # Save current widget values before preparing the download.
                # Only base64 data is interpolated; reviewer text never becomes JS.
                st.html(f"""<script>
                    (() => {{
                    const link = document.createElement('a');
                    link.href = 'data:text/csv;charset=utf-8;base64,{encoded_data}';
                    link.download = atob('{encoded_name}');
                    document.body.appendChild(link);
                    link.click();
                    link.remove();
                    }})();
                </script>""", unsafe_allow_javascript=True)
                st.caption("평가를 저장했습니다. 다운로드가 시작되지 않으면 아래 버튼을 누르세요.")
                st.download_button("CSV 내려받기", csv_data, file_name=filename, mime="text/csv", on_click="ignore")


def _review_record(run_dir, manifest, ids, by_id, reviewer, refresh_was_active=False):
    if refresh_was_active and (get_status(run_dir) or {}).get('state') not in {'running', 'stopping'}:
        # The timer was created while the run was live. Rebuild once without it.
        st.rerun()
    direct = manifest.get('pipeline_mode') == 'gemma_only'
    st.caption('판정이 완료된 문제부터 평가할 수 있습니다.')
    # Reserve the body before navigation so its selector chooses the record for
    # this render, while the navigation controls appear below the survey.
    body = st.container()
    _, record_id = _navigate(ids, run_dir)
    row = by_id[record_id]
    with body:
        source = None
        metadata = None
        source_error = None
        try:
            source, metadata = prepare_image(resolve_image(Path(manifest["dataset_dir"]), row), background=manifest.get("background", "white"))
        except Exception as exc:
            source_error = str(exc)
        result = load_result(run_dir, "vl", record_id)
        prediction = load_result(run_dir, 'gemma', record_id)
        baseline = load_result(run_dir, 'gemma_no_ocr', record_id)
        cleaned = load_result(run_dir, 'cleaned', record_id)
        original_col, result_col = (st.container(), None) if direct else st.columns(2)
        with original_col:
            st.markdown("#### 원본 이미지", width="stretch", text_alignment="center")
            fallback = baseline if direct else result
            if source is None and ((fallback and fallback.get("input_path")) or direct):
                cached_path = run_dir / (fallback.get("input_path", f"inputs/{record_id}.png") if fallback else f"inputs/{record_id}.png")
                if cached_path.is_file():
                    with Image.open(cached_path) as cached:
                        source = cached.convert("RGB")
                    st.caption("원본 파일을 찾을 수 없어 저장된 입력 이미지를 표시합니다.")
            if source is not None:
                with st.container(horizontal_alignment="center"):
                    st.image(source, caption=f"{source.width}×{source.height}", width=_preview_width(source))
                if metadata and metadata.get('background_adjusted'):
                    st.caption('흰 전경이 보이도록 밝은 회색 배경 적용 · #E0E0E0')
            else:
                st.error(f"원본 미리보기를 만들 수 없습니다: {source_error}")
            st.caption(f"이미지 ID: {record_id}")
            alt_text = row.get("alt_text") or 'alt=""'
            st.write(f"**실제 alt:** {alt_text}")
            with st.expander('입력 메타데이터'):
                from gemma import METADATA_FIELDS
                st.json({key: row.get(key) for key in METADATA_FIELDS})
        if not direct:
            with result_col:
                _result_panel(source, result, run_dir)
                if cleaned and cleaned.get('status') == 'ok':
                    with st.expander(f"Gemma 정제 OCR · 보존 {len(cleaned['kept_items'])} / 제외 {len(cleaned['excluded_ids'])}", expanded=True):
                        with st.container(height=160, border=False):
                            st.text(cleaned['text'] or '보존한 OCR 문구 없음')
                        st.download_button('정제 OCR JSON 내려받기', json.dumps(cleaned, ensure_ascii=False, indent=2),
                                           file_name=f'{record_id}-cleaned-ocr.json', mime='application/json')
                        if cleaned.get('excluded_ids'):
                            st.caption('제외 문구: ' + ' / '.join(item['text'] for item in cleaned.get('candidates', [])
                                                               if item['id'] in cleaned['excluded_ids']))
                elif cleaned:
                    st.error('OCR 정제 실패: ' + cleaned.get('error', '결과 없음'))
        gemma_config = manifest.get('gemma') or {}
        priority_run = _is_priority_run(gemma_config)
        if direct:
            prediction = baseline
        variants = [('Gemma', 'gemma_no_ocr', prediction)] if direct else [('정제 OCR 사용', 'gemma', prediction)]
        if direct and gemma_config.get('compare_review_role'):
            if prediction and prediction.get('status') == 'ok':
                with st.expander('역할 정보 포함 폴백 · 보조 이진 비교'):
                    _binary_review_comparison(prediction, row.get('label'))
                    st.caption('최종 라벨과 설문은 역할 정보 미포함 폴백 기준입니다.')
        if not direct and (manifest.get('compare_without_ocr') or baseline):
            variants.append(('OCR 미사용', 'gemma_no_ocr', baseline))
        with st.container(border=True):
            columns = st.columns(1 + len(variants))
            with columns[0]:
                st.metric('실제 레이블 · 데이터 정답', row.get('label') or '미공개')
            for column, (condition, model, saved) in zip(columns[1:], variants):
                with column:
                    ok = saved and saved.get('status') == 'ok'
                    binary_only = bool(ok and direct and (gemma_config.get('binary_only') or saved.get('binary_only')))
                    if binary_only:
                        st.metric('최종 이진판정 · Gemma', saved['verdict'])
                    elif not (priority_run and ok):
                        title = ((f"{'가이드라인 최종 판정' if gemma_config.get('classification_mode') == 'guideline_verdict_candidates_final_v1' else '기계적 최종 판정'} · {condition}")
                                 if priority_run else f'Gemma 예측 · {condition}')
                        st.metric(title, saved['label'] if ok else ('실패' if saved else '결과 없음'))
                    if ok:
                        if binary_only:
                            st.write(saved['reason'])
                            with st.expander('이진판정 과정 · 근거'):
                                _binary_only_result_details(saved, row.get('label'))
                        elif priority_run:
                            if direct:
                                final_step = '4' if gemma_config.get('guideline_mode') else '5'
                                types_step = '3' if gemma_config.get('guideline_mode') else '4'
                                st.metric(f'{final_step}단계 · 최종 라벨', saved['label'])
                                st.caption(f'{types_step}단계 · 오류 유형 분류 생략 (적절)' if saved.get('verdict') == '적절'
                                           else f'{types_step}단계 · 후보: ' + _matched_type_summary(saved, gemma_config))
                                st.write(saved['reason'])
                                with st.expander(('1~4단계' if gemma_config.get('guideline_mode') else '1~5단계') + ' 판정 과정 · 근거'):
                                    _priority_result_details(saved, gemma_config, row.get('label'), condition)
                            else:
                                _priority_result_details(saved, gemma_config, row.get('label'), condition)
                        else:
                            if row.get('label'):
                                st.caption('✅ 정답과 일치' if saved['label'] == row['label'] else '❌ 정답과 불일치')
                            else:
                                st.caption('정답 미공개 · 일치 여부 비교 불가')
                            st.write(saved['reason'])
                        with st.expander(f'예측 근거 · JSON ({condition})'):
                            st.write('이미지 근거: ' + saved.get('image_evidence', ''))
                            st.write('문맥 근거: ' + saved.get('context_evidence', ''))
                            st.json(saved)
                            st.download_button('예측 JSON 내려받기', json.dumps(saved, ensure_ascii=False, indent=2),
                                               file_name=f'{record_id}-{model}.json', mime='application/json', key=f'prediction:{model}:{record_id}')
                    elif saved:
                        st.error(saved.get('error', '결과 없음'))
        _ratings(run_dir, record_id, reviewer, prediction, row.get('label'), gemma_config)
    if manifest.get('gemma'):
        with st.expander('전체 판정 성능' if direct else '전체 OCR 유무 벤치마크'):
            from benchmark import show_benchmark
            show_benchmark(run_dir)


def main() -> None:
    st.set_page_config(page_title="CV_PBL", page_icon="📝", layout="wide")
    st.sidebar.link_button("① 테스트 · 실행", "http://localhost:8501")
    runs = [run for run in list_runs() if run.name != "reviewed_guidelines" and (load_manifest(run).get("gemma") or "vl" in load_manifest(run).get("models", []))]
    if not runs:
        st.info("검토할 저장 실행이 없습니다. 실행 페이지에서 작업을 시작하세요.")
        return
    labels = {str(run): Path(run).name for run in runs}
    chosen = st.sidebar.selectbox("저장된 실행", list(labels), format_func=labels.get)
    run_dir = Path(chosen)
    manifest = load_manifest(run_dir) or {}
    # The run snapshot keeps labels and alt text stable when its source dataset changes.
    records = read_json(run_dir / "records.json")
    if records is None:
        try:
            records = load_records(Path(manifest["dataset_dir"]))
        except Exception as exc:
            st.error(f"원본 데이터를 읽을 수 없습니다: {exc}")
            return
    selected_ids = manifest.get("record_ids", [])
    by_id = {_record_id(row): row for row in records}
    ids = [record_id for record_id in selected_ids if record_id in by_id]
    if not ids:
        st.warning("실행에 연결된 레코드를 찾을 수 없습니다.")
        return
    reviewer = st.sidebar.text_input("검토자", st.session_state.get("reviewer", ""), key="reviewer_sidebar")
    live = (get_status(run_dir) or {}).get('state') in {'running', 'stopping'}
    auto_refresh = st.sidebar.toggle("결과 자동 갱신 (3초)", value=True, key="review_live", disabled=not live,
                                     help='선택한 실행이 진행 중일 때만 자동으로 갱신합니다.')
    st.sidebar.button("결과 새로고침")
    refreshing = live and auto_refresh
    st.fragment(_review_record, run_every=3 if refreshing else None)(
        run_dir, manifest, ids, by_id, reviewer, refreshing)
    st.sidebar.download_button("평가 CSV 내려받기", lambda: ratings_csv(run_dir), file_name=f"{run_dir.name}-quality-ratings.csv", mime="text/csv")


if __name__ == "__main__":
    main()
