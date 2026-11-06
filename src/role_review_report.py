"""Paired binary evaluation of the two review inputs; no extra inference."""
import copy
import csv
import io
import json

from common import read_json
from gemma import needs_appropriate_review, parse_response, uses_empty_alt_prompt

CONDITIONS = {'initial': '1차 · 폴백 없음', 'without_role': '폴백 · 역할 정보 미포함',
              'with_role': '폴백 · 역할 정보 포함'}


def build_role_review_report(run_dir, config, records, results, paired_ids, excluded):
    if not config.get('compare_review_role'):
        return None
    from benchmark import BINARY_LABELS, score_labels
    predictions, errors = [], dict(excluded)
    for rid in paired_ids:
        result, record = results[rid], records[rid]
        first = result['binary_decision']
        eligible = needs_appropriate_review(config, record, first['verdict'], first)
        try:
            role_review = result['role_review_decision']
            if eligible:
                base = run_dir / 'raw/gemma_no_ocr'
                expected = copy.deepcopy(read_json(base / f'{rid}.recheck.request.json'))
                fields = json.loads(expected['messages'][1]['content'][1]['text'])
                fields['visual_evidence'] = result['role_decision']
                expected['messages'][1]['content'][1]['text'] = json.dumps(fields, ensure_ascii=False)
                if read_json(base / f'{rid}.recheck_role.request.json') != expected:
                    raise ValueError('역할 포함 폴백의 입력·프롬프트·생성 설정 불일치')
                raw = read_json(base / f'{rid}.recheck_role.json')
                if parse_response(raw, 'recheck_role_no_ocr', config) != role_review:
                    raise ValueError('역할 포함 폴백의 저장 결과와 원응답 불일치')
            elif role_review is not None:
                raise ValueError('폴백 대상이 아닌 문제에 역할 포함 폴백 결과가 있음')
            predictions.append({'record_id': rid, 'truth': '적절' if record['label'] == '적절' else '부적절',
                                'empty_alt': uses_empty_alt_prompt(config, record), 'reviewed': eligible,
                                'initial': first['verdict'], 'without_role': result['verdict'],
                                'with_role': role_review['verdict'] if eligible else first['verdict']})
        except (KeyError, TypeError, IndexError, ValueError) as exc:
            errors[rid] = str(exc) or '역할 포함 폴백 기록 오류'
    groups = {}
    for name, rows in [('all', predictions), ('nonempty_alt', [r for r in predictions if not r['empty_alt']]),
                       ('empty_alt', [r for r in predictions if r['empty_alt']])]:
        groups[name] = {}
        for condition in CONDITIONS:
            score = score_labels([r['truth'] for r in rows], [r[condition] for r in rows], labels=BINARY_LABELS)
            if score:
                score.pop('combined_score')  # The competition composite requires seven-class predictions.
            groups[name][condition] = score
    changes = {}
    for condition in ('without_role', 'with_role'):
        changed = [r for r in predictions if r[condition] != r['initial']]
        changes[condition] = {'reviewed': sum(r['reviewed'] for r in predictions), 'changed': len(changed),
                              'corrected': sum(r[condition] == r['truth'] for r in changed),
                              'harmed': sum(r['initial'] == r['truth'] for r in changed)}
    return {'paired_ids': [r['record_id'] for r in predictions], 'excluded': errors,
            'groups': groups, 'changes': changes, 'predictions': predictions,
            'primary_seven_class_arm': 'without_role'}


def role_review_csv(comparison):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=['record_id', 'truth', 'empty_alt', 'reviewed',
                                           'initial', 'without_role', 'with_role'])
    writer.writeheader()
    writer.writerows(comparison['predictions'])
    return out.getvalue().encode('utf-8-sig')


def show_role_review_report(report):
    import streamlit as st
    comparison = report.get('role_review_comparison')
    if comparison is None:
        return
    st.subheader('폴백 역할 정보 포함·미포함 · 이진 비교')
    st.caption(f"같은 문제 {len(comparison['paired_ids']):,} / 정답 공개 {report['labeled_count']:,}개 · "
               '역할 관찰과 1차 판정을 공유합니다. 일반 alt의 최초 적절에만 두 폴백을 각각 적용합니다. '
               '포함 조건에는 역할·관찰 JSON 전체를 추가하며, 역할 판정을 새로 호출하지 않습니다.')
    for group, title in [('nonempty_alt', '일반 alt'), ('empty_alt', '빈 alt · 전용 판정, 폴백 없음'),
                          ('all', '전체 · 빈 alt 포함')]:
        st.markdown(f'**{title}**')
        conditions = {'initial': '역할 관찰 → 빈 alt 전용 판정'} if group == 'empty_alt' else CONDITIONS
        rows = []
        for condition, name in conditions.items():
            m = comparison['groups'][group][condition]
            rows.append({'조건': name, '평가 문제': m['count'] if m else 0,
                         '이진 F1 (부적절)': m['binary_f1'] if m else None,
                         '정확도': m['accuracy'] if m else None,
                         '부적절 정밀도': m['per_class']['부적절']['precision'] if m else None,
                         '적절 재현율': m['per_class']['적절']['recall'] if m else None,
                         '부적절 재현율': m['per_class']['부적절']['recall'] if m else None})
        st.dataframe(rows, hide_index=True)
    st.dataframe([{'조건': CONDITIONS[key], '재검토 수': value['reviewed'],
                   '판정 변경': value['changed'], '오답 → 정답': value['corrected'], '정답 → 오답': value['harmed']}
                  for key, value in comparison['changes'].items()], hide_index=True)
    st.caption('두 폴백은 이진 성능으로 비교합니다. 아래 7범주 점수·최종 라벨은 역할 정보 미포함 폴백 기준입니다. '
               '빈 alt는 세 전체 조건에 동일한 전용 판정을 반영합니다. 실행 중에는 완료분의 잠정 점수입니다.')
    if comparison['excluded']:
        st.warning(f"입력·결과 기록을 확인하지 못한 {len(comparison['excluded']):,}개는 세 조건에서 함께 제외했습니다.")
    st.download_button('폴백 역할 비교 · 문제별 이진 CSV', role_review_csv(comparison),
                       file_name='role_review_predictions.csv', mime='text/csv')
