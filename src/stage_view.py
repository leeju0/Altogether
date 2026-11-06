"""The evaluation panels for current and historical staged pipelines."""
import streamlit as st

from benchmark import VARIANT_NAMES, show_fallback_comparison
from candidate_view import show_candidate_scores
from role_review_report import show_role_review_report
from gemma import REFLECTION_FROZEN

OFFICIAL_EVAL_BASELINE = {
    '방식': '공식 제로샷 VLM 베이스라인',
    '평가 문제 수': '584 / 600',
    'ACC': .764,
    '7범주 macro-F1': .631,
    'Binary F1': .747,
    '종합 (0.5:0.5)': .689,
}


def _binary_row(metrics):
    return {'평가 문제 수': metrics['count'] if metrics else 0,
            'Binary F1': metrics['binary_f1'] if metrics else None,
            'ACC': metrics['accuracy'] if metrics else None}


def show_stage_scores(report):
    names = report.get('variant_names') or VARIANT_NAMES
    first = report['first_stage']
    reflection = report.get('label_reflection')
    direct = report.get('stage_layout') in ('guideline_direct_v1', 'guideline_all_stages_v1',
                                             'label_reflection_v1', 'frozen_reflection_v1')
    first_step, fallback_step = ('1', '2') if direct else ('2', '3')
    if report.get('binary_only'):
        st.subheader('이진분류 전용 평가')
        st.caption('적절/부적절 판정만 평가합니다. 오류 유형 후보와 최종 7범주 점수는 생성하지 않습니다.')
    elif report.get('candidate_only'):
        st.subheader('Frozen MD · 후보풀 전용 평가')
        st.caption('이진판정과 오류 유형 후보 Reflection까지만 평가합니다. 최종 오류 유형 선택과 7범주 점수는 생성하지 않습니다.')
    else:
        st.subheader('공식 제로샷 VLM 베이스라인 · 평가 데이터')
        st.dataframe([OFFICIAL_EVAL_BASELINE], hide_index=True, width='stretch')
        st.caption('출처: 베이스라인_20261026.md · 가이드라인 v1.1 · 공개된 평가 라벨 109건 반영. '
                   '현재 Train 실행과 표본이 다르므로 직접 비교에 주의하세요.')
    st.caption('Binary F1의 양성은 부적절입니다. 각 표에 표시한 완료 문제 기준으로 계산합니다.')
    st.subheader(f'{first_step}단계 · 1차 이진 판단')
    scope = 'paired' if len(names) > 1 and first['paired_ids'] else 'available'
    st.dataframe([{'방식': names[v], '1차 완료': first['coverage'][v]['completed'],
                   **_binary_row(first[scope][v])} for v in names], hide_index=True)
    st.caption('유형 분류가 끝나기 전에도 저장된 1차 판정으로 집계합니다.' +
               (f" 두 방식 공통 {len(first['paired_ids']):,}개 기준입니다." if scope == 'paired' else ''))
    if not report['labeled_count']:
        st.info('정답이 공개되지 않아 F1과 ACC를 계산하지 않습니다.')
    with st.expander(f'{first_step}단계 상세 · 정밀도·재현율·혼동행렬'):
        for variant in names:
            m = first[scope][variant]
            if m:
                st.caption(names[variant])
                st.dataframe([{'라벨': label, **values} for label, values in m['per_class'].items()], hide_index=True)
                st.dataframe([{'실제 정답': label, **dict(zip(first['labels'], row))}
                              for label, row in zip(first['labels'], m['confusion_matrix'])], hide_index=True)
        if first['pairing_excluded']:
            st.caption(f"입력·설정을 확인하지 못한 {len(first['pairing_excluded']):,}개는 공통 비교에서 제외했습니다.")

    if reflection:
        st.subheader('2·3단계 · Reflection 피드백 전후 이진 성능')
        scope = 'paired' if len(names) > 1 and reflection['paired_ids'] else 'available'
        st.dataframe([{
            '방식': names[variant], 'Reflection 조건': title,
            **_binary_row(data[key] if data else None),
        } for variant in names for data in [reflection[scope].get(variant)]
          for key, title in [('before', '적용 전 · 최초 판정'),
                             ('after', '적용 후 · 피드백 재판정')]], hide_index=True)
        if reflection.get('reflection_mode') == REFLECTION_FROZEN:
            st.caption('Frozen MD 추론은 실제 레이블을 모델 입력으로 보내지 않고, 선택한 train 실행의 고정 Reflection 가이드라인으로 '
                       '피드백 제공 또는 통과를 선택합니다. 통과이면 최초 판정을 유지하고, 피드백 제공일 때만 재판정합니다. '
                       '실제 레이블은 추론이 끝난 뒤 이 표의 성능 계산에만 사용합니다. '
                       '결정적 링크 구조 게이트는 시스템 판정을 유지합니다.')
        else:
            st.caption('Train Reflection은 최초 이진판정과 train 정답의 이진 방향이 다른 문제에만 별도 세션이 피드백을 생성하는 teacher-assisted 실험입니다. '
                       '맞힌 문제는 피드백·재판정·MD 갱신에서 제외되며 최초 판정을 유지합니다. '
                       '같은 링크의 alt 중복 게이트는 정답 비교와 관계없이 시스템 판정을 유지합니다. '
                       '테스트셋 일반화 성능으로 해석하지 않습니다.')
        with st.expander('Reflection 상세 · 판정 변경과 교정·악화'):
            st.dataframe([{
                '방식': names[variant], '평가 문제 수': data['count'],
                '구조 게이트 유지': data.get('deterministic_gate_preserved', 0),
                '장식오용형 전용 게이트 확정': data.get('decorative_gate_selected', 0),
                '최초 이진 정답·학습 제외': data.get('initial_correct_skipped', 0),
                'Frozen 피드백 없이 통과': data.get('frozen_feedback_passed', 0),
                'Reflection 적용': data.get('feedback_applied', data['count']),
                '판정 변경': data['changed'], '오답 → 정답': data['corrected'],
                '정답 → 오답': data['harmed'],
            } for variant in names for data in [reflection[scope].get(variant)] if data], hide_index=True)

    type_reflection = report.get('type_reflection')
    if type_reflection:
        st.subheader('오류 유형 후보 Reflection 전후 · 정답 포함률')
        scope = 'paired' if len(names) > 1 and type_reflection['paired_ids'] else 'available'
        st.dataframe([{
            '방식': names[variant],
            '평가 가능한 부적절 문제': data['count'] if data else 0,
            '최초 후보에 정답 포함': data['before_hit_count'] if data else 0,
            '최초 후보 정답 포함률': data['before_hit_rate'] if data else None,
            '재판정 후보에 정답 포함': data['after_hit_count'] if data else 0,
            '재판정 후보 정답 포함률': data['after_hit_rate'] if data else None,
            '재판정 후보 정답 누락': data.get('after_miss_count') if data else 0,
            '최초 평균 후보 수': data.get('before_mean_size') if data else None,
            '재판정 평균 후보 수': data.get('after_mean_size') if data else None,
            '미포함 → 포함': data['corrected'] if data else 0,
            '포함 → 미포함': data['harmed'] if data else 0,
        } for variant in names
          for data in [type_reflection[scope].get(variant)]], hide_index=True)
        st.caption('후보 단계는 F1으로 채점하지 않습니다. 실제 부적절 유형이 후보 집합에 하나라도 포함됐는지만 계산하며, '
                   '실제 레이블이 적절인 문제는 이 표에서 제외합니다. 최종 1개 라벨 선택 성능은 아래 7범주 점수로 별도 평가합니다.')

    final_reflection = report.get('final_reflection')
    if final_reflection:
        st.subheader('최종 오류 유형 선택 Reflection 전후')
        scope = 'paired' if len(names) > 1 and final_reflection['paired_ids'] else 'available'
        st.dataframe([{
            '방식': names[variant],
            '선택 가능 문제': data['reachable_count'] if data else 0,
            '후보에 정답 없음': data['truth_missing_count'] if data else 0,
            '게이트 확정·선택기 생략': data.get('selector_bypassed_count', 0) if data else 0,
            '최초 선택 정답': data['initial_correct_count'] if data else 0,
            'Reflection 후 정답': data['final_correct_count'] if data else 0,
            '최초 선택 ACC': data['initial_accuracy'] if data else None,
            'Reflection 후 ACC': data['final_accuracy'] if data else None,
            '오답 → 정답': data['corrected'] if data else 0,
            '정답 → 오답': data['harmed'] if data else 0,
        } for variant in names
          for data in [final_reflection[scope].get(variant)]], hide_index=True)
        st.caption('선택 가능 문제는 실제 부적절 유형이 확정 후보 집합에 포함된 항목입니다. '
                   '후보에 정답이 없는 항목은 최종 선택기의 정확도에서 제외합니다. '
                   '장식오용형 게이트가 확정한 항목은 최종 선택기를 생략하므로 전후 선택 정확도에서는 제외하지만, '
                   '아래 전체 7범주 성능에는 전후 동일한 판정으로 포함합니다.')
        rows = []
        for variant in names:
            data = final_reflection[scope].get(variant)
            if not data:
                continue
            for key, title in [('full_before', 'Reflection 전'), ('full_after', 'Reflection 후')]:
                metrics = data[key]
                rows.append({
                    '방식': names[variant], '최종 선택 조건': title,
                    '평가 문제 수': metrics['count'],
                    '7범주 macro-F1': metrics['macro_f1'],
                    'Binary F1': metrics['binary_f1'],
                    '종합 (0.5:0.5)': metrics['combined_score'],
                    'ACC': metrics['accuracy'],
                })
        st.dataframe(rows, hide_index=True)
        if final_reflection.get('reflection_mode') == REFLECTION_FROZEN:
            st.caption('Frozen 일반화 평가는 실제 레이블 없이 고정 최종 선택 MD가 통과 또는 피드백 제공을 결정한 결과입니다.')
        else:
            st.caption('Train 수치는 후보에 정답이 있으면서 최초 최종 선택이 틀린 문제만 교정한 teacher-assisted 결과이며, 테스트셋 일반화 성능이 아닙니다.')
        st.caption('최종 선택 Reflection은 확정된 부적절 후보 안에서만 다시 선택하므로 적절/부적절 Binary F1은 전후 동일해야 합니다.')

    review = report.get('appropriate_review')
    if review:
        st.subheader(f'{fallback_step}단계 · 폴백 전후 이진 성능')
        scope = 'paired' if len(names) > 1 and review['paired_ids'] else 'available'
        on_title = '있음 · 역할 정보 미포함' if report.get('role_review_comparison') else '있음 · 저장된 폴백'
        st.dataframe([{'방식': names[v], '폴백 조건': title,
                       **_binary_row(data[key] if data else None)}
                      for v in names for data in [review[scope][v]]
                      for key, title in [('before', '없음 · 1차 판정 유지'), ('after', on_title)]], hide_index=True)
        st.caption('전후를 확인할 수 있는 동일 문제끼리 비교합니다. 빈 alt 등 재검토 생략 대상도 포함합니다.')
        with st.expander(f'{fallback_step}단계 상세 · 교정·악화와 역할 정보 비교'):
            st.dataframe([{'방식': names[v], '재검토 수': d['reviewed'], '판정 변경': d['changed'],
                           '오답 → 정답': d['corrected'], '정답 → 오답': d['harmed']}
                          for v in names for d in [review[scope][v]] if d], hide_index=True)
            show_role_review_report(report)
    if report.get('binary_only') or report.get('candidate_only'):
        return
    comparison = report.get('binary_stage_comparison')
    if comparison:
        st.subheader('이진 성능 · 후보 생성 전 vs 최종 7범주')
        scope = 'paired' if len(names) > 1 and comparison['paired_ids'] else 'available'
        rows = []
        for variant in names:
            data = comparison[scope].get(variant)
            if not data:
                continue
            for key, title in [('pre_candidate', '후보 생성 전 · 이진 판정'),
                               ('final_label', '최종 7범주 라벨 환산')]:
                metrics = data[key]
                rows.append({'방식': names[variant], '이진 성능 기준': title,
                             '평가 문제 수': metrics['count'],
                             'Binary F1': metrics['binary_f1'], '이진 ACC': metrics['accuracy']})
        st.dataframe(rows, hide_index=True)
        st.caption('두 행은 같은 최종 완료 문제로 계산합니다. 후보 생성 전은 Reflection까지 끝난 이진 판정이고, '
                   '최종 환산은 7범주 예측에서 적절과 6개 부적절 유형을 다시 이진화한 값입니다. '
                   '대회 종합 점수에는 최종 7범주 라벨 환산 이진 F1을 사용합니다.')
    if report.get('fallback_comparison'):
        show_fallback_comparison(report)
    else:
        show_candidate_scores(report)
