"""Candidate coverage diagnostics, separate from the competition's single-label F1."""
import streamlit as st

from benchmark import VARIANT_NAMES
from gemma import LABELS


def show_fallback_candidate_scores(report):
    fallback = report['fallback_comparison']
    candidates = fallback['candidate_sets']
    names = report.get('variant_names') or VARIANT_NAMES
    direct = report.get('stage_layout') in ('guideline_direct_v1', 'guideline_all_stages_v1')
    candidate_step, final_step = ('3', '4') if direct else ('4', '5')
    final_name = '가이드라인 최종 선택' if report.get('stage_layout') == 'guideline_all_stages_v1' else '최종 선택'
    st.subheader(f'{candidate_step}·{final_step}단계 · 후보 생성과 최종 선택')
    st.caption(f"동일 문제 {len(candidates['paired_ids']):,} / 정답 공개 {report['labeled_count']:,}개 · "
               '각 방식의 폴백 전·후 판정과 후보가 모두 확인된 문제만 네 조건에 공통으로 사용합니다.')
    if report.get('role_review_comparison'):
        st.caption('폴백 있음은 역할 정보 미포함 조건입니다. 역할 정보 포함 조건은 3단계 상세에서 이진 성능만 비교합니다.')
    fallback_title = '폴백 있음 · 역할 정보 미포함' if report.get('role_review_comparison') else '폴백 있음'
    arms = [(variant, variant + '_' + mode, title)
            for variant in names for mode, title in [('no_fallback', '폴백 없음'), ('fallback', fallback_title)]]
    criteria = [('oracle', f'{candidate_step} · 후보 생성'), ('priority', f'{final_step} · {final_name}')]
    st.dataframe([{
        '방식': names[variant], '재검토': title,
        '평가 단계': label,
        '동일 문제 수': value['count'] if value else 0,
        '7범주 macro-F1': value[criterion]['macro_f1'] if value else None,
        'ACC': value[criterion]['accuracy'] if value else None,
    } for variant, key, title in arms for value in [candidates['paired'][key]]
      for criterion, label in criteria], hide_index=True)
    st.caption(f'{candidate_step}단계: 정답이 후보에 있으면 정답으로, 없으면 기존 오답 라벨로 계산한 진단용 F1입니다. '
               f'ACC는 후보 정답 포함률입니다. {final_step}단계: {final_name} 결과인 실제 예측의 F1·ACC입니다. '
               '두 점수 모두 앞선 이진 판정을 포함한 전체 문제 기준입니다.')
    if not candidates['paired_ids']:
        st.info('폴백 전·후 후보와 정답을 모두 확인할 수 있는 완료 결과가 아직 없습니다.')
    elif len(candidates['paired_ids']) < report['labeled_count']:
        st.caption('완료된 공통 문제의 잠정 점수입니다. 대기·실패·기록 불일치·후보 미확인 문제는 제외합니다.')
    with st.expander('유형별 F1 · 후보 포함률'):
        st.caption('후보 점수는 정답 정보를 사용하는 진단용이며 표준 멀티레이블 F1이 아닙니다. '
                   '7범주 고정, 분모가 0인 F1은 0입니다. 같은 폴백 조건 안에서는 두 방식의 이진 F1이 같습니다.')
        st.dataframe([{
            '방식': names[variant], '재검토': title, '동일 문제 수': value['count'],
            '최종 정답률': value['final_accuracy'], '후보 정답 포함률': value['hit_rate'],
            '선택에서 놓친 정답': value['selection_miss_count'],
            '후보에 정답 없음': value['candidate_miss_count'], '평균 후보 수': value['mean_size'],
        } for variant, key, title in arms for value in [candidates['paired'][key]] if value], hide_index=True)
        st.caption('후보 정답 포함률은 F1이 아닙니다. 후보 수가 많으면 포함률도 높아질 수 있습니다. '
                   '선택에서 놓친 정답은 후보에 정답이 있지만 우선순위 최종 라벨은 오답인 건수입니다.')
        for variant, key, title in arms:
            value = candidates['paired'][key]
            if value:
                st.caption(f'{names[variant]} · {title}')
                st.dataframe([{
                    '실제 레이블': label, '표본 수': value['per_class'][label]['support'],
                    '최종 선택 F1': value['priority']['per_class'][label]['f1'],
                    '후보 정답 인정 F1': value['oracle']['per_class'][label]['f1'],
                    '선택에서 놓친 정답': value['per_class'][label]['selection_miss_count'],
                    '후보에 정답 없음': value['per_class'][label]['candidate_miss_count'],
                } for label in LABELS], hide_index=True)
    with st.expander('문제별 폴백 전·후 후보와 최종 라벨'):
        for variant in names:
            before, after = variant + '_no_fallback', variant + '_fallback'
            st.caption(names[variant])
            st.dataframe([{
                '이미지 ID': row['record_id'], '실제 레이블': row['truth'],
                '폴백 전 후보': ', '.join(row[before + '_candidates']), '폴백 전 최종': row[before],
                '폴백 전 정답 포함': row[before + '_candidate_hit'],
                '폴백 후 후보': ', '.join(row[after + '_candidates']), '폴백 후 최종': row[after],
                '폴백 후 정답 포함': row[after + '_candidate_hit'],
            } for row in fallback['predictions'] if row['fallback_candidate_paired']], hide_index=True)
        st.caption('페이지 아래 성능 CSV에 네 조건의 점수, 항목별 예측 CSV에 두 후보 집합·최종 라벨·'
                   '정답 포함 여부·공통 채점 여부(fallback_candidate_paired)를 포함합니다.')


def show_candidate_scores(report):
    if (report.get('fallback_comparison') or {}).get('candidate_sets'):
        return  # The four-way table already includes the final candidate diagnostics.
    layout = report.get('stage_layout')
    direct = layout in ('guideline_direct_v1', 'guideline_all_stages_v1', 'label_reflection_v1')
    heading = ('5·6단계 · 후보 생성과 최종 선택' if layout == 'label_reflection_v1'
               else '2·3단계 · 후보 생성과 최종 선택' if direct
               else '최종 선택 vs 후보 집합')
    st.subheader(heading)
    candidates = report.get('candidate_sets')
    if not candidates:
        st.caption('후보 목록이 없는 단일 라벨 결과·CSV에서는 후보 포함 점수를 계산하지 않습니다.')
        return
    paired = bool(candidates['paired_ids'])
    metrics = candidates['paired'] if paired else candidates['available']
    names = report.get('variant_names') or VARIANT_NAMES
    final_name = '가이드라인 최종 선택' if report.get('stage_layout') == 'guideline_all_stages_v1' else '우선순위 최종 선택'
    single_run = len(metrics) == 1
    st.caption('단일 실행의 후보를 확인할 수 있는 완료 항목' if single_run
               else ('두 방식의 동일 ID 비교' if paired else '방식별 완료분 · 후보를 확인할 수 있는 항목만 비교 (잠정)'))
    criteria = ([('priority', final_name)] if report.get('type_reflection') else
                [('priority', final_name), ('oracle', '후보 정답 인정 (진단용)')])
    st.dataframe([{
        '방식': names[variant],
        '평가 단계': '후보 생성 단계' if criterion == 'oracle' else '최종 선택 단계', '판정 기준': title,
        '비교 수': value['count'] if value else 0,
        '7범주 macro-F1': value[criterion]['macro_f1'] if value else None,
        '이진 F1': value[criterion]['binary_f1'] if value else None,
        '종합 (0.5:0.5)': value[criterion]['combined_score'] if value else None,
    } for variant, value in metrics.items() for criterion, title in criteria], hide_index=True)
    if report.get('type_reflection'):
        st.caption('후보 생성 단계는 위의 정답 포함률로만 평가합니다. 이 표의 F1은 Reflection 후 후보를 받은 기존 최종 선택기의 실제 1개 라벨 성능입니다.')
    else:
        st.caption('후보 정답 인정 F1: 정답이 후보에 있으면 정답 라벨로, 없으면 기존 최종 예측으로 계산합니다. '
                   '정답 정보를 이용한 진단용 점수이며 후보 안에서 올바르게 선택했을 때의 개선 여지를 보여줍니다. '
                   '두 F1은 후보를 확인할 수 있는 동일 ID, 7개 범주 고정으로 계산합니다.')
    st.caption('현재 후보는 적절 단독 또는 부적절 유형들입니다. 부적절 유형 사이의 선택만 바뀌므로 이진 F1은 동일합니다.')
    st.dataframe([{
        '방식': names[variant], '비교 수': value['count'] if value else 0,
        '최종 정답률': value['final_accuracy'] if value else None,
        '후보 정답 포함률': value['hit_rate'] if value else None,
        '선택에서 놓친 정답': value['selection_miss_count'] if value else 0,
        '후보에 정답 없음': value['candidate_miss_count'] if value else 0,
        '평균 후보 수': value['mean_size'] if value else None,
    } for variant, value in metrics.items()], hide_index=True)
    st.caption('최종 정답률과 후보 포함률은 같은 항목으로 계산합니다. 후보 포함률은 공식 macro-F1이 아닙니다. '
               '후보를 많이 넣으면 포함률도 높아지므로 평균 후보 수를 함께 확인하세요.')
    st.caption('선택에서 놓친 정답 = 후보에 정답이 있지만 최종 라벨은 오답인 건수. '
               '적절 후보는 모델이 verdict="적절"을 명시한 경우에만 인정합니다.')
    for variant, coverage in candidates['coverage'].items():
        if coverage['unavailable']:
            st.caption(f"{names[variant]}: 최종 판정·정답은 있으나 후보를 확인할 수 없는 "
                       f"{coverage['unavailable']:,}개 제외. 기존 빈 목록을 적절로 추정하지 않습니다.")
    with st.expander('유형별 후보 포함률 · 선택에서 놓친 사례'):
        for variant, value in metrics.items():
            if not value:
                continue
            st.caption(names[variant])
            if report.get('type_reflection'):
                st.dataframe([{
                    '실제 레이블': label, '표본 수': value['per_class'][label]['support'],
                    '후보 정답 포함률': value['per_class'][label]['hit_rate'],
                    '후보에 정답 없음': value['per_class'][label]['candidate_miss_count'],
                } for label in LABELS], hide_index=True)
            else:
                st.dataframe([{
                    '실제 레이블': label, '표본 수': value['per_class'][label]['support'],
                    '최종 재현율': value['per_class'][label]['final_recall'],
                    '우선순위 F1': value['priority']['per_class'][label]['f1'],
                    '후보 정답 인정 F1': value['oracle']['per_class'][label]['f1'],
                    '후보 정답 포함률': value['per_class'][label]['hit_rate'],
                    '선택에서 놓친 정답': value['per_class'][label]['selection_miss_count'],
                    '후보에 정답 없음': value['per_class'][label]['candidate_miss_count'],
                } for label in LABELS], hide_index=True)
            ids = set(candidates['paired_ids']) if paired else None
            missed = [{'이미지 ID': row['record_id'], '실제 레이블': row['truth'],
                       'Gemma 후보': row.get(variant + '_candidates'), '최종 라벨': row[variant]}
                      for row in report['predictions'] if row.get(variant + '_selection_miss')
                      and (ids is None or row['record_id'] in ids)]
            if missed:
                st.dataframe(missed, hide_index=True, height=240)
