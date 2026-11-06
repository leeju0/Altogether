"""제출용 성능 화면: 공식 데이터 평가와 저장된 공식 성적."""
from pathlib import Path
import json
import streamlit as st
from common import ROOT,list_runs
from benchmark import show_benchmark
st.set_page_config(page_title='ALTogether · 성능',layout='wide')
st.title('ALTogether · 성능 및 판정 근거')
score=json.loads((ROOT.parent/'results/official_score.json').read_text(encoding='utf-8'))
st.caption('대회 제출 채점값 · 참가자 전달값 · 원본 제출 CSV와 해시를 results/에 보관')
cols=st.columns(3)
for c,k,label in zip(cols,('binary_f1','macro_f1','combined_score'),('부적절 탐지 F1','7범주 macro-F1','종합 점수')):
    c.metric(label,f'{score[k]:.4f}')
runs=[p for p in list_runs() if p.name!='reviewed_guidelines']
if runs:
    chosen=st.selectbox('실행 선택',runs,format_func=lambda p:p.name)
    show_benchmark(chosen)
else:
    st.info('실행 대시보드에서 Train 또는 Frozen 추론을 시작하면 단계별 평가가 표시됩니다.')
