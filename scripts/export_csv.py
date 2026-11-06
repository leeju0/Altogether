"""완료된 Test 결과만 record_id,label CSV로 내보낸다."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import load_manifest,load_records,load_result
from gemma import LABELS
from benchmark import label_csv
from run_app import _public_test_labels
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--apply-public-labels',action='store_true',help='대회가 허용한 공개 정답 109건 반영')
a=p.parse_args()
ids=[r['record_id'] for r in load_records(ROOT/'data/test')]
if set(load_manifest(a.run)['record_ids'])!=set(ids): p.error('전체 Test 600건 실행이어야 합니다.')
pred={}
for rid in ids:
    r=load_result(a.run,'gemma_no_ocr',rid)
    if not r or r.get('status')!='ok' or r.get('label') not in LABELS: p.error(f'미완료 또는 오류: {rid}')
    pred[rid]=r['label']
if a.apply_public_labels: pred.update(_public_test_labels(ROOT/'data/docs/경계사례_20261013.csv'))
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_bytes(label_csv(pred))
print(a.output)
