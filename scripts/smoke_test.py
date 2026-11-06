"""GPU 없이 데이터·문서·프롬프트·평가 계산의 연결을 점검한다."""
from pathlib import Path
import json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from common import load_records,resolve_image,prepare_image
from gemma import LABELS,default_config
from workflow import configuration,dataset,FILES
from benchmark import score_labels,read_label_csv

def main():
    train=load_records(ROOT/'data/train');test=load_records(ROOT/'data/test')
    assert len(train)==2400 and len(test)==600
    assert all(r['label'] in LABELS for r in train)
    assert all(not r.get('label') for r in test)
    assert len(dataset('public109')[1])==109
    for split, rows in [('train',train),('test',test)]:
        assert all(resolve_image(ROOT/'data'/split,r).is_file() for r in rows)
    image,meta=prepare_image(resolve_image(ROOT/'data/test',test[0]),'auto_gray')
    assert image.mode=='RGB'
    config=json.loads((ROOT/'model/frozen_config.json').read_text(encoding='utf-8'))
    for key,name in zip(('reflection_guideline_sha256','type_reflection_guideline_sha256','final_reflection_guideline_sha256'),FILES):
        assert config[key]==hashlib.sha256((ROOT/'model/guidelines'/name).read_bytes()).hexdigest()
    assert score_labels(['적절','불충분형'],['적절','불충분형'])['binary_f1']==1
    assert len(read_label_csv((ROOT/'results/official_submission.csv').read_bytes()))==600
    cfg_train=default_config(4096,24576,binary_prompt='guideline_direct',label_reflection=True,type_reflection=True,final_reflection=True)
    assert cfg_train['reflection_training_route']=='initial_binary_errors_only_v1'
    assert config['reflection_mode']=='frozen_guideline_v1'
    print('PASS: Train 2400 / Test 600 / 공개 109 · 원본 이미지 · 세 MD 해시 · Train/Frozen 설정 · 제출 CSV')
if __name__=='__main__':main()
