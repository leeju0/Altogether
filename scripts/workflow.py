"""동일한 판정 엔진으로 Train 문서 축적 / Frozen 운영을 실행한다."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import acquire_worker_lock, create_run, load_records
from gemma import LABELS, REFLECTION_FROZEN, REFLECTION_TRAIN, default_config, validate_config
from run_ocr import execute_pipeline
FILES = ('reflection_guidelines.md', 'type_reflection_guidelines.md', 'final_reflection_guidelines.md')

def configuration(mode, documents, scope):
    if mode == 'frozen':
        config = json.loads((ROOT / 'model/frozen_config.json').read_text(encoding='utf-8'))
        config['reflection_guideline_source_run'] = documents.name
        for name, key in zip(FILES, ('reflection_guideline_sha256','type_reflection_guideline_sha256','final_reflection_guideline_sha256')):
            config[key] = hashlib.sha256((documents/name).read_bytes()).hexdigest()
        config['binary_only'] = scope == 'binary'
        config['candidate_only'] = scope == 'candidates'
        if scope != 'full': config.pop('decorative_gate', None)
        if scope == 'binary':
            config['type_reflection'] = config['final_reflection'] = False
            config.pop('type_reflection_guideline_sha256',None)
            config.pop('final_reflection_guideline_sha256',None)
        elif scope == 'candidates':
            config['final_reflection'] = False
            config.pop('final_reflection_guideline_sha256',None)
    else:
        if scope == 'candidates': raise ValueError('Train은 full 또는 binary 범위를 사용하세요.')
        config = default_config(4096,24576,binary_prompt='guideline_direct',
            model_variant='gemma4-31b-iq4nl',label_reflection=True,reflection_mode=REFLECTION_TRAIN,
            binary_only=scope=='binary',type_reflection=scope=='full',final_reflection=scope=='full',
            decorative_gate=scope=='full')
    for key in ('model_path','mmproj_path','server_path'):
        config[key] = str((ROOT / config[key]).resolve())
    validate_config(config)
    return config

def dataset(name):
    directory = ROOT / 'data' / ('test' if name == 'public109' else name)
    rows = load_records(directory)
    if name == 'public109':
        with (ROOT/'data/docs/경계사례_20261013.csv').open(encoding='utf-8-sig',newline='') as f:
            cases = [r for r in csv.DictReader(f) if r['분할']=='test']
        labels = {r['record_id']:r['데이터셋_라벨'] for r in cases}
        if len(cases)!=109 or len(labels)!=109 or not set(labels.values())<=set(LABELS):
            raise ValueError('공개 데이터셋 라벨 109건을 확인하세요.')
        rows = [{**r,'label':labels[r['record_id']]} for r in rows if r['record_id'] in labels]
        if len(rows)!=109: raise ValueError('공개 109건의 원본 레코드가 부족합니다.')
    return directory, rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('train','frozen'),default='frozen')
    p.add_argument('--dataset',choices=('train','test','public109'),default='test')
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--limit',type=int)
    p.add_argument('--scope',choices=('full','binary','candidates'),default='full')
    p.add_argument('--guidelines',type=Path,default=ROOT/'model/guidelines')
    p.add_argument('--seed-reviewed',action='store_true',help='Train 시작 문서로 검토된 MD 사본 사용')
    p.add_argument('--prepare-only',action='store_true',help='실행 입력만 만들고 GPU 추론은 시작하지 않음')
    args=p.parse_args()
    if args.gpu<0 or (args.limit is not None and args.limit<1): p.error('GPU는 0 이상, limit은 1 이상이어야 합니다.')
    directory,rows=dataset(args.dataset)
    if args.mode=='train' and any(r.get('label') not in LABELS for r in rows):
        p.error('Train에는 공개 정답이 있는 train 또는 public109 데이터가 필요합니다.')
    if args.limit: rows=rows[:args.limit]
    config=configuration(args.mode,args.guidelines.resolve(),args.scope)
    with acquire_worker_lock():
        run=create_run(directory,[r['record_id'] for r in rows],[],background='auto_gray',
                       device=f'gpu:{args.gpu}',gemma_config=config,pipeline_mode='gemma_only',records=rows)
        if args.mode=='frozen' or args.seed_reviewed:
            for name in FILES: shutil.copy2(args.guidelines/name,run/name)
        print(f'RUN_DIR={run}',flush=True)
        if not args.prepare_only: return execute_pipeline(run)
    return 0
if __name__=='__main__': sys.exit(main())
