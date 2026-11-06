"""공식 제공 정답으로만 저장된 추론 결과를 재평가한다. GPU 불필요."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark import score_labels
from workflow import dataset

def evaluate(saved, split):
    _,rows=dataset(split)
    truth={r['record_id']:r['label'] for r in rows}
    items=json.loads(Path(saved).read_text(encoding='utf-8'))['items']
    valid=[r for r in items if r.get('status')=='ok' and r['record_id'] in truth]
    return score_labels([truth[r['record_id']] for r in valid],[r['label'] for r in valid])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=('train','public109'),default='train')
    p.add_argument('--saved',type=Path)
    a=p.parse_args(); saved=a.saved or ROOT/'results'/('train_frozen_saved.json' if a.dataset=='train' else 'test_frozen_saved.json')
    print(json.dumps(evaluate(saved,a.dataset),ensure_ascii=False,indent=2))
