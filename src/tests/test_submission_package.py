"""제출 경로·Train/Frozen 구분·CSV 계약은 GPU 없이 검사한다."""
from pathlib import Path
import copy,json,sys,subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import workflow
import gemma

def test_modes_and_scopes_keep_contracts(monkeypatch):
    validate=gemma.validate_config
    def schema_only(config):
        runtime=copy.deepcopy(config)
        for k in ('model_path','mmproj_path','server_path'):runtime[k]=sys.executable
        validate(runtime)
    monkeypatch.setattr(workflow,'validate_config',schema_only)
    for scope in ('full','binary','candidates'):
        c=workflow.configuration('frozen',ROOT/'model/guidelines',scope)
        assert c['reflection_mode']=='frozen_guideline_v1'
        assert c['binary_only']==(scope=='binary')
        assert c['candidate_only']==(scope=='candidates')
        assert c['model_path'].startswith(str(ROOT))
    c=workflow.configuration('train',ROOT/'model/guidelines','full')
    assert c['reflection_training_route']=='initial_binary_errors_only_v1'
    assert 'reflection_guideline_sha256' not in c

def test_train_rejects_unlabeled_test_before_loading_model():
    r=subprocess.run([sys.executable,str(ROOT/'scripts/workflow.py'),'--mode','train','--dataset','test','--limit','1','--prepare-only'],capture_output=True,text=True)
    assert r.returncode!=0
    assert '공개 정답' in r.stderr

def test_export_contract_from_saved_results(tmp_path):
    from common import load_records,write_json
    from benchmark import read_label_csv
    rows=load_records(ROOT/'data/test');run=tmp_path/'run'
    write_json(run/'manifest.json',{'record_ids':[r['record_id'] for r in rows]})
    saved=json.loads((ROOT/'results/test_frozen_saved.json').read_text())['items']
    for r in saved:write_json(run/'results/gemma_no_ocr'/f'{r["record_id"]}.json',r)
    output=tmp_path/'submission.csv'
    r=subprocess.run([sys.executable,str(ROOT/'scripts/export_csv.py'),'--run',str(run),'--output',str(output)],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    parsed=read_label_csv(output.read_bytes())
    assert len(parsed)==600
    assert parsed=={r['record_id']:r['label'] for r in saved}
