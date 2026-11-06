"""경량 ZIP에 생략된 공개 모델 두 개를 고정 revision에서 내려받고 해시 확인."""
from pathlib import Path
import hashlib,json,urllib.request,os
ROOT=Path(__file__).resolve().parents[1]
REV='c1ac76e99d5513b141e8adde7288b85c3f9c32ec'
BASE=f'https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/resolve/{REV}/'

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    for item in json.loads((ROOT/'model/weights_manifest.json').read_text())['files']:
        target=ROOT/'model/weights'/item['name']; target.parent.mkdir(exist_ok=True)
        if target.is_file() and digest(target)==item['sha256']:
            print('확인:',target.name);continue
        temporary=target.with_suffix('.download')
        print('다운로드:',target.name,flush=True)
        with urllib.request.urlopen(BASE+item['name'],timeout=120) as response, temporary.open('wb') as out:
            while block:=response.read(8*1024*1024):out.write(block)
        if temporary.stat().st_size!=item['bytes'] or digest(temporary)!=item['sha256']:
            raise ValueError('배포 파일 해시 불일치: '+str(temporary))
        os.replace(temporary,target)
        print('완료:',target.name,flush=True)
if __name__=='__main__':main()
