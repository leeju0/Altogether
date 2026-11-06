"""Open execution, review and score dashboards without starting inference."""
from __future__ import annotations
import sys

import argparse
import os
from pathlib import Path
import subprocess
import signal
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser


ROOT = Path(__file__).resolve().parent


def healthy(port):
    try:
        with urlopen(f'http://127.0.0.1:{port}/_stcore/health', timeout=1) as response:
            return response.status == 200 and response.read(32).strip() == b'ok'
    except (OSError, URLError):
        return False


def matches_dashboard(args, script, port):
    return (len(args) >= 5 and args[1:4] == ['-m', 'streamlit', 'run']
            and Path(args[4]).resolve() == ROOT / script
            and '--server.port' in args and args.index('--server.port') + 1 < len(args)
            and args[args.index('--server.port') + 1] == str(port))


def start_dashboard(script, port, restart=False):
    if restart and healthy(port):
        owners = []
        for entry in Path('/proc').iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                args = entry.joinpath('cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0')
                if matches_dashboard(args, script, port):
                    owners.append(int(entry.name))
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        if len(owners) != 1:
            raise RuntimeError(f'포트 {port}의 이 프로젝트 웹 서버를 특정할 수 없어 종료하지 않았습니다.')
        os.kill(owners[0], signal.SIGTERM)
        deadline = time.monotonic() + 10
        while healthy(port) and time.monotonic() < deadline:
            time.sleep(.2)
        if healthy(port):
            raise RuntimeError(f'포트 {port}의 웹 서버가 종료되지 않았습니다.')
    if healthy(port):
        return
    python = Path(sys.executable)
    if not python.exists():
        raise RuntimeError('프로젝트 .venv가 없습니다. README의 환경 설치 절차를 확인하세요.')
    logs = ROOT / '.logs'
    logs.mkdir(exist_ok=True)
    log_path = logs / f'{Path(script).stem}.log'
    with log_path.open('a', encoding='utf-8') as log:
        process = subprocess.Popen(
            [str(python), '-m', 'streamlit', 'run', str(ROOT / script),
             '--server.address', '127.0.0.1', '--server.port', str(port),
             '--server.headless', 'true', '--browser.gatherUsageStats', 'false'],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'{script} 시작 실패. 포트 {port}와 로그를 확인하세요: {log_path}')
        if healthy(port):
            return
        time.sleep(0.2)
    process.terminate()
    raise RuntimeError(f'{script} 시작 시간 초과. 로그: {log_path}')


def main():
    parser = argparse.ArgumentParser(description='웹 대시보드를 열고, Gemma 판정은 웹에서 실행합니다.')
    parser.add_argument('--no-browser', action='store_true', help='서버만 준비하고 브라우저 자동 열기 생략')
    parser.add_argument('--restart', action='store_true', help='이 프로젝트의 웹 서버 세 개를 최신 코드로 재시작 (모델 작업은 유지)')
    args = parser.parse_args()
    try:
        start_dashboard('run_app.py', 8501, restart=args.restart)
        start_dashboard('review_app.py', 8502, restart=args.restart)
        start_dashboard('score_app.py', 8503, restart=args.restart)
    except RuntimeError as exc:
        parser.exit(1, f'{exc}\n')
    print('실행 화면: http://localhost:8501')
    print('평가 화면: http://localhost:8502')
    print('CSV·F1 성능 비교: http://localhost:8503')
    print('웹에서 대상 이미지와 GPU를 선택한 뒤 [실행]을 누르세요.')
    print('이 명령은 웹만 준비합니다. 모델 판정은 실행 버튼을 누르면 시작합니다.')
    if not args.no_browser:
        try:
            webbrowser.open('http://localhost:8501', new=2)
        except webbrowser.Error:
            pass  # Printed URLs also work in headless/SSH environments.


if __name__ == '__main__':
    main()
