# 모델과 문서형 학습 산출물

## 모델 가중치

모델: [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it)  
사용 배포본: [unsloth/gemma-4-31B-it-GGUF](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/tree/c1ac76e99d5513b141e8adde7288b85c3f9c32ec)

공개 사전학습 모델이므로 다음 두 가중치 파일은 ZIP에 포함하지 않는다. 가중치를 추가 학습하지 않았으며, 학습 산출물은 아래의 단계별 MD다.

| 파일 | 바이트 | SHA-256 |
|---|---:|---|
| `weights/gemma-4-31B-it-IQ4_NL.gguf` | 17,287,670,720 | `811dccf90edd3e0a5341c601473d42642f4acc238966326395d4f4c1289b2e92` |
| `weights/mmproj-BF16.gguf` | 1,200,726,496 | `7a4601b12ec680f706a7e0c7e1f78f579b1db64d485a9e352ee87d4b9daa45e4` |

패키지 루트에서 `python scripts/download_models.py`를 실행해 두 파일을 내려받는다. 다운로드에는 약 18.5GB의 전송량이 필요하다. 공식 공개 저장소 `unsloth/gemma-4-31B-it-GGUF`의 revision `c1ac76e99d5513b141e8adde7288b85c3f9c32ec`에 있는 두 파일의 LFS SHA-256과 로컬 파일 해시가 일치함을 확인했다.

## 사람이 검토한 단계별 판단 기준

- `guidelines/reflection_guidelines.md`: 적절·부적절 판정 재검토 기준.
- `guidelines/type_reflection_guidelines.md`: 실제 오류 유형을 후보에 포함하기 위한 기준.
- `guidelines/final_reflection_guidelines.md`: 고정된 후보 중 하나를 선택하기 위한 경계 기준.

이는 최종 제출 기반 실행 `20261105_080500_5cf95c`가 사용한 문서의 사본이다. 자동 생성 후 반복 비교 실험·사례 대조·인간 검토를 통해 선정한 상태를 보존한다. 실험 중인 추가 버전으로 바꾸지 않았다.

`frozen_config.json`에는 해당 실행의 프롬프트·설정을 보존했고, 모델·런타임 경로만 패키지 상대 경로로 바꾸었다. CLI는 실행 시 문서 해시를 확인하고 새 실행에 복사한다. 문서를 직접 고치면 새로운 문서 버전으로 취급되므로 기존 실행의 해시를 임의로 덮어쓰지 않는다.

평가용 [가이드라인 열람 Notebook](../notebooks/02_최종_판정가이드라인.ipynb)과 [브라우저용 문서](../docs/최종_판정가이드라인.html)에 세 파일의 전문을 동일하게 제시했다. Notebook은 실행 없이 읽을 수 있다.

## 런타임

실행 바이너리도 ZIP에 포함하지 않는다. `src/llama.cpp-source.tar.gz`와 `scripts/build_runtime.sh`로 같은 커밋을 현재 GPU에 맞게 빌드한다. CUDA·NVIDIA 드라이버는 별도 시스템 설치가 필요하다.
