[Instruction]
원본 이미지와 메타데이터를 보고 alt_text가 이미지의 역할에 맞는 정보를 전달하는지 판단하라. 출력은 "적절" 또는 "부적절" 중 하나다. 오류 유형은 분류하지 않는다.

[설명]
파이프라인 입력:
- image: 원본 이미지.

메타데이터 정의 (학습 데이터 README 데이터 사전):
- alt_text: 원본 대체텍스트 (원문 그대로). 빈 문자열 포함.
- page_title: 수집 페이지 제목.
- context_text: 이미지 인접 텍스트. 최대 500자.
- in_link: 링크·버튼 내부 여부. true, false.
- link_dest: 링크의 표시 텍스트. in_link=true일 때.
- heading_path: 문서 내 헤딩 경로.
- kogl_marker: 수집 페이지의 공공누리 표시 여부. true, false.

[Rule]
1. verdict는 "적절" 또는 "부적절" 중 하나로 출력한다.
2. reason에는 판정 근거, image_evidence에는 이미지 근거, context_evidence에는 문맥 근거를 쓴다. 각 근거는 한국어 350자 이내로 쓴다.
3. 아래 형식의 JSON 객체 하나만 반환한다.

[Output format]
{"verdict":"적절","reason":"판정 근거","image_evidence":"이미지에서 확인한 내용","context_evidence":"문맥에서 확인한 내용"}
또 다른 형식 예시:
{"verdict":"부적절","reason":"판정 근거","image_evidence":"이미지에서 확인한 내용","context_evidence":"문맥에서 확인한 내용"}
예시는 출력 형식이다. 실제 입력의 관찰 근거로 바꾸며, 두 객체 중 실제 판정에 해당하는 객체 하나만 출력한다.

[Output]
