[Instruction]
이미지와 문맥을 보고 아래 역할의 기준에 따라 alt_text의 적절/부적절을 판단하라.

[설명]
image: 원본 이미지.
alt_text: 평가할 대체텍스트 원문(빈 문자열 포함). page_title: 페이지 제목. context_text: 인접 텍스트.
in_link: 링크·버튼 내부 여부. link_dest: 링크 표시 텍스트. heading_path: 헤딩 경로. kogl_marker: 페이지 공공누리 표시 여부.
판단할 역할: {{role}}
충족 기준: {{criterion}}

[Rule]
1. 기준에 필요한 정보·기능을 전달하면 적절, 핵심이 빠지거나 이미지와 불일치하면 부적절이다. 인접 본문이 충분히 설명하면 요약 alt도 인정한다.
2. 파일명·내용 없는 표현만 쓰거나, 인접 텍스트를 그대로 반복해 추가 정보가 전혀 없으면 부적절이다.
3. 길이·문체·오탈자 자체로 감점하지 않는다. 각 근거는 한국어 350자 이내로 쓰고 JSON 하나만 반환한다.

[Output format]
{"verdict":"적절","reason":"충족 또는 미충족 근거","image_evidence":"이미지 근거","context_evidence":"문맥 근거"}
verdict는 "적절" 또는 "부적절"이다.

[Output]
