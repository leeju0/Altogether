[Instruction]
image, 웹 문맥, visual_evidence를 대조해 alt_text가 역할에 필요한 정보를 실제로 전달하는지 판단하라. 오류 유형은 분류하지 않는다.

[설명]
image는 원본 이미지다. visual_evidence.roles는 관찰한 역할들(기능성, 정보성, 텍스트 포함 이미지, 장식성), description은 짧은 관찰 설명이며 오류 가능성이 있어 원본과 다시 비교한다. alt_text는 평가 원문, page_title은 페이지 제목, context_text는 인접 표시 텍스트, in_link는 링크·버튼 내부 여부, link_dest는 수집된 링크 문자열(표시 문구 또는 URL), heading_path는 헤딩 경로, kogl_marker는 공공누리 표시다.

[Rule]
기능성은 목적지·동작을, 정보성은 핵심 내용과 차트·도표의 핵심 수치·결론을 확인하며 인접 본문이 충분히 설명하면 요약 alt를 인정한다. 텍스트 포함 이미지는 이미지 안 내용을 실질적으로 alt에 전달해야 하며, 긴 전문은 핵심 문구와 상세 설명 위치 안내를 인정한다. 장식성은 빈 alt가 적절하고 내용성 alt는 부적절이다. 여러 역할이면 각 역할의 기준을 함께 검사한다. 파일명·내용 없는 표현, 이미지와 불일치는 부적절이다. checks는 최대 3개, quote는 실제 문구만 쓰며 없으면 ""다. JSON 하나만 반환한다.

[Output format]
{"verdict":"적절","reason":"역할에 필요한 정보를 전달한다","image_evidence":"이미지 근거","context_evidence":"문맥 근거","checks":[{"information":"주요 대상과 장면","alt_quote":"alt 실제 문구","context_quote":"문맥 실제 문구","status":"충족"}],"needs_review":false}

[Output]
