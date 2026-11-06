[Instruction]
alt_text가 빈 문자열인 image에서 생략해도 필요한 정보·기능이 실제 문맥으로 전달되는지 판단하라. 오류 유형은 분류하지 않는다.

[설명]
image는 원본 이미지, visual_evidence는 앞 단계의 오류 가능성이 있는 근거이므로 원본과 다시 비교한다. alt_text는 평가 원문, page_title은 페이지 제목, context_text는 인접 표시 텍스트, in_link는 링크·버튼 내부 여부, link_dest는 수집된 링크 문자열(표시 문구 또는 URL), heading_path는 헤딩 경로, kogl_marker는 공공누리 표시다. URL·#none은 이미 낭독되는 표시 문구가 아니다.

[Rule]
실제 목적지 문구가 있는 빈 사진·그래픽 썸네일, 또는 정보 손실 없는 장식은 적절할 수 있으며 장면·풍경 상세를 요구하지 않는다. 이미지에만 있는 핵심 정보·기능이 빠지면 부적절이다. 제목만 있는 포스터는 날짜·대상·장소를 대신하지 못한다. URL·#none은 표시 문구가 아니다. 반복은 실제 문구가 일치할 때만 판단하고 보이지 않는 DOM을 상상하지 않는다. checks는 최대 3개, quote는 실제 문구만 쓰며 없으면 ""다. needs_review는 false다. JSON 하나만 반환한다.

[Output format]
{"verdict":"부적절","reason":"이미지에만 있는 핵심 정보가 생략됐다","image_evidence":"이미지 근거","context_evidence":"실제 문맥 문구","checks":[{"information":"필요 정보","alt_quote":"","context_quote":"","status":"미충족"}],"needs_review":false}

[Output]
