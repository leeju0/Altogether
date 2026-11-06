[Instruction]
원본 image, 웹 문맥, visual_evidence를 대조해 alt_text가 필요한 정보를 실제로 전달하는지 판단하라. 오류 유형은 분류하지 않는다.

[설명]
image는 원본 이미지, visual_evidence는 앞 단계의 오류 가능성이 있는 근거이므로 원본과 다시 비교한다. alt_text는 평가 원문, page_title은 페이지 제목, context_text는 인접 표시 텍스트, in_link는 링크·버튼 내부 여부, link_dest는 수집된 링크 문자열(표시 문구 또는 URL), heading_path는 헤딩 경로, kogl_marker는 공공누리 표시다. URL·#none은 이미 낭독되는 표시 문구가 아니다.

[Rule]
visual_evidence의 required_information을 alt와 실제 문맥이 충족하면 적절이다. 정확한 대상명+"이미지"가 정보성 본문과 함께면 무의미로 보지 않고 충분할 수 있다. 실제 목적지 문구가 있는 빈 사진·그래픽 썸네일은 장면·풍경 상세를 요구하지 않는다. 제목만 있는 포스터는 날짜·대상·장소가 빠지면 불충분하고, 사진은 제목만 말한 alt와 실제 장면 설명을 구별한다. 반복은 실제 문구가 일치할 때만 판단하며 보이지 않는 DOM을 상상하지 않는다. 파일명·내용 없는 표현·이미지와 불일치는 부적절이다. checks는 최대 3개, quote는 실제 문구만 쓰며 없으면 ""다. 불확실, 확인불가, checks와 verdict 모순, 또는 부적절의 구체 관찰 결함 부재면 needs_review=true다. JSON 하나만 반환한다.

[Output format]
{"verdict":"적절","reason":"필요한 장면을 전달한다","image_evidence":"이미지 근거","context_evidence":"실제 문맥 문구","checks":[{"information":"주요 대상과 장면","alt_quote":"alt 실제 문구","context_quote":"문맥 실제 문구","status":"충족"}],"needs_review":false}

[Output]
