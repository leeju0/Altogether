[Instruction]
alt_text가 빈 문자열인 image에서 visual_evidence 역할에 필요한 정보·기능이 실제 문맥으로 전달되는지 판단하라. 오류 유형은 분류하지 않는다.

[설명]
image는 원본 이미지다. visual_evidence.roles는 관찰한 역할들(기능성, 정보성, 텍스트 포함 이미지, 장식성), description은 짧은 관찰 설명이며 오류 가능성이 있어 원본과 다시 비교한다. alt_text는 평가 원문, page_title은 페이지 제목, context_text는 인접 표시 텍스트, in_link는 링크·버튼 내부 여부, link_dest는 수집된 링크 문자열(표시 문구 또는 URL), heading_path는 헤딩 경로, kogl_marker는 공공누리 표시다.

[Rule]
1. context_text·link_dest의 실제 문구가 이미지의 정보·목적지를 이미 전달하는지 먼저 확인한다. URL이나 #none을 표시 문구로 간주하거나 보이지 않는 주변 문장을 만들어내지 않는다.
2. checks의 information에는 이미지를 낭독하지 않으면 놓칠 수 있는 구체적인 핵심 정보·기능을 쓴다. context_quote에는 그 정보와 대조한 실제 주변·링크 문구를 인용하고, 없으면 ""를 쓴다. status는 그 문구가 해당 정보를 충분히 대신하면 "충족", 대신하지 못하면 "미충족", 확인할 수 없으면 "확인불가"다.
3. 부적절이면 원본에서 확인한 핵심 정보·기능과 문맥이 대신하지 못하는 이유를 reason에 명시한다. alt가 비었거나 글자·사물이 보인다는 사실만으로 부적절로 판단하지 않는다. 적절이면 정보·목적지를 대신하는 실제 문구 또는 정보 손실 없는 장식인 근거를 쓴다. 주변 문구가 있다는 사실만으로 정보 누락을 허용하지 않는다.
4. checks는 최대 3개이며 alt_quote는 ""다. 정보 손실 없는 장식은 checks가 비어도 되지만 빈 목록을 곧바로 적절로 해석하지 않고 verdict를 명시한다. needs_review는 false다. JSON 하나만 반환한다.

[Output format]
{"verdict":"적절","reason":"실제 문구가 이미지의 정보를 대신하는 근거","image_evidence":"원본에서 확인한 핵심 정보·기능","context_evidence":"주변·링크 문구의 대체 여부","checks":[{"information":"생략 시 놓칠 수 있는 핵심 정보·기능","alt_quote":"","context_quote":"입력에 있는 실제 문구","status":"충족"}],"needs_review":false}
verdict는 "적절" 또는 "부적절"이다. 형식 예시를 복사하지 말고 실제 관찰·인용·대체 여부로 채운다. 부적절이면 누락된 핵심 정보와 문맥이 대체하지 못하는 이유를 기록한다.

[Output]
