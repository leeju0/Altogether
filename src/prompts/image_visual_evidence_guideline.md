[Instruction]
이미지와 웹 문맥을 보고 역할들과 짧은 이미지 설명만 작성하라. alt 품질과 필요한 정보는 판단하지 않는다.

[설명]
image는 원본 이미지다. page_title은 페이지 제목, context_text는 인접 표시 텍스트, in_link는 링크·버튼 내부 여부, link_dest는 수집된 링크 문자열(표시 문구 또는 URL), heading_path는 헤딩 경로, kogl_marker는 공공누리 표시다. URL·#none은 이미 낭독되는 표시 문구가 아니다. alt_text는 이 단계에 제공되지 않는다.

[Rule]
roles에는 해당하는 것을 하나 이상 고른다: 기능성(링크·버튼 안에서 목적지·동작 전달), 정보성(본문 정보 전달), 텍스트 포함 이미지(이미지 안 글자로 정보 전달), 장식성(제거해도 정보 손실 없음). 아이콘·로고 모양만으로 장식성이라 하지 말고 기능·정보를 본다. description은 원본과 문맥에서 확인한 짧은 설명이며 350자 이내다. visible_facts, context_quotes, required_information, 판정·품질 판단은 쓰지 않는다. JSON 하나만 반환한다.

[Output format]
{"roles":["정보성","텍스트 포함 이미지"],"description":"두 사람이 협약서를 들고 있는 안내 이미지"}

[Output]
