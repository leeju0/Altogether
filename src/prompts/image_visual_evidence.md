[Instruction]
이미지와 웹 문맥에서 alt 품질 판단에 필요한 시각 근거를 짧게 정리하라. alt 품질은 판정하지 않는다.

[설명]
image는 원본 이미지다. page_title은 페이지 제목, context_text는 인접 표시 텍스트, in_link는 링크·버튼 내부 여부, link_dest는 수집된 링크 문자열(표시 문구 또는 URL), heading_path는 헤딩 경로, kogl_marker는 공공누리 표시다. URL·#none은 이미 낭독되는 표시 문구가 아니다. alt_text는 이 단계에 제공되지 않는다. role은 주된 역할 하나(정보 전달, 기능 전달, 텍스트 전달, 자료 식별, 장식)다. visible_facts는 보이는 사실, context_quotes는 제공 필드의 실제 짧은 문구, required_information은 문맥만으로 이미 전달되지 않은 최소 필요 정보다.

[Rule]
각 배열은 0~3개, reason은 150자 이내다. URL·#none은 표시 문구가 아니다. 아이콘이 기하학적 모양이라는 이유만으로 장식이라 하지 말고 실제 기능·문맥을 본다. required_information에 모든 보이는 세부를 나열하지 않는다. JSON 하나만 반환한다.

[Output format]
{"role":"정보 전달","reason":"사진의 주요 장면","visible_facts":["두 사람이 협약서를 든 모습"],"context_quotes":["업무협약 체결"],"required_information":["주요 대상과 협약 장면"]}

[Output]
