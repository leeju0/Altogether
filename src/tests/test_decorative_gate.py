import json

import pytest

import gemma
import run_ocr
import benchmark
import common


def _observation(**values):
    return {
        'visual_kind': 'simple_icon',
        'other_error': 'none',
        'pure_decorative_role': True,
        'decorative_pattern': 'error_status_icon',
        'confidence': 'high',
        'reason': '완전한 오류 문장 옆의 상태 아이콘이다.',
        **values,
    }


def test_decorative_gate_contract_hides_alt_and_truncates_observer_context(tmp_path):
    image = tmp_path / 'image.png'
    image.write_bytes(b'png')
    config = gemma.default_config(
        binary_prompt='guideline_direct', label_reflection=True, decorative_gate=True)
    data = {
        'alt_text': '에러 아이콘', 'page_title': ('가 \n' * 300),
        'context_text': '오류가 발생했습니다.' * 60, 'in_link': False,
        'link_dest': '나' * 600, 'heading_path': '다' * 600, 'kogl_marker': False,
    }

    router = gemma.request_payload('decorative_router', image, data, config)
    observer = gemma.request_payload('decorative_observer', image, data, config)
    router_fields = json.loads(router['messages'][1]['content'][1]['text'])
    observer_fields = json.loads(observer['messages'][1]['content'][1]['text'])

    assert config['decorative_gate'] is True
    assert {'decorative_router', 'decorative_observer'} <= set(config['prompts'])
    assert 'alt_text' not in router_fields
    assert router['response_format']['json_schema']['name'] == 'binary_verdict'
    assert router['max_tokens'] == 512 and observer['max_tokens'] == 512
    assert observer_fields['alt_text'] == '에러 아이콘'
    assert observer_fields['page_title'] == data['page_title'][:500]
    assert all(len(observer_fields[key]) == 500
               for key in ('page_title', 'context_text', 'link_dest', 'heading_path'))
    assert observer['response_format']['json_schema']['name'] == 'decorative_observation'


@pytest.mark.parametrize(('observation', 'data', 'expected'), [
    (_observation(), {'alt_text': '에러 아이콘', 'context_text': '오류가 발생했습니다.',
                      'in_link': False}, True),
    (_observation(other_error='information_omission'),
     {'alt_text': '신청 절차', 'context_text': '신청 절차', 'in_link': False}, False),
    (_observation(visual_kind='identity_symbol', pure_decorative_role=False,
                  decorative_pattern='redundant_identity_visual'),
     {'alt_text': '기관 로고', 'context_text': '기관', 'in_link': True}, False),
    (_observation(visual_kind='identity_symbol', pure_decorative_role=False,
                  decorative_pattern='redundant_identity_visual'),
     {'alt_text': '기관 로고', 'context_text': '기관', 'in_link': False}, False),
    (_observation(visual_kind='identity_symbol', pure_decorative_role=True,
                  decorative_pattern='redundant_identity_visual'),
     {'alt_text': '기관 로고', 'context_text': '기관', 'in_link': False}, False),
    (_observation(visual_kind='simple_icon', other_error='mismatch',
                  pure_decorative_role=False, decorative_pattern='none'),
     {'alt_text': '비', 'context_text': '비', 'in_link': False}, False),
    (_observation(visual_kind='photo', pure_decorative_role=False,
                  decorative_pattern='none'),
     {'alt_text': '창계숭절사 이미지', 'context_text': '추천 관광지 창계숭절사 자세히보기',
      'in_link': False}, False),
    (_observation(visual_kind='photo', pure_decorative_role=False,
                  decorative_pattern='none'),
     {'alt_text': '행사 현장 사진', 'context_text': '행사 소식', 'in_link': False}, False),
    (_observation(visual_kind='identity_symbol', pure_decorative_role=False,
                  decorative_pattern='none'),
     {'alt_text': '위텍스봇', 'page_title': '분야별챗봇모음',
      'context_text': '위텍스봇 지방세 상담', 'in_link': False}, False),
    (_observation(visual_kind='identity_symbol', pure_decorative_role=False,
                  decorative_pattern='none'),
     {'alt_text': '국세청홈텍스', 'page_title': '분야별챗봇모음',
      'context_text': '국세청 홈텍스봇', 'in_link': False}, False),
    (_observation(visual_kind='theme_visual', pure_decorative_role=False,
                  decorative_pattern='theme_background'),
     {'alt_text': '개인정보의 보유 기간',
      'context_text': '관련 법령에 따라 개인정보를 보유합니다.', 'in_link': False}, False),
    (_observation(visual_kind='theme_visual', pure_decorative_role=True,
                  decorative_pattern='theme_background'),
     {'alt_text': '사람 일러스트 4명',
      'context_text': '언제 어디서나 누리는 복지 서비스', 'in_link': False}, True),
    (_observation(visual_kind='photo', other_error='mismatch',
                  pure_decorative_role=True, decorative_pattern='promotional_atmosphere'),
     {'alt_text': '문화투어', 'context_text': '문화투어 테마별 투어',
      'in_link': False}, False),
    (_observation(),
     {'alt_text': '전시 해설 예약', 'context_text': '전시 해설 예약',
      'in_link': True, 'link_dest': '전시 해설 예약'}, False),
    (_observation(decorative_pattern='auxiliary_link_icon'),
     {'alt_text': '어린이박물관 관람 예약 바로가기', 'context_text': '관람 예약',
      'in_link': True, 'link_dest': 'M0104010000.do?schM=child&act=intro'}, False),
    (_observation(),
     {'alt_text': '자동으로 수집하는 장치의 설치·운영 및 거부방법',
      'context_text': '자동으로 수집하는 장치의 설치·운영 및 거부방법',
      'in_link': False}, False),
    (_observation(decorative_pattern='auxiliary_link_icon'),
     {'alt_text': 'NEW', 'context_text': '구정안내/백서',
      'in_link': False}, False),
    (_observation(visual_kind='theme_visual', pure_decorative_role=True,
                  decorative_pattern='theme_background', confidence='low'),
     {'alt_text': '주제 이미지', 'context_text': '주제', 'in_link': False}, False),
])
def test_decorative_observation_decision_matches_validated_experiment(observation, data, expected):
    assert gemma.decorative_observation_decision(observation, data) is expected


def test_decorative_observation_response_is_strictly_validated():
    config = gemma.default_config(
        binary_prompt='guideline_direct', label_reflection=True, decorative_gate=True)
    raw = {'choices': [{'finish_reason': 'stop', 'message': {
        'content': json.dumps(_observation(pure_decorative_role='true'), ensure_ascii=False)}}]}

    with pytest.raises(ValueError, match='장식오용형 구조화 관찰'):
        gemma.parse_response(raw, 'decorative_observer', config)


def test_decorative_gate_requires_full_reflection_pipeline():
    with pytest.raises(ValueError, match='장식오용형 전용 게이트'):
        gemma.default_config(binary_prompt='guideline_direct', decorative_gate=True)
    with pytest.raises(ValueError, match='장식오용형 전용 게이트'):
        gemma.default_config(binary_prompt='guideline_direct', label_reflection=True,
                             binary_only=True, decorative_gate=True)


class _DecorativeServer:
    def __init__(self, router, observer):
        self.router = router
        self.observer = observer
        self.calls = []

    def infer_decorative_router(self, image_path, data, audit_path, slot_id=1):
        self.calls.append(('router', slot_id, audit_path.name))
        return self.router, {'stage': 'decorative_router', 'usage': {},
                             'elapsed_seconds': .1, 'raw_path': 'raw/router.json'}

    def infer_decorative_observer(self, image_path, data, audit_path, slot_id=1):
        self.calls.append(('observer', slot_id, audit_path.name))
        return self.observer, {'stage': 'decorative_observer', 'usage': {},
                               'elapsed_seconds': .2, 'raw_path': 'raw/observer.json'}


def test_decorative_route_runs_observer_only_for_nonempty_alt_router_pass(tmp_path):
    server = _DecorativeServer(
        {'verdict': '적절', 'reason': '후보', 'image_evidence': '아이콘',
         'context_evidence': '오류 문장'}, _observation())
    route = run_ocr._decorative_route(
        server, tmp_path / 'image.png',
        {'alt_text': '에러 아이콘', 'context_text': '오류가 발생했습니다.', 'in_link': False},
        tmp_path / 'raw/gemma_no_ocr/a.json')

    assert route['selected'] is True
    assert [call[0] for call in server.calls] == ['router', 'observer']
    assert [call['stage'] for call in route['calls']] == [
        'decorative_router', 'decorative_observer']

    assert run_ocr._decorative_route(
        server, tmp_path / 'image.png', {'alt_text': ''},
        tmp_path / 'raw/gemma_no_ocr/b.json') is None


def test_decorative_gate_result_is_a_complete_terminal_label(tmp_path):
    config = gemma.default_config(
        binary_prompt='guideline_direct', label_reflection=True, decorative_gate=True)
    initial = {'verdict': '적절', 'reason': '최초',
               'image_evidence': '이미지', 'context_evidence': '문맥'}
    first = {'stage': 'classify_no_ocr', 'usage': {}, 'elapsed_seconds': .3,
             'raw_path': 'raw/gemma_no_ocr/a.json'}
    router = {'verdict': '적절', 'reason': '장식 후보',
              'image_evidence': '상태 아이콘', 'context_evidence': '완전한 오류 문장'}
    observation = _observation()
    route = {'selected': True, 'router': router, 'observation': observation,
             'calls': [
                 {'stage': 'decorative_router', 'usage': {}, 'elapsed_seconds': .1,
                  'raw_path': 'raw/gemma_no_ocr/a.decorative_router.json'},
                 {'stage': 'decorative_observer', 'usage': {}, 'elapsed_seconds': .2,
                  'raw_path': 'raw/gemma_no_ocr/a.decorative_observer.json'},
             ]}

    value, stats = run_ocr._decorative_gate_result(config, initial, first, route)

    assert value['verdict'] == '부적절' and value['label'] == '장식오용형'
    assert value['binary_decision'] == initial
    assert value['reflection_decision']['verdict'] == '부적절'
    assert value['reflection_skipped'] == 'decorative_structured_gate'
    assert value['type_classification']['matched_types'][0]['label'] == '장식오용형'
    assert value['final_decision']['label'] == '장식오용형'
    assert value['decorative_gate_selected'] is True
    assert [call['stage'] for call in stats['calls']] == [
        'classify_no_ocr', 'decorative_router', 'decorative_observer']


def _raw(value):
    return {'choices': [{'finish_reason': 'stop', 'message': {
        'content': json.dumps(value, ensure_ascii=False)}}]}


def test_benchmark_audits_decorative_terminal_route(tmp_path):
    image = tmp_path / 'inputs/a.png'
    image.parent.mkdir()
    image.write_bytes(b'png')
    raw_dir = tmp_path / 'raw/gemma_no_ocr'
    raw_dir.mkdir(parents=True)
    config = gemma.default_config(
        binary_prompt='guideline_direct', label_reflection=True,
        type_reflection=True, final_reflection=True, decorative_gate=True)
    metadata = {'alt_text': '에러 아이콘', 'page_title': '오류',
                'context_text': '오류가 발생했습니다.', 'in_link': False,
                'link_dest': '', 'heading_path': '오류', 'kogl_marker': False}
    initial = {'verdict': '적절', 'reason': '최초',
               'image_evidence': '아이콘', 'context_evidence': '오류 문장'}
    router = {'verdict': '적절', 'reason': '장식 후보',
              'image_evidence': '상태 아이콘', 'context_evidence': '완전한 오류 문장'}
    observation = _observation()
    for stage, suffix, value in (
            ('classify_no_ocr', '', initial),
            ('decorative_router', '.decorative_router', router),
            ('decorative_observer', '.decorative_observer', observation)):
        path = raw_dir / f'a{suffix}.json'
        common.write_json(path.with_suffix('.request.json'), gemma.audit_request(
            gemma.request_payload(stage, image, metadata, config), image))
        common.write_json(path, _raw(value))
    route = {'selected': True, 'router': router, 'observation': observation,
             'calls': [
                 {'stage': 'decorative_router', 'usage': {}, 'elapsed_seconds': .1,
                  'raw_path': 'raw/gemma_no_ocr/a.decorative_router.json'},
                 {'stage': 'decorative_observer', 'usage': {}, 'elapsed_seconds': .2,
                  'raw_path': 'raw/gemma_no_ocr/a.decorative_observer.json'},
             ]}
    result, _ = run_ocr._decorative_gate_result(
        config, initial,
        {'stage': 'classify_no_ocr', 'usage': {}, 'elapsed_seconds': .3,
         'raw_path': 'raw/gemma_no_ocr/a.json'}, route)
    result['input_path'] = 'inputs/a.png'
    common.write_json(tmp_path / 'records.json', [
        {'record_id': 'a', 'label': '장식오용형', **metadata}])

    assert benchmark._single_audit_error(
        tmp_path, 'gemma_no_ocr', 'a', result, config) is None
