import json
from pathlib import Path

import pytest

import gemma


def test_model_variant_selects_matching_model_and_mmproj():
    config = gemma.default_config(model_variant='gemma4-31b-iq4nl')
    assert config['model_path'].endswith('model/weights/gemma-4-31B-it-IQ4_NL.gguf')
    assert config['mmproj_path'].endswith('model/weights/mmproj-BF16.gguf')
    assert config['server_path'].endswith('model/runtime/llama-server')


def two_stage_answer(verdict):
    return {'verdict': verdict, 'reason': '판정 근거', 'image_evidence': '이미지 근거',
            'context_evidence': '문맥 근거'}


def wrap_response(value):
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value, ensure_ascii=False)}}]}


def v5_config():
    return {**gemma.default_config(binary_prompt='criteria'), 'classification_mode': 'explicit_verdict_multilabel_v1',
            'prompt_version': 'v5-explicit-verdict-types'}


def test_binary_prompt_comparison_changes_only_first_stage_instructions():
    basic = gemma.default_config(binary_prompt='basic')
    criteria = gemma.default_config(binary_prompt='criteria')
    assert basic['prompt_version'] == 'v7-two-stage-basic+image-metadata-v1'
    assert criteria['prompt_version'] == 'v8-two-stage-role-criteria+image-metadata-v1'
    assert criteria == gemma.default_config(binary_prompt='criteria')
    assert 'clean' not in basic['prompts'] and 'clean' not in criteria['prompts']
    assert basic['prompts']['types'] == criteria['prompts']['types']
    assert not any(label in criteria['prompts']['classify'] for label in gemma.LABELS[1:])
    start = criteria['prompts']['classify'].index('[판정 기준]')
    end = criteria['prompts']['classify'].index('[Rule]')
    assert criteria['prompts']['classify'][:start] + criteria['prompts']['classify'][end:] == basic['prompts']['classify']
    for config in (basic, criteria):
        config.pop('prompt_version')
        config['prompts'].pop('classify')
    assert basic == criteria
    with pytest.raises(ValueError):
        gemma.default_config(binary_prompt='unknown')


def test_default_two_stage_payloads_keep_verdict_and_types_contracts_separate(tmp_path):
    image = tmp_path / 'input.png'; image.write_bytes(b'png')
    config = gemma.default_config(binary_prompt='criteria')
    assert config['classification_mode'] == 'two_stage_verdict_types_v1'
    binary = gemma.request_payload('classify', image, {}, config)
    types = gemma.request_payload('types', image, {}, config)
    assert binary['messages'][0]['content'] == config['prompts']['classify']
    assert types['messages'][0]['content'] == config['prompts']['types']
    assert set(binary['response_format']['json_schema']['schema']['properties']) == {
        'verdict', 'reason', 'image_evidence', 'context_evidence'}
    assert 'verdict' not in types['response_format']['json_schema']['schema']['properties']
    assert types['response_format']['json_schema']['schema']['properties']['matched_types']['minItems'] == 1


def test_two_stage_no_ocr_omits_ocr_from_both_calls(tmp_path):
    image = tmp_path / 'input.png'; image.write_bytes(b'png')
    data = {'alt_text': '대체 텍스트', 'cleaned_ocr_text': 'PRIVATE_OCR'}
    config = gemma.default_config(binary_prompt='criteria')
    for stage in ('classify_no_ocr', 'types_no_ocr'):
        text = gemma.request_payload(stage, image, data, config)['messages'][1]['content'][1]['text']
        assert 'PRIVATE_OCR' not in text


def test_two_stage_parsers_reject_empty_types_and_keep_binary_independent():
    config = gemma.default_config(binary_prompt='criteria')
    wrap = lambda value: {'choices': [{'finish_reason': 'stop', 'message': {'content': value}}]}
    binary = gemma.parse_response(wrap(json.dumps(two_stage_answer('부적절'))), 'classify', config)
    assert binary == two_stage_answer('부적절')
    with pytest.raises(ValueError, match='matched_types'):
        gemma.parse_response(wrap(json.dumps({'matched_types': [], 'reason': 'x', 'image_evidence': 'y',
                                              'context_evidence': 'z'})), 'types', config)


def test_type_reflection_fallback_merges_duplicate_candidate_labels():
    config = gemma.default_config(binary_prompt='guideline_direct', label_reflection=True,
                                  type_reflection=True)
    answer = {
        'matched_types': [
            {'label': '중복형', 'reason': '최초 근거'},
            {'label': '불충분형', 'reason': '최초 근거'},
            {'label': '불충분형', 'reason': 'Reflection으로 보강한 근거'},
        ],
        'reason': '후보 종합 근거',
        'image_evidence': '이미지 근거',
        'context_evidence': '문맥 근거',
    }

    parsed = gemma.parse_response(wrap_response(answer), 'type_reflection_fallback', config)

    assert parsed['matched_types'] == [
        {'label': '중복형', 'reason': '최초 근거'},
        {'label': '불충분형', 'reason': 'Reflection으로 보강한 근거'},
    ]


def test_two_stage_rejects_invalid_priority_and_missing_saved_prompt(tmp_path):
    image = tmp_path / 'input.png'; image.write_bytes(b'png')
    config = gemma.default_config(binary_prompt='criteria')
    with pytest.raises(ValueError, match='우선순위'):
        gemma.request_payload('classify', image, {}, {**config, 'label_priority': list(gemma.ERROR_PRIORITY[:-1])})
    for name in ('model_path', 'mmproj_path', 'server_path'):
        path = tmp_path / name; path.write_bytes(b'fixture'); config[name] = str(path)
    config['prompts'] = {**config['prompts'], 'types': '   '}
    with pytest.raises(ValueError, match='프롬프트'):
        gemma.validate_config(config)


def test_server_two_stage_skips_types_for_appropriate_and_audits_binary(tmp_path, monkeypatch):
    image = tmp_path / 'input.png'; image.write_bytes(b'png')
    server = gemma.GemmaServer(gemma.default_config(binary_prompt='criteria'), 'gpu:2', tmp_path)
    replies = [wrap_response(two_stage_answer('적절'))]
    monkeypatch.setattr(server, 'http', lambda *args: replies.pop(0))
    value, stats = server.infer('classify', image, {'alt_text': '검색'}, tmp_path / 'raw/gemma/a.json')
    assert value['label'] == '적절' and value['type_classification'] is None
    assert stats['raw_paths'] == ['raw/gemma/a.json']
    assert (tmp_path / 'raw/gemma/a.request.json').is_file()


def test_server_two_stage_runs_types_and_uses_priority(tmp_path, monkeypatch):
    image = tmp_path / 'input.png'; image.write_bytes(b'png')
    server = gemma.GemmaServer(gemma.default_config(binary_prompt='criteria'), 'gpu:2', tmp_path)
    replies = [wrap_response(two_stage_answer('부적절')), wrap_response({
        'matched_types': [{'label': '중복형', 'reason': '중복'}, {'label': '파일명형', 'reason': '파일명'}],
        'reason': '유형 근거', 'image_evidence': '이미지', 'context_evidence': '문맥'})]
    monkeypatch.setattr(server, 'http', lambda *args: replies.pop(0))
    value, stats = server.infer('classify_no_ocr', image, {'alt_text': '검색', 'cleaned_ocr_text': 'SECRET'},
                                tmp_path / 'raw/gemma_no_ocr/a.json')
    assert value['label'] == '파일명형' and value['binary_decision']['verdict'] == '부적절'
    assert stats['raw_paths'] == ['raw/gemma_no_ocr/a.json', 'raw/gemma_no_ocr/a.types.json']
    assert (tmp_path / 'raw/gemma_no_ocr/a.types.request.json').is_file()


def test_server_two_stage_preserves_binary_audit_when_types_are_invalid(tmp_path, monkeypatch):
    image = tmp_path / 'input.png'; image.write_bytes(b'png')
    server = gemma.GemmaServer(gemma.default_config(binary_prompt='criteria'), 'gpu:2', tmp_path)
    replies = [wrap_response(two_stage_answer('부적절')), wrap_response({
        'matched_types': [], 'reason': '유형 근거', 'image_evidence': '이미지', 'context_evidence': '문맥'})]
    monkeypatch.setattr(server, 'http', lambda *args: replies.pop(0))
    with pytest.raises(ValueError, match='matched_types'):
        server.infer('classify', image, {'alt_text': '검색'}, tmp_path / 'raw/gemma/a.json')
    assert (tmp_path / 'raw/gemma/a.json').is_file()
    assert (tmp_path / 'raw/gemma/a.types.json').is_file()


def test_requests_exclude_ground_truth_and_identifiers(tmp_path):
    image = tmp_path / 'input.png'
    image.write_bytes(b'png fixture')
    row = {'label': 'SECRET_GROUND_TRUTH', 'record_id': 'SECRET_ID',
           'page_url': 'SECRET_URL', 'alt_text': '검색', 'in_link': True,
           'link_dest': '검색', 'kogl_marker': True}
    config = gemma.default_config(binary_prompt='criteria')
    payload = gemma.request_payload('classify', image, row, config)
    serialized = json.dumps(payload, ensure_ascii=False)
    assert all(secret not in serialized for secret in ('SECRET_GROUND_TRUTH', 'SECRET_ID', 'SECRET_URL'))
    assert 'data:image/png;base64,' in serialized
    assert '검색' in serialized and 'kogl_marker' in serialized


def test_current_requests_never_include_ocr_even_if_supplied(tmp_path):
    image = tmp_path / 'input.png'
    image.write_bytes(b'png')
    data = {'alt_text': '검색', 'context_text': '검색창', 'cleaned_ocr_text': 'PRIVATE_OCR',
            'label': 'PRIVATE_LABEL', 'record_id': 'PRIVATE_ID'}
    config = gemma.default_config(binary_prompt='criteria')
    with_ocr = gemma.request_payload('classify', image, data, config)
    without = gemma.request_payload('classify_no_ocr', image, data, config)
    assert with_ocr == without  # prompts, image bytes, schema, decoding and metadata identical
    assert 'PRIVATE_' not in json.dumps(without)
    answer = {'label': '적절', 'reason': '기능 설명', 'image_evidence': '돋보기', 'context_evidence': '검색'}
    assert gemma.parse_response({'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer)}}]},
                                'classify_no_ocr') == answer




def test_invalid_or_truncated_model_output_is_not_a_label():
    answer = {'label': '적절', 'reason': '버튼의 동작을 설명함', 'image_evidence': '돋보기', 'context_evidence': '검색 링크'}
    envelope = lambda content, finish='stop': {'choices': [{'finish_reason': finish, 'message': {'content': content}}]}
    assert gemma.parse_response(envelope(json.dumps(answer)), 'classify') == answer
    for text, finish in [(json.dumps(answer), 'length'), ('not JSON', 'stop'), (json.dumps({**answer, 'label': '좋음'}), 'stop')]:
        with pytest.raises(ValueError):
            gemma.parse_response(envelope(text, finish), 'classify')


def test_legacy_multiple_types_use_saved_priority_independent_of_model_order():
    from itertools import permutations

    config = {**gemma.default_config(binary_prompt='criteria'), 'classification_mode': 'multilabel_priority_v1'}
    expected_order = ['파일명형', '무관형', '무의미형', '장식오용형', '중복형', '불충분형']
    assert config['label_priority'] == expected_order
    for labels, winner in [([], '적절'), (['불충분형', '중복형', '장식오용형'], '장식오용형'),
                           (['중복형', '무의미형', '무관형', '파일명형'], '파일명형')]:
        for order in permutations(labels):
            answer = {'matched_types': [{'label': label, 'reason': label + ' 근거'} for label in order],
                      'reason': '모델의 종합 평가', 'image_evidence': '이미지 근거', 'context_evidence': '문맥 근거'}
            raw = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer)}}]}
            for stage in ['classify', 'classify_no_ocr']:
                result = gemma.parse_response(raw, stage, config)
                assert result['label'] == winner
                assert result['matched_types'] == answer['matched_types']
                assert result['reason'] == (winner + ' 근거' if labels else answer['reason'])
                assert result['assessment_reason'] == answer['reason']
                assert result['decision_rule'] == {'name': 'error-priority-v1', 'priority': expected_order}


@pytest.mark.parametrize('types', [None, '중복형', ['중복형'], [{'label': '적절', 'reason': '근거'}],
                                  [{'label': '보류', 'reason': '근거'}], [{'label': '중복형', 'reason': ''}],
                                  [{'label': '중복형', 'reason': 'x' * 351}],
                                  [{'label': '중복형', 'reason': '근거'}] * 2])
def test_invalid_multiple_types_never_become_appropriate(types):
    answer = {'verdict': '부적절', 'matched_types': types, 'reason': '종합 평가', 'image_evidence': '이미지', 'context_evidence': '문맥'}
    raw = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer)}}]}
    with pytest.raises(ValueError):
        gemma.parse_response(raw, 'classify', gemma.default_config(binary_prompt='criteria'))


def test_new_schema_and_legacy_contract_are_separate(tmp_path):
    image = tmp_path / 'input.png'
    image.write_bytes(b'png')
    config = v5_config()
    request = gemma.request_payload('classify', image, {}, config)
    assert 'matched_types' in request['response_format']['json_schema']['schema']['required']
    assert 'label' not in request['response_format']['json_schema']['schema']['properties']
    assert 'label_priority' not in json.loads(request['messages'][1]['content'][1]['text'])
    legacy = {key: value for key, value in config.items() if key not in ('classification_mode', 'label_priority')}
    legacy['prompt_version'] = 'v3-koaltq-v0.9'
    legacy['prompts'] = {'classify': 'saved single-label prompt', 'clean': 'saved cleaner'}
    request = gemma.request_payload('classify_no_ocr', image, {}, legacy)
    assert 'label' in request['response_format']['json_schema']['schema']['required']
    assert request['messages'][0]['content'] == 'saved single-label prompt'
    single = {'label': '적절', 'reason': '근거', 'image_evidence': '이미지', 'context_evidence': '문맥'}
    raw = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(single)}}]}
    assert gemma.parse_response(raw, 'classify', legacy) == single
    with pytest.raises(ValueError):
        gemma.parse_response(raw, 'classify', config)


def test_priority_stays_in_saved_config_and_never_reaches_model(tmp_path):
    image = tmp_path / 'input.png'
    image.write_bytes(b'png')
    config = v5_config()
    changed = {**config, 'label_priority': list(reversed(config['label_priority']))}
    assert gemma.request_payload('classify', image, {}, config) == gemma.request_payload('classify', image, {}, changed)
    answer = {'verdict': '부적절', 'matched_types': [{'label': label, 'reason': label + ' 근거'} for label in ['중복형', '장식오용형']],
              'reason': '종합 평가', 'image_evidence': '이미지', 'context_evidence': '문맥'}
    raw = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer)}}]}
    assert gemma.parse_response(raw, 'classify', config)['label'] == '장식오용형'
    assert gemma.parse_response(raw, 'classify', changed)['label'] == '중복형'
    for priority in [None, [], ['중복형'] * 6, ['적절'] + config['label_priority'][1:]]:
        with pytest.raises(ValueError):
            gemma.parse_response(raw, 'classify', {**config, 'label_priority': priority})


def test_explicit_verdict_controls_appropriate_and_requires_error_candidates(tmp_path):
    config = v5_config()
    image = tmp_path / 'input.png'
    image.write_bytes(b'png')
    schema = gemma.request_payload('classify', image, {}, config)['response_format']['json_schema']['schema']
    assert 'verdict' in schema['required']
    assert next(iter(schema['properties'])) == 'verdict'  # llama.cpp emits fields in schema order.
    base = {'reason': '종합 근거', 'image_evidence': '이미지', 'context_evidence': '문맥'}
    def parse(verdict, candidates):
        value = {**base, 'verdict': verdict, 'matched_types': candidates}
        raw = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(value)}}]}
        return gemma.parse_response(raw, 'classify', config)
    good = parse('적절', [])
    assert good['verdict'] == good['label'] == '적절'
    candidates = [{'label': label, 'reason': '유형 근거'} for label in ['불충분형', '중복형']]
    bad = parse('부적절', candidates)
    assert bad['verdict'] == '부적절' and bad['label'] == '중복형'
    assert bad['matched_types'] == candidates
    for verdict, matches in [('부적절', []), ('적절', candidates), ('모름', []), (None, [])]:
        with pytest.raises(ValueError):
            parse(verdict, matches)
    raw = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({**base, 'matched_types': []})}}]}
    with pytest.raises(ValueError):
        gemma.parse_response(raw, 'classify', config)
    legacy = {**config, 'classification_mode': 'multilabel_priority_v1'}
    assert gemma.parse_response(raw, 'classify', legacy)['label'] == '적절'
    assert 'verdict' not in gemma.request_payload('classify', image, {}, legacy)['response_format']['json_schema']['schema']['properties']




def test_owned_server_is_stopped_when_request_fails(tmp_path, monkeypatch):
    class Process:
        dead = False
        def poll(self): return 0 if self.dead else None
        def terminate(self): self.dead = True
        def wait(self, timeout=None): return 0
    process = Process()
    server = gemma.GemmaServer(gemma.default_config(binary_prompt='criteria'), 'gpu:2', tmp_path)
    def start():
        server.process = process
        return server
    monkeypatch.setattr(server, 'start', start)
    with pytest.raises(RuntimeError):
        with server:
            raise RuntimeError('request failed')
    assert process.dead


def test_whitespace_heavy_title_is_normalized_without_altering_alt(tmp_path):
    image = tmp_path / 'input.png'
    image.write_bytes(b'png')
    row = {'page_title': '제목' + '\r\n    ' * 10000 + '본문', 'alt_text': ' 원문\r\n  alt '}
    payload = gemma.request_payload('classify', image, row, gemma.default_config(binary_prompt='criteria'))
    supplied = json.loads(payload['messages'][1]['content'][1]['text'])
    assert supplied['page_title'] == '제목 본문'
    assert supplied['alt_text'] == row['alt_text']


def test_owned_child_exits_even_when_parent_is_killed():
    import os
    import subprocess
    import sys
    import time
    script = '''
import sys, time
from gemma import owned_process
p = owned_process([sys.executable, '-c', 'import time; time.sleep(60)'])
print(p.pid, flush=True)
time.sleep(60)
'''
    parent = subprocess.Popen([sys.executable, '-c', script], cwd=Path(gemma.__file__).parent,
                              stdout=subprocess.PIPE, text=True)
    child_pid = None
    try:
        line = parent.stdout.readline().strip()
        assert line, 'owned_process did not start a child'
        child_pid = int(line)
        parent.kill()
        parent.wait(timeout=5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = Path(f'/proc/{child_pid}/stat')
            if not status.exists() or status.read_text().split()[2] == 'Z':
                break
            time.sleep(0.05)
        else:
            pytest.fail('owned child survived its parent')
    finally:
        if parent.poll() is None: parent.kill()
        parent.wait(timeout=5)
        if child_pid:
            try: os.kill(child_pid, 9)
            except ProcessLookupError: pass
