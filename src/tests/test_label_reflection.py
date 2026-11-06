import json
import threading

import pytest

import benchmark
import common
import gemma
import run_ocr


def _binary(verdict, reason):
    return {
        'verdict': verdict,
        'reason': reason,
        'image_evidence': '이미지 근거',
        'context_evidence': '문맥 근거',
    }


def _visual():
    return {
        'role': '기능 전달', 'reason': '링크 이미지다.',
        'visible_facts': ['링크 목적'], 'context_quotes': ['링크 문구'],
        'required_information': ['링크 목적'],
    }


def _schema_responder(**responses):
    """Return deterministic fake responses even when independent stages overlap."""
    lock = threading.Lock()

    def respond(path, payload, timeout):
        contract = payload['response_format']['json_schema']['name']
        with lock:
            value = responses[contract].pop(0)
        return {'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps(value, ensure_ascii=False)}}]}
    return respond


def test_label_reflection_prompts_use_current_document_and_separate_slots(tmp_path):
    image = tmp_path / 'input.png'
    image.write_bytes(b'png')
    config = gemma.default_config(binary_prompt='guideline_direct', label_reflection=True)
    initial = _binary('적절', '최초 판정')
    feedback = {'feedback': '링크 문구가 기능을 대신 전달하는지 먼저 확인한다.'}
    data = {
        'alt_text': '',
        'in_link': True,
        'link_dest': '정책자료 바로가기',
        '_initial_assessment': initial,
        '_actual_label': '불충분형',
        '_reflection_document': '# 누적 기준\n\n- 기존 기준',
        '_reflection_feedback': feedback,
    }

    generated = gemma.request_payload('reflection_feedback', image, data, config)
    fallback = gemma.request_payload('reflection_fallback', image, data, config)
    updated = gemma.request_payload('reflection_document', image, data, config)
    generated_fields = json.loads(generated['messages'][1]['content'][1]['text'])
    fallback_fields = json.loads(fallback['messages'][1]['content'][1]['text'])
    updated_fields = json.loads(updated['messages'][1]['content'][0]['text'])

    assert config['label_reflection'] is True
    assert config['reflection_gate_policy'] == 'mismatch_reflection_v1'
    assert config['reflection_training_route'] == 'initial_binary_errors_only_v1'
    assert config['error_classification_input'] == 'image_metadata_candidates_v1'
    assert config['session_slots'] == 2
    assert config['execution_mode'] == 'label_reflection_sequential_v1'
    assert set(config['prompts']) == {
        'classify', 'types', 'final', 'reflection_feedback',
        'reflection_fallback', 'reflection_document', 'reflection_document_compact',
        'empty_alt_link_visual_review',
    }
    feedback_prompt = config['prompts']['reflection_feedback']
    assert '맞았다고 전제하거나 틀렸다고 전제하지 말고' in feedback_prompt
    assert '최초 판정이 놓친 비교 조건' in feedback_prompt
    assert '존재하지 않는 결함을 찾거나 반대 결론으로 바꾸도록 유도하지 않는다' in feedback_prompt
    assert '"다시 판단할 때,"로 시작' not in feedback_prompt
    assert '핵심 비교 조건부터 바로' in feedback_prompt
    assert '다시 확인하라/고려하라' in feedback_prompt
    assert '정답 라벨이나 최종 판정 방향을 직접 말하지 않는다' in feedback_prompt
    assert '관련 입력 필드 이름' in feedback_prompt
    fallback_prompt = config['prompts']['reflection_fallback']
    assert 'initial_assessment' in fallback_prompt
    assert '최초 판정의 결론을 그대로 유지하지 않는다' not in fallback_prompt
    assert '피드백 조건이 원본에서 명확히 확인되고 최초 근거를 뒤집을 때만 결론을 변경' in fallback_prompt
    document_prompt = config['prompts']['reflection_document']
    assert '공식 가이드라인 원문이나 라벨 표를 복사하지 않는다' in document_prompt
    assert '짧은 체크리스트' in document_prompt
    assert '기존 항목과 새 피드백이 겹치면 하나로 병합한다' in document_prompt
    assert '전체 3,000자 이내' in document_prompt
    assert '새 피드백과 무관한 기존 항목을 요약하거나 삭제하지 않는다' not in document_prompt
    assert '서로 충돌하는 두 항목을 동시에 남기지 않는다' in document_prompt
    compact_prompt = config['prompts']['reflection_document_compact']
    assert '5,000자를 초과한' in compact_prompt
    assert '동일한 판단 기준의 반복만 제거' in compact_prompt
    assert '상세 조건과 예외 보존을 글자 수보다 우선' in compact_prompt
    assert generated['id_slot'] == 1
    assert fallback['id_slot'] == 0
    assert updated['id_slot'] == 1
    assert generated_fields['current_guidelines'] == data['_reflection_document']
    assert generated_fields['initial_assessment'] == initial
    assert generated_fields['actual_label'] == '불충분형'
    assert fallback_fields['reflection_feedback'] == feedback
    assert fallback_fields['initial_assessment'] == initial
    assert fallback_fields['alt_text'] == ''
    assert fallback_fields['in_link'] is True
    assert 'actual_label' not in fallback_fields
    assert 'current_guidelines' not in fallback_fields
    assert updated_fields == {
        'current_guidelines': data['_reflection_document'],
        'new_feedback': feedback,
    }
    assert updated['messages'][1]['content'][0]['type'] == 'text'
    assert 'image_url' not in json.dumps(updated['messages'][1], ensure_ascii=False)
    assert 'reflection_decision' not in updated_fields
    # llama.cpp rejects large bounded string repetitions while converting JSON Schema to grammar.
    assert 'maxLength' not in gemma.SCHEMAS['reflection_document']['properties']['markdown']

    candidates = {
        'matched_types': [{'label': '불충분형', 'reason': '누락'}],
        'reason': '후보', 'image_evidence': '이미지', 'context_evidence': '문맥',
    }
    types = gemma.request_payload('types_no_ocr', image, {
        **data, '_binary_assessment': initial}, config)
    final = gemma.request_payload('final_no_ocr', image, {
        **data, '_binary_assessment': initial, '_candidate_assessment': candidates}, config)
    type_fields = json.loads(types['messages'][1]['content'][1]['text'])
    final_fields = json.loads(final['messages'][1]['content'][1]['text'])
    assert 'binary_assessment' not in type_fields
    assert 'binary_assessment' not in final_fields
    assert final_fields['candidate_assessment'] == candidates
    assert 'binary_assessment' not in config['prompts']['types']
    assert 'binary_assessment' not in config['prompts']['final']


def test_binary_only_stops_after_binary_without_type_or_final_calls(tmp_path):
    config = gemma.default_config(
        binary_prompt='guideline_direct', label_reflection=True, binary_only=True)
    server = gemma.GemmaServer(config, 'gpu:3', tmp_path)
    initial = _binary('부적절', '핵심 정보가 누락됨')
    first = {'stage': 'classify_no_ocr', 'usage': {}, 'elapsed_seconds': .1,
             'raw_path': 'raw/gemma_no_ocr/a.json'}

    def unexpected_call(*args, **kwargs):
        raise AssertionError('이진분류 전용 실행에서 오류 유형 모델을 호출하면 안 됩니다.')

    server._infer_once = unexpected_call
    value, stats = server.infer_after_binary(
        'classify_no_ocr', tmp_path / 'input.png', {'alt_text': '설명'},
        tmp_path / 'raw/gemma_no_ocr/a.json', initial, first)

    assert config['binary_only'] is True
    assert value['binary_only'] is True
    assert value['verdict'] == '부적절'
    assert value['label'] is None
    assert value['type_classification'] is None
    assert value['final_decision'] is None
    assert [call['stage'] for call in stats['calls']] == ['classify_no_ocr']


def test_reflection_document_prompt_is_selected_by_python_character_count(tmp_path, monkeypatch):
    image = tmp_path / 'input.png'
    image.write_bytes(b'png')
    config = gemma.default_config(binary_prompt='guideline_direct', label_reflection=True)
    server = gemma.GemmaServer(config, 'gpu:1', tmp_path)
    sent = []

    def respond(path, payload, timeout):
        sent.append(payload)
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
            'markdown': '# Reflection 판단 가이드라인\n\n## 누적 재검토 체크리스트\n\n- 기준\n',
        }, ensure_ascii=False)}}]}

    monkeypatch.setattr(server, 'http', respond)
    feedback = {'feedback': '이미지와 문맥을 다시 대조하라.'}
    normal_document = '#' + ('가' * 4999)
    compact_document = '#' + ('가' * 5000)

    _, normal_stats = server.infer_reflection_document(
        image, {'_reflection_document': normal_document, '_reflection_feedback': feedback},
        tmp_path / 'raw/a.reflection_document.json')
    _, compact_stats = server.infer_reflection_document(
        image, {'_reflection_document': compact_document, '_reflection_feedback': feedback},
        tmp_path / 'raw/b.reflection_document.json')

    assert normal_stats['stage'] == 'reflection_document'
    assert compact_stats['stage'] == 'reflection_document_compact'
    assert sent[0]['messages'][0]['content'] == config['prompts']['reflection_document']
    assert sent[1]['messages'][0]['content'] == config['prompts']['reflection_document_compact']
    assert [payload['max_tokens'] for payload in sent] == [6144, 6144]
    assert all(payload['id_slot'] == 1 for payload in sent)
    assert all(len(payload['messages'][1]['content']) == 1 for payload in sent)


def test_reflection_markdown_longer_than_old_limit_is_valid():
    config = gemma.default_config(binary_prompt='guideline_direct', label_reflection=True)
    markdown = '# Reflection 판단 가이드라인\n\n' + ('상세 기준을 보존한다.\n' * 350)
    assert len(markdown) > 3000
    raw = {'choices': [{'finish_reason': 'stop', 'message': {
        'content': json.dumps({'markdown': markdown}, ensure_ascii=False)}}]}

    assert gemma.parse_response(raw, 'reflection_document', config)['markdown'] == markdown


def test_reflection_document_is_replaced_atomically_and_initialized(tmp_path):
    import reflection

    initial = reflection.read_document(tmp_path)
    assert initial.startswith('# Reflection 판단 가이드라인')
    assert (tmp_path / 'reflection_guidelines.md').read_text(encoding='utf-8') == initial

    revised = '# Reflection 판단 가이드라인\n\n## 링크 이미지\n- 링크 문구가 기능을 대신 전달하는지 확인한다.\n'
    reflection.write_document(tmp_path, revised)

    assert reflection.read_document(tmp_path) == revised
    assert not (tmp_path / 'reflection_guidelines.md.tmp').exists()


def test_frozen_document_requires_matching_snapshot(tmp_path):
    import hashlib
    import reflection

    document = '# Reflection 판단 가이드라인\n\n- 고정 기준\n'
    digest = hashlib.sha256(document.encode()).hexdigest()
    reflection.write_document(tmp_path, document)
    assert reflection.read_frozen_document(tmp_path, digest) == document
    with pytest.raises(ValueError, match='SHA-256'):
        reflection.read_frozen_document(tmp_path, '0' * 64)


def test_reflection_inference_uses_feedback_then_fallback_then_document(tmp_path, monkeypatch):
    image = tmp_path / 'input.png'
    image.write_bytes(b'png')
    config = gemma.default_config(binary_prompt='guideline_direct', label_reflection=True)
    initial = _binary('적절', '최초 판정')
    feedback = {'feedback': '이미지의 추가 정보가 링크 문구로 모두 대체되는지 확인한다.'}
    fallback = _binary('부적절', '피드백 재판정')
    document = {'markdown': '# Reflection 판단 가이드라인\n\n- 링크 문구의 정보 대체 범위를 확인한다.\n'}
    candidates = {
        'matched_types': [{'label': '불충분형', 'reason': '추가 정보 누락'}],
        'reason': '후보', 'image_evidence': '이미지 근거', 'context_evidence': '문맥 근거',
    }
    final = {
        'label': '불충분형', 'reason': '최종',
        'image_evidence': '이미지 근거', 'context_evidence': '문맥 근거',
    }
    responses = [feedback, fallback, document, candidates, final]
    sent = []
    server = gemma.GemmaServer(config, 'gpu:1', tmp_path)

    def respond(path, payload, timeout):
        sent.append(payload)
        return {'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps(responses.pop(0), ensure_ascii=False)}}]}

    monkeypatch.setattr(server, 'http', respond)
    data = {'alt_text': '', 'in_link': True, 'link_dest': '정책자료 바로가기'}
    current = '# Reflection 판단 가이드라인\n\n- 기존 기준\n'
    feedback_value, feedback_stats = server.infer_reflection_feedback(
        image, {**data, '_initial_assessment': initial, '_actual_label': '불충분형',
                '_reflection_document': current}, tmp_path / 'raw/a.reflection_feedback.json')
    fallback_value, fallback_stats = server.infer_reflection_fallback(
        image, {**data, '_initial_assessment': initial,
                '_reflection_feedback': feedback_value},
        tmp_path / 'raw/a.reflection_fallback.json')
    document_value, document_stats = server.infer_reflection_document(
        image, {'_reflection_document': current, '_reflection_feedback': feedback_value},
        tmp_path / 'raw/a.reflection_document.json')
    value, stats = server.infer_after_binary(
        'classify_no_ocr', image, data, tmp_path / 'raw/a.json',
        fallback_value, fallback_stats, 'label_reflection_v1', 0)

    assert [payload['id_slot'] for payload in sent] == [1, 0, 1, 0]
    assert feedback_value == feedback
    assert fallback_value == fallback
    assert document_value == document
    assert value['label'] == '불충분형'
    assert stats['calls'][0]['stage'] == 'reflection_fallback'
    assert feedback_stats['stage'] == 'reflection_feedback'
    assert document_stats['stage'] == 'reflection_document'


def test_runner_reflection_chain_updates_document_and_keeps_initial_decision(tmp_path, monkeypatch):
    image = tmp_path / 'input.png'
    image.write_bytes(b'png')
    config = gemma.default_config(binary_prompt='guideline_direct', label_reflection=True)
    initial = _binary('적절', '최초 판정')
    first = {'stage': 'classify_no_ocr', 'usage': {}, 'elapsed_seconds': .1,
             'raw_path': 'raw/gemma_no_ocr/a.json'}
    responses = [
        {'feedback': '핵심 시각 정보가 링크 문구에 없는지 확인한다.'},
        _binary('부적절', '피드백 재판정'),
        {'markdown': '# Reflection 판단 가이드라인\n\n- 핵심 시각 정보가 링크 문구에 없는지 확인한다.\n'},
        {'matched_types': [{'label': '불충분형', 'reason': '핵심 정보 누락'}],
         'reason': '후보', 'image_evidence': '이미지 근거', 'context_evidence': '문맥 근거'},
        {'label': '불충분형', 'reason': '최종',
         'image_evidence': '이미지 근거', 'context_evidence': '문맥 근거'},
    ]
    server = gemma.GemmaServer(config, 'gpu:1', tmp_path)
    monkeypatch.setattr(server, 'http', _schema_responder(
        reflection_feedback=[responses[0]], binary_verdict=[responses[1]],
        reflection_document=[responses[2]], error_types=[responses[3]],
        final_error_type=[responses[4]],
    ))

    value, stats = run_ocr._reflection_chain(
        server, tmp_path, 'a', image,
        {'alt_text': '', 'in_link': True, 'link_dest': '정책자료 바로가기'},
        '불충분형', initial, first, None)

    assert value['binary_decision'] == initial
    assert value['reflection_decision']['verdict'] == '부적절'
    assert value['verdict'] == '부적절'
    assert value['label'] == '불충분형'
    assert value['reflection_feedback']['feedback'].startswith('핵심 시각 정보')
    assert value['reflection_document_sha256']
    assert [call['stage'] for call in stats['calls']] == [
        'classify_no_ocr', 'reflection_feedback', 'reflection_fallback',
        'reflection_document', 'types_no_ocr',
    ]
    assert '핵심 시각 정보' in (tmp_path / 'reflection_guidelines.md').read_text(encoding='utf-8')


def test_runner_overlaps_fallback_and_document_revision_for_initial_error(tmp_path):
    image = tmp_path / 'input.png'
    image.write_bytes(b'png')
    initial = _binary('적절', '최초 판정')
    first = {'stage': 'classify_no_ocr', 'usage': {}, 'elapsed_seconds': .1,
             'raw_path': 'raw/gemma_no_ocr/a.json'}
    barrier = threading.Barrier(2, timeout=1)
    overlapped = []

    class Server:
        def infer_reflection_feedback(self, *args):
            return {'feedback': '다시 확인하라.'}, {
                'stage': 'reflection_feedback', 'usage': {}, 'elapsed_seconds': .1,
                'raw_path': 'raw/gemma_no_ocr/a.reflection_feedback.json'}

        def infer_reflection_fallback(self, *args):
            barrier.wait()
            overlapped.append('fallback')
            return _binary('적절', '재판정'), {
                'stage': 'reflection_fallback', 'usage': {}, 'elapsed_seconds': .1,
                'raw_path': 'raw/gemma_no_ocr/a.reflection_fallback.json'}

        def infer_reflection_document(self, *args):
            barrier.wait()
            overlapped.append('document')
            return {'markdown': '# Reflection 판단 가이드라인\n\n- 다시 확인한다.\n'}, {
                'stage': 'reflection_document', 'usage': {}, 'elapsed_seconds': .1,
                'raw_path': 'raw/gemma_no_ocr/a.reflection_document.json'}

        def infer_after_binary(self, *args):
            return ({'verdict': '적절', 'label': '적절', 'matched_types': []}, {
                'usage': {}, 'elapsed_seconds': .1,
                'calls': [args[5]], 'raw_paths': [args[5]['raw_path']]})

    value, _ = run_ocr._reflection_chain(
        Server(), tmp_path, 'a', image, {'alt_text': '설명'}, '불충분형', initial, first)

    assert set(overlapped) == {'fallback', 'document'}
    assert value['label'] == '적절'


def test_runner_keeps_classification_when_document_update_fails(tmp_path):
    image = tmp_path / 'input.png'
    image.write_bytes(b'png')
    initial = _binary('부적절', '최초 판정')
    first = {'stage': 'classify_no_ocr', 'usage': {}, 'elapsed_seconds': .1,
             'raw_path': 'raw/gemma_no_ocr/a.json'}
    original_document = '# Reflection 판단 가이드라인\n\n- 기존 상세 기준\n'
    (tmp_path / 'reflection_guidelines.md').write_text(original_document, encoding='utf-8')

    class Server:
        config = {'reflection_mode': gemma.REFLECTION_TRAIN}

        def infer_reflection_feedback(self, *args):
            return {'feedback': '핵심 정보를 다시 확인하라.'}, {
                'stage': 'reflection_feedback', 'usage': {}, 'elapsed_seconds': .1,
                'raw_path': 'raw/gemma_no_ocr/a.reflection_feedback.json'}

        def infer_reflection_fallback(self, *args):
            return _binary('적절', '재판정'), {
                'stage': 'reflection_fallback', 'usage': {}, 'elapsed_seconds': .1,
                'raw_path': 'raw/gemma_no_ocr/a.reflection_fallback.json'}

        def infer_reflection_document(self, *args):
            raise ValueError('Reflection Markdown 출력 미완료')

        def infer_after_binary(self, *args):
            return ({'verdict': '적절', 'label': '적절', 'matched_types': []}, {
                'usage': {}, 'elapsed_seconds': .1,
                'calls': [args[5]], 'raw_paths': [args[5]['raw_path']]})

    value, stats = run_ocr._reflection_chain(
        Server(), tmp_path, 'a', image, {'alt_text': '설명'}, '적절', initial, first)

    assert value['label'] == '적절'
    assert value['reflection_document_status'] == 'error'
    assert 'Reflection Markdown 출력 미완료' in value['reflection_document_error']
    assert value['reflection_document_sha256'] == __import__('hashlib').sha256(
        original_document.encode()).hexdigest()
    assert (tmp_path / 'reflection_guidelines.md').read_text(encoding='utf-8') == original_document
    assert [call['stage'] for call in stats['calls']] == [
        'classify_no_ocr', 'reflection_feedback', 'reflection_fallback']


def test_train_correct_initial_binary_skips_feedback_fallback_and_md_but_keeps_subtype(tmp_path):
    image = tmp_path / 'input.png'
    image.write_bytes(b'png')
    initial = _binary('부적절', '최초 판정')
    first = {'stage': 'classify_no_ocr', 'usage': {}, 'elapsed_seconds': .1,
             'raw_path': 'raw/gemma_no_ocr/a.json'}
    original_document = '# Reflection 판단 가이드라인\n\n- 기존 기준\n'
    (tmp_path / 'reflection_guidelines.md').write_text(original_document, encoding='utf-8')

    class Server:
        config = {'reflection_mode': gemma.REFLECTION_TRAIN,
                  'reflection_training_route': 'initial_binary_errors_only_v1'}

        def infer_reflection_feedback(self, *args):
            pytest.fail('이진 방향을 맞힌 train 문제는 피드백을 생성하면 안 됩니다.')

        def infer_reflection_fallback(self, *args):
            pytest.fail('이진 방향을 맞힌 train 문제는 재판정하면 안 됩니다.')

        def infer_reflection_document(self, *args):
            pytest.fail('이진 방향을 맞힌 train 문제는 MD를 갱신하면 안 됩니다.')

        def infer_after_binary(self, *args):
            assert args[4] == initial
            type_call = {'stage': 'types_no_ocr', 'usage': {}, 'elapsed_seconds': .1,
                         'raw_path': 'raw/gemma_no_ocr/a.types.json'}
            final_call = {'stage': 'final_no_ocr', 'usage': {}, 'elapsed_seconds': .1,
                          'raw_path': 'raw/gemma_no_ocr/a.final.json'}
            return ({'verdict': '부적절', 'label': '불충분형',
                     'matched_types': [{'label': '불충분형', 'reason': '핵심 정보 누락'}]}, {
                        'usage': {}, 'elapsed_seconds': .2,
                        'calls': [args[5], type_call, final_call],
                        'raw_paths': [args[5]['raw_path'], type_call['raw_path'], final_call['raw_path']]})

    value, stats = run_ocr._reflection_chain(
        Server(), tmp_path, 'a', image, {'alt_text': '행사'}, '불충분형', initial, first)

    assert value['label'] == '불충분형'
    assert value['reflection_decision'] == initial
    assert value['reflection_feedback'] is None
    assert value['reflection_skipped'] == 'initial_binary_correct'
    assert value['reflection_decision_source'] == 'train_initial_binary_correct_v1'
    assert [call['stage'] for call in stats['calls']] == [
        'classify_no_ocr', 'types_no_ocr', 'final_no_ocr']
    assert (tmp_path / 'reflection_guidelines.md').read_text(encoding='utf-8') == original_document


def test_label_reflection_benchmark_compares_initial_and_feedback_fallback(tmp_path):
    config = gemma.default_config(binary_prompt='guideline_direct', label_reflection=True)
    common.write_json(tmp_path / 'manifest.json', {
        'record_ids': ['a', 'b', 'c'], 'pipeline_mode': 'gemma_only', 'models': [], 'gemma': config})
    common.write_json(tmp_path / 'records.json', [
        {'record_id': 'a', 'label': '적절'},
        {'record_id': 'b', 'label': '불충분형'},
        {'record_id': 'c', 'label': '적절'},
    ])
    cases = {
        'a': ('부적절', '적절', {'feedback': '놓친 기준을 다시 확인하라.'}),
        'b': ('부적절', '부적절', None),
        'c': ('적절', '적절', None),
    }
    for rid, (before, after, feedback) in cases.items():
        common.write_json(common.result_path(tmp_path, 'gemma_no_ocr', rid), {
            'status': 'ok', 'record_id': rid,
            'label': '적절' if after == '적절' else '불충분형',
            'verdict': after,
            'binary_decision': _binary(before, '최초'),
            'reflection_decision': _binary(after, '피드백 재판정'),
            'reflection_feedback': feedback,
            **({'reflection_skipped': 'initial_binary_correct'} if feedback is None else {}),
            'matched_types': [] if after == '적절' else [
                {'label': '불충분형', 'reason': '핵심 정보 누락'}],
        })

    report = benchmark.build_benchmark(tmp_path)
    comparison = report['label_reflection']
    metrics = comparison['available']['without_ocr']

    assert metrics['count'] == 3
    assert metrics['feedback_applied'] == 1
    assert metrics['initial_correct_skipped'] == 2
    assert metrics['changed'] == 1
    assert metrics['corrected'] == 1
    assert metrics['harmed'] == 0
    assert metrics['before']['accuracy'] == pytest.approx(2 / 3)
    assert metrics['after']['accuracy'] == 1
    assert comparison['teacher_assisted_train_only'] is True


def test_binary_only_benchmark_scores_initial_and_reflected_verdicts_without_seven_labels(tmp_path):
    config = gemma.default_config(
        binary_prompt='guideline_direct', label_reflection=True, binary_only=True)
    common.write_json(tmp_path / 'manifest.json', {
        'record_ids': ['a', 'b'], 'pipeline_mode': 'gemma_only', 'models': [], 'gemma': config})
    common.write_json(tmp_path / 'records.json', [
        {'record_id': 'a', 'label': '적절'},
        {'record_id': 'b', 'label': '불충분형'},
    ])
    for rid, truth, before, after in [
            ('a', '적절', '적절', '적절'),
            ('b', '불충분형', '적절', '부적절')]:
        common.write_json(common.result_path(tmp_path, 'gemma_no_ocr', rid), {
            'status': 'ok', 'record_id': rid, 'binary_only': True,
            'label': None, 'verdict': after, 'matched_types': [],
            'binary_decision': _binary(before, '최초'),
            'reflection_decision': _binary(after, '최종'),
            'reflection_feedback': None,
        })

    report = benchmark.build_benchmark(tmp_path, audit_requests=False)

    assert report['binary_only'] is True
    assert report['coverage']['without_ocr'] == {
        'ok': 2, 'error': 0, 'pending': 0, 'labeled_ok': 2}
    assert report['available']['without_ocr'] is None
    assert report['candidate_sets'] is None
    assert report['first_stage']['available']['without_ocr']['accuracy'] == .5
    reflected = report['label_reflection']['available']['without_ocr']
    assert reflected['after']['accuracy'] == 1
    assert reflected['after']['binary_f1'] == 1


def test_candidate_only_benchmark_scores_binary_and_candidate_recall_without_final_labels(tmp_path):
    sha = '0' * 64
    config = gemma.default_config(
        binary_prompt='guideline_direct', label_reflection=True,
        reflection_mode=gemma.REFLECTION_FROZEN,
        reflection_guideline_source_run='train-run',
        reflection_guideline_sha256=sha,
        candidate_only=True, type_reflection=True,
        type_reflection_guideline_sha256=sha,
    )
    common.write_json(tmp_path / 'manifest.json', {
        'record_ids': ['a', 'b'], 'pipeline_mode': 'gemma_only', 'models': [], 'gemma': config})
    common.write_json(tmp_path / 'records.json', [
        {'record_id': 'a', 'label': '적절'},
        {'record_id': 'b', 'label': '불충분형'},
    ])
    common.write_json(common.result_path(tmp_path, 'gemma_no_ocr', 'a'), {
        'status': 'ok', 'record_id': 'a', 'candidate_only': True,
        'label': None, 'verdict': '적절', 'matched_types': [],
        'binary_decision': _binary('적절', '최초'),
        'reflection_decision': _binary('적절', '최종'),
        'type_classification': None, 'final_decision': None,
    })
    before = {'matched_types': [{'label': '중복형', 'reason': '중복 근거'}],
              'reason': '최초 후보', 'image_evidence': '이미지', 'context_evidence': '문맥'}
    after = {'matched_types': [
                {'label': '중복형', 'reason': '중복 근거'},
                {'label': '불충분형', 'reason': '누락 근거'}],
             'reason': '확장 후보', 'image_evidence': '이미지', 'context_evidence': '문맥'}
    common.write_json(common.result_path(tmp_path, 'gemma_no_ocr', 'b'), {
        'status': 'ok', 'record_id': 'b', 'candidate_only': True,
        'label': None, 'verdict': '부적절', 'matched_types': after['matched_types'],
        'binary_decision': _binary('부적절', '최초'),
        'reflection_decision': _binary('부적절', '최종'),
        'initial_type_classification': before,
        'type_classification': after, 'final_decision': None,
    })

    report = benchmark.build_benchmark(tmp_path, audit_requests=False)

    assert report['candidate_only'] is True
    assert report['available']['without_ocr'] is None
    assert report['paired']['without_ocr'] is None
    assert report['candidate_sets'] is None
    metrics = report['type_reflection']['available']['without_ocr']
    assert metrics['count'] == 1
    assert metrics['before_hit_rate'] == 0
    assert metrics['after_hit_rate'] == 1
    assert metrics['after_miss_count'] == 0
    assert metrics['before_mean_size'] == 1
    assert metrics['after_mean_size'] == 2


def test_completed_reflection_chain_is_audited_for_final_scores(tmp_path, monkeypatch):
    image = tmp_path / 'inputs/a.png'
    image.parent.mkdir()
    image.write_bytes(b'png')
    config = gemma.default_config(binary_prompt='guideline_direct', label_reflection=True)
    row = {'record_id': 'a', 'label': '불충분형', 'alt_text': '행사', 'in_link': False}
    common.write_json(tmp_path / 'manifest.json', {
        'record_ids': ['a'], 'pipeline_mode': 'gemma_only', 'models': [], 'gemma': config})
    common.write_json(tmp_path / 'records.json', [row])
    decisions = [
        _binary('적절', '최초 판정'),
        {'feedback': '정보성 이미지의 핵심 정보를 alt가 전달하는지 확인한다.'},
        _binary('부적절', '피드백 재판정'),
        {'markdown': '# Reflection 판단 가이드라인\n\n- 정보성 이미지의 핵심 정보를 확인한다.\n'},
        {'matched_types': [{'label': '불충분형', 'reason': '핵심 정보 누락'}],
         'reason': '후보', 'image_evidence': '이미지 근거', 'context_evidence': '문맥 근거'},
        {'label': '불충분형', 'reason': '최종',
         'image_evidence': '이미지 근거', 'context_evidence': '문맥 근거'},
    ]
    server = gemma.GemmaServer(config, 'gpu:1', tmp_path)
    monkeypatch.setattr(server, 'http', _schema_responder(
        binary_verdict=[decisions[0], decisions[2]],
        reflection_feedback=[decisions[1]], reflection_document=[decisions[3]],
        error_types=[decisions[4]], final_error_type=[decisions[5]],
    ))
    metadata = {key: row.get(key) for key in gemma.METADATA_FIELDS}
    initial, first = server.infer_binary(
        'classify_no_ocr', image, metadata, tmp_path / 'raw/gemma_no_ocr/a.json', 0)
    value, stats = run_ocr._reflection_chain(
        server, tmp_path, 'a', image, metadata, row['label'], initial, first)
    value.update(status='ok', record_id='a', input_path='inputs/a.png', fingerprint='key',
                 calls=stats['calls'], raw_paths=stats['raw_paths'])
    common.write_json(common.result_path(tmp_path, 'gemma_no_ocr', 'a'), value)

    report = benchmark.build_benchmark(tmp_path)

    assert report['paired_ids'] == ['a']
    assert report['label_reflection']['available']['without_ocr']['after']['accuracy'] == 1
    assert run_ocr._cached_stage(tmp_path, 'gemma_no_ocr', 'a', 'key') == value


def test_correct_deterministic_gate_skips_train_reflection_or_document_update(tmp_path, monkeypatch):
    image = tmp_path / 'inputs/a.png'
    image.parent.mkdir()
    image.write_bytes(b'png')
    config = gemma.default_config(
        binary_prompt='guideline_direct', label_reflection=True, binary_only=True)
    row = {'record_id': 'a', 'label': '불충분형', 'alt_text': '', 'in_link': True,
           'link_dest': 'https://example.test/post'}
    common.write_json(tmp_path / 'manifest.json', {
        'record_ids': ['a'], 'pipeline_mode': 'gemma_only', 'models': [], 'gemma': config})
    common.write_json(tmp_path / 'records.json', [row])
    metadata = {key: row.get(key) for key in gemma.METADATA_FIELDS}
    initial = gemma.empty_alt_link_gate(metadata)
    first = run_ocr._gate_audit(tmp_path, 'a', metadata, initial)
    original_document = '# Reflection 판단 가이드라인\n\n- 기존 기준\n'
    (tmp_path / 'reflection_guidelines.md').write_text(original_document, encoding='utf-8')
    server = gemma.GemmaServer(config, 'gpu:1', tmp_path)
    monkeypatch.setattr(server, 'http', lambda *args: pytest.fail('결정적 게이트는 Reflection을 호출하면 안 됩니다.'))

    value, stats = run_ocr._reflection_chain(
        server, tmp_path, 'a', image, metadata, row['label'], initial, first,
        gemma.EMPTY_ALT_LINK_GATE)
    value.update(status='ok', record_id='a', input_path='inputs/a.png',
                 calls=stats['calls'], raw_paths=stats['raw_paths'])
    common.write_json(common.result_path(tmp_path, 'gemma_no_ocr', 'a'), value)
    report = benchmark.build_benchmark(tmp_path)

    assert report['paired_ids'] == ['a']
    assert value['reflection_decision'] == initial
    assert value['reflection_feedback'] is None
    assert value['reflection_skipped'] == 'initial_binary_correct'
    assert value['reflection_decision_source'] == 'train_initial_binary_correct_v1'
    assert [call['stage'] for call in stats['calls']] == ['binary_gate']
    assert (tmp_path / 'reflection_guidelines.md').read_text(encoding='utf-8') == original_document
    assert not list((tmp_path / 'raw/gemma_no_ocr').glob('a.reflection_*.json'))


def test_correct_improper_gate_skips_train_reflection_and_classifies_subtype(tmp_path, monkeypatch):
    image = tmp_path / 'input.png'
    image.write_bytes(b'png')
    config = gemma.default_config(binary_prompt='guideline_direct', label_reflection=True)
    metadata = {'alt_text': '', 'in_link': True, 'link_dest': 'https://example.test/post'}
    initial = gemma.empty_alt_link_gate(metadata)
    first = {'stage': 'binary_gate', 'usage': {}, 'elapsed_seconds': 0,
             'raw_path': 'raw/gemma_no_ocr/a.gate.json'}
    server = gemma.GemmaServer(config, 'gpu:1', tmp_path)
    monkeypatch.setattr(server, 'http', _schema_responder(
        error_types=[{
            'matched_types': [{'label': '불충분형', 'reason': '목적지 이름 없음'}],
            'reason': '후보', 'image_evidence': '이미지', 'context_evidence': 'URL만 있음',
        }],
        final_error_type=[{
            'label': '불충분형', 'reason': '최종',
            'image_evidence': '이미지', 'context_evidence': 'URL만 있음',
        }],
    ))

    value, stats = run_ocr._reflection_chain(
        server, tmp_path, 'a', image, metadata, '불충분형', initial, first,
        gemma.EMPTY_ALT_LINK_GATE)

    assert initial['verdict'] == '부적절'
    assert value['label'] == '불충분형'
    assert value['reflection_decision'] == initial
    assert value['reflection_feedback'] is None
    assert value['reflection_skipped'] == 'initial_binary_correct'
    assert [call['stage'] for call in stats['calls']] == [
        'binary_gate', 'types_no_ocr']


def test_same_link_redundancy_gate_skips_train_reflection_even_if_label_disagrees(
        tmp_path, monkeypatch):
    image = tmp_path / 'input.png'
    image.write_bytes(b'png')
    config = gemma.default_config(
        binary_prompt='guideline_direct', label_reflection=True, binary_only=True)
    metadata = {'alt_text': '서구의회 이미지', 'in_link': True,
                'link_dest': '서구의회 서구의회 홈페이지'}
    initial = gemma.same_link_alt_redundancy_gate(metadata)
    first = {'stage': 'binary_gate', 'usage': {}, 'elapsed_seconds': 0,
             'raw_path': 'raw/gemma_no_ocr/a.gate.json'}
    original_document = '# Reflection 판단 가이드라인\n\n- 기존 기준\n'
    (tmp_path / 'reflection_guidelines.md').write_text(original_document, encoding='utf-8')
    server = gemma.GemmaServer(config, 'gpu:1', tmp_path)
    monkeypatch.setattr(
        server, 'http', lambda *args: pytest.fail('같은 링크 중복 게이트는 LLM을 호출하면 안 됩니다.'))

    value, stats = run_ocr._reflection_chain(
        server, tmp_path, 'a', image, metadata, '적절', initial, first,
        gemma.SAME_LINK_ALT_REDUNDANCY_GATE)

    assert value['verdict'] == '부적절'
    assert value['reflection_decision'] == initial
    assert value['reflection_feedback'] is None
    assert value['reflection_skipped'] == 'deterministic_binary_gate'
    assert value['reflection_decision_source'] == 'deterministic_gate_v1'
    assert value['binary_decision_source'] == 'same_link_alt_redundancy_v1'
    assert [call['stage'] for call in stats['calls']] == ['binary_gate']
    assert (tmp_path / 'reflection_guidelines.md').read_text(encoding='utf-8') == original_document


def test_wrong_train_visual_review_receives_feedback_and_updates_md(tmp_path, monkeypatch):
    image = tmp_path / 'input.png'
    image.write_bytes(b'png')
    config = gemma.default_config(binary_prompt='guideline_direct', label_reflection=True)
    metadata = {'alt_text': '', 'in_link': True, 'link_dest': '정책자료 바로가기'}
    assert gemma.empty_alt_link_gate(metadata) is None
    assert gemma.uses_empty_alt_link_visual_review(metadata)
    initial = _binary('적절', '시각 예외 검사 최초 판정')
    first = {'stage': 'empty_alt_link_visual_review', 'usage': {}, 'elapsed_seconds': 0,
             'raw_path': 'raw/gemma_no_ocr/a.json'}
    server = gemma.GemmaServer(config, 'gpu:1', tmp_path)
    monkeypatch.setattr(server, 'http', _schema_responder(
        reflection_feedback=[{'feedback': '링크 문구가 이미지의 고유 정보를 모두 대신하는지 다시 확인하라.'}],
        binary_verdict=[_binary('부적절', '재판정')],
        reflection_document=[{'markdown': '# Reflection 판단 가이드라인\n\n- 고유 정보 대체 범위를 확인한다.\n'}],
        error_types=[{
            'matched_types': [{'label': '불충분형', 'reason': '고유 정보 누락'}],
            'reason': '후보', 'image_evidence': '이미지', 'context_evidence': '링크 문구',
        }],
        final_error_type=[{
            'label': '불충분형', 'reason': '최종',
            'image_evidence': '이미지', 'context_evidence': '링크 문구',
        }],
    ))

    value, stats = run_ocr._reflection_chain(
        server, tmp_path, 'a', image, metadata, '불충분형', initial, first,
        gemma.EMPTY_ALT_LINK_VISUAL_REVIEW)

    assert initial['verdict'] == '적절'
    assert value['reflection_decision']['verdict'] == '부적절'
    assert value.get('reflection_skipped') is None
    assert value['label'] == '불충분형'
    assert [call['stage'] for call in stats['calls']] == [
        'empty_alt_link_visual_review', 'reflection_feedback', 'reflection_fallback',
        'reflection_document', 'types_no_ocr']


def test_frozen_reflection_replaces_label_with_trained_guideline(tmp_path):
    image = tmp_path / 'input.png'
    image.write_bytes(b'png')
    digest = 'a' * 64
    config = gemma.default_config(
        binary_prompt='guideline_direct', label_reflection=True,
        reflection_mode='frozen_guideline_v1',
        reflection_guideline_source_run='train_run',
        reflection_guideline_sha256=digest,
        binary_only=True,
    )
    initial = _binary('적절', '최초 판정')
    document = '# Reflection 판단 가이드라인\n\n- 링크 목적을 다시 확인한다.\n'
    payload = gemma.request_payload('reflection_feedback', image, {
        'alt_text': '설명', '_initial_assessment': initial,
        '_reflection_document': document,
    }, config)
    fields = json.loads(payload['messages'][1]['content'][1]['text'])
    fallback_payload = gemma.request_payload('reflection_fallback', image, {
        'alt_text': '설명', '_initial_assessment': initial,
        '_reflection_feedback': {'feedback': '링크 목적을 다시 확인하라.'},
        '_reflection_document': document,
    }, config)
    fallback_fields = json.loads(fallback_payload['messages'][1]['content'][1]['text'])

    assert config['reflection_mode'] == 'frozen_guideline_v1'
    assert '+frozen-reflection-v3-feedback-gate-md-fallback' in config['prompt_version']
    assert config['frozen_feedback_policy'] == 'applicability_gate_v1'
    assert config['reflection_guideline_source_run'] == 'train_run'
    assert config['reflection_guideline_sha256'] == digest
    assert 'reflection_document' not in config['prompts']
    assert 'actual_label' not in fields
    assert fields['current_guidelines'] == document
    assert fields['initial_assessment'] == initial
    assert fallback_fields['current_guidelines'] == document
    assert 'current_guidelines' in config['prompts']['reflection_fallback']
    assert '실제 라벨을 입력으로 사용하지 않는다' not in config['prompts']['reflection_feedback']
    assert '"action":"피드백 제공"' in config['prompts']['reflection_feedback']
    assert '"action":"통과"' in config['prompts']['reflection_feedback']
    assert '구체적인 대상명' in config['prompts']['reflection_feedback']
    assert '`N페이지_`' in config['prompts']['reflection_feedback']
    assert '`첨부파일 아이콘`' in config['prompts']['reflection_feedback']
    assert payload['response_format']['json_schema']['name'] == 'reflection_feedback_gate'


def test_frozen_v9_evidence_contract_hides_alt_before_coverage_audit(tmp_path):
    image = tmp_path / 'input.png'
    image.write_bytes(b'png')
    document = '# Reflection 판단 가이드라인\n\n- 고정 기준\n'
    import hashlib
    config = gemma.default_config(
        binary_prompt='guideline_direct', label_reflection=True,
        reflection_mode='frozen_guideline_v1',
        reflection_guideline_source_run='train_run',
        reflection_guideline_sha256=hashlib.sha256(document.encode()).hexdigest())
    config['frozen_evidence_pipeline'] = 'visual_coverage_v9'
    config['prompt_version'] += '+frozen-evidence-v9'
    visual = {
        'role': '정보 전달', 'reason': '행사 포스터다.',
        'visible_facts': ['행사 기간'], 'context_quotes': ['행사 안내'],
        'required_information': ['행사 기간'],
    }
    data = {
        'alt_text': '행사 안내', 'page_title': '행사', 'context_text': '행사 안내',
        'in_link': False, 'link_dest': '', 'heading_path': '행사', 'kogl_marker': False,
    }

    evidence = gemma.request_payload('reflection_visual_evidence', image, data, config)
    coverage = gemma.request_payload('reflection_coverage_audit', image, {
        **data, '_initial_assessment': visual}, config)
    evidence_fields = json.loads(evidence['messages'][1]['content'][1]['text'])
    coverage_fields = json.loads(coverage['messages'][1]['content'][1]['text'])

    assert config['frozen_evidence_pipeline'] == 'visual_coverage_v9'
    assert '+frozen-evidence-v9' in config['prompt_version']
    assert '[재검토 위험 신호]' in config['prompts']['reflection_feedback']
    assert '재검토 완료로 보지 않는다' in config['prompts']['reflection_feedback']
    assert 'alt_text' not in evidence_fields
    assert 'initial_assessment' not in evidence_fields
    assert coverage_fields['alt_text'] == '행사 안내'
    assert coverage_fields['initial_assessment'] == visual
    assert evidence['response_format']['json_schema']['name'] == 'visual_evidence'
    assert coverage['response_format']['json_schema']['name'] == 'binary_verdict'
    assert evidence['id_slot'] == coverage['id_slot'] == 1


def test_frozen_v9_coverage_failure_drives_fallback_without_feedback_call(tmp_path, monkeypatch):
    image = tmp_path / 'inputs/a.png'
    image.parent.mkdir()
    image.write_bytes(b'png')
    document = '# Reflection 판단 가이드라인\n\n- 행사 핵심 정보를 확인한다.\n'
    import hashlib
    digest = hashlib.sha256(document.encode()).hexdigest()
    config = gemma.default_config(
        binary_prompt='guideline_direct', label_reflection=True,
        reflection_mode='frozen_guideline_v1',
        reflection_guideline_source_run='train_run',
        reflection_guideline_sha256=digest, binary_only=True)
    config['frozen_evidence_pipeline'] = 'visual_coverage_v9'
    row = {'record_id': 'a', 'label': '불충분형', 'alt_text': '행사 안내',
           'context_text': '행사 안내', 'in_link': False}
    common.write_json(tmp_path / 'manifest.json', {
        'record_ids': ['a'], 'pipeline_mode': 'gemma_only', 'models': [], 'gemma': config})
    common.write_json(tmp_path / 'records.json', [row])
    (tmp_path / 'reflection_guidelines.md').write_text(document, encoding='utf-8')
    initial = _binary('적절', '행사 안내 문구가 있다.')
    visual = {
        'role': '정보 전달', 'reason': '기간이 적힌 행사 포스터다.',
        'visible_facts': ['10월 3일'], 'context_quotes': ['행사 안내'],
        'required_information': ['행사 기간'],
    }
    coverage = _binary('부적절', '행사 기간이 alt와 인접 문구에 없다.')
    fallback = _binary('부적절', '행사 기간 누락을 확인했다.')
    server = gemma.GemmaServer(config, 'gpu:1', tmp_path)
    monkeypatch.setattr(server, 'http', _schema_responder(
        visual_evidence=[visual], binary_verdict=[initial, coverage, fallback]))
    metadata = {key: row.get(key) for key in gemma.METADATA_FIELDS}
    first_value, first = server.infer_binary(
        'classify_no_ocr', image, metadata, tmp_path / 'raw/gemma_no_ocr/a.json', 0)

    value, stats = run_ocr._reflection_chain(
        server, tmp_path, 'a', image, metadata, row['label'], first_value, first)
    value.update(status='ok', record_id='a', input_path='inputs/a.png',
                 calls=stats['calls'], raw_paths=stats['raw_paths'])
    common.write_json(common.result_path(tmp_path, 'gemma_no_ocr', 'a'), value)
    report = benchmark.build_benchmark(tmp_path)

    assert value['reflection_visual_evidence'] == visual
    assert value['reflection_coverage_audit'] == coverage
    assert value['reflection_feedback_source'] == 'visual_coverage_v9'
    assert value['reflection_feedback']['action'] == '피드백 제공'
    assert value['reflection_decision'] == fallback
    assert [call['stage'] for call in stats['calls']] == [
        'classify_no_ocr', 'reflection_visual_evidence',
        'reflection_coverage_audit', 'reflection_fallback']
    assert not (tmp_path / 'raw/gemma_no_ocr/a.reflection_feedback.json').exists()
    assert report['paired_ids'] == ['a']
    assert report['label_reflection']['available']['without_ocr']['after']['accuracy'] == 1


def test_frozen_reflection_uses_md_without_updating_it(tmp_path, monkeypatch):
    image = tmp_path / 'input.png'
    image.write_bytes(b'png')
    document = '# Reflection 판단 가이드라인\n\n- 링크 목적을 다시 확인한다.\n'
    import hashlib
    digest = hashlib.sha256(document.encode()).hexdigest()
    config = gemma.default_config(
        binary_prompt='guideline_direct', label_reflection=True,
        reflection_mode='frozen_guideline_v1',
        reflection_guideline_source_run='train_run',
        reflection_guideline_sha256=digest,
    )
    (tmp_path / 'reflection_guidelines.md').write_text(document, encoding='utf-8')
    initial = _binary('적절', '최초 판정')
    first = {'stage': 'classify_no_ocr', 'usage': {}, 'elapsed_seconds': .1,
             'raw_path': 'raw/gemma_no_ocr/a.json'}
    feedback = {'action': '피드백 제공',
                'feedback': '누적 기준에 따라 링크 목적 전달 여부를 다시 확인하라.',
                'reason': '현재 사례에 링크 목적 기준이 구체적으로 적용된다.'}
    fallback = _binary('적절', '고정 MD 재판정')
    server = gemma.GemmaServer(config, 'gpu:1', tmp_path)
    monkeypatch.setattr(server, 'http', _schema_responder(
        reflection_feedback_gate=[feedback], binary_verdict=[fallback]))

    value, stats = run_ocr._reflection_chain(
        server, tmp_path, 'a', image, {'alt_text': '설명'}, None, initial, first)

    assert value['label'] == '적절'
    assert value['reflection_decision'] == fallback
    assert value['reflection_decision_source'] == 'frozen_guideline_v1'
    assert value['reflection_document_sha256'] == digest
    assert [call['stage'] for call in stats['calls']] == [
        'classify_no_ocr', 'reflection_feedback', 'reflection_fallback']
    fallback_request = common.read_json(
        tmp_path / 'raw/gemma_no_ocr/a.reflection_fallback.request.json')
    fallback_fields = json.loads(fallback_request['messages'][1]['content'][1]['text'])
    assert fallback_fields['reflection_feedback'] == {'feedback': feedback['feedback']}
    assert fallback_fields['current_guidelines'] == document
    assert (tmp_path / 'reflection_guidelines.md').read_text(encoding='utf-8') == document
    assert not (tmp_path / 'raw/gemma_no_ocr/a.reflection_document.json').exists()


def test_frozen_reflection_can_pass_without_fallback(tmp_path, monkeypatch):
    image = tmp_path / 'inputs/a.png'
    image.parent.mkdir()
    image.write_bytes(b'png')
    document = '# Reflection 판단 가이드라인\n\n- 링크 목적을 다시 확인한다.\n'
    import hashlib
    digest = hashlib.sha256(document.encode()).hexdigest()
    config = gemma.default_config(
        binary_prompt='guideline_direct', label_reflection=True,
        reflection_mode='frozen_guideline_v1',
        reflection_guideline_source_run='train_run',
        reflection_guideline_sha256=digest,
        binary_only=True,
    )
    row = {'record_id': 'a', 'label': '적절', 'alt_text': '이미지 설명',
           'in_link': False}
    common.write_json(tmp_path / 'manifest.json', {
        'record_ids': ['a'], 'pipeline_mode': 'gemma_only', 'models': [], 'gemma': config})
    common.write_json(tmp_path / 'records.json', [row])
    (tmp_path / 'reflection_guidelines.md').write_text(document, encoding='utf-8')
    initial = _binary('적절', '이미지와 alt가 일치한다.')
    gate = {'action': '통과', 'feedback': '',
            'reason': '누적 MD에서 최초 판정을 다시 검토할 구체적인 기준이 없다.'}
    server = gemma.GemmaServer(config, 'gpu:1', tmp_path)
    monkeypatch.setattr(server, 'http', _schema_responder(
        binary_verdict=[initial], reflection_feedback_gate=[gate]))
    metadata = {key: row.get(key) for key in gemma.METADATA_FIELDS}
    first_value, first = server.infer_binary(
        'classify_no_ocr', image, metadata, tmp_path / 'raw/gemma_no_ocr/a.json', 0)

    value, stats = run_ocr._reflection_chain(
        server, tmp_path, 'a', image, metadata, None, first_value, first)
    value.update(status='ok', record_id='a', input_path='inputs/a.png',
                 calls=stats['calls'], raw_paths=stats['raw_paths'])
    common.write_json(common.result_path(tmp_path, 'gemma_no_ocr', 'a'), value)
    report = benchmark.build_benchmark(tmp_path)

    assert value['label'] is None
    assert value['verdict'] == '적절'
    assert value['reflection_decision'] == initial
    assert value['reflection_feedback'] == gate
    assert value['reflection_skipped'] == 'frozen_feedback_not_needed'
    assert [call['stage'] for call in stats['calls']] == [
        'classify_no_ocr', 'reflection_feedback']
    assert report['paired_ids'] == ['a']
    assert report['label_reflection']['available']['without_ocr']['frozen_feedback_passed'] == 1
    assert not (tmp_path / 'raw/gemma_no_ocr/a.reflection_fallback.json').exists()


def test_frozen_reflection_benchmark_audits_without_teacher_label(tmp_path, monkeypatch):
    import hashlib

    image = tmp_path / 'inputs/a.png'
    image.parent.mkdir()
    image.write_bytes(b'png')
    document = '# Reflection 판단 가이드라인\n\n- 링크 목적을 다시 확인한다.\n'
    digest = hashlib.sha256(document.encode()).hexdigest()
    config = gemma.default_config(
        binary_prompt='guideline_direct', label_reflection=True,
        reflection_mode='frozen_guideline_v1',
        reflection_guideline_source_run='train_run',
        reflection_guideline_sha256=digest,
    )
    row = {'record_id': 'a', 'label': '적절', 'alt_text': '기관 홈페이지',
           'in_link': True, 'link_dest': '기관 홈페이지'}
    common.write_json(tmp_path / 'manifest.json', {
        'record_ids': ['a'], 'pipeline_mode': 'gemma_only', 'models': [], 'gemma': config})
    common.write_json(tmp_path / 'records.json', [row])
    (tmp_path / 'reflection_guidelines.md').write_text(document, encoding='utf-8')
    initial = _binary('적절', '링크 목적을 전달함')
    feedback = {'action': '피드백 제공',
                'feedback': 'alt_text가 link_dest의 목적을 전달하는지 다시 확인하라.',
                'reason': '고정 MD의 링크 목적 기준이 적용된다.'}
    fallback = _binary('적절', '고정 기준으로 다시 확인해도 목적을 전달함')
    server = gemma.GemmaServer(config, 'gpu:1', tmp_path)
    monkeypatch.setattr(server, 'http', _schema_responder(
        visual_evidence=[_visual()],
        binary_verdict=[initial, _binary('적절', '링크 목적이 전달된다.'), fallback],
        reflection_feedback_gate=[feedback]))
    metadata = {key: row.get(key) for key in gemma.METADATA_FIELDS}
    first_value, first_stats = server.infer_binary(
        'classify_no_ocr', image, metadata, tmp_path / 'raw/gemma_no_ocr/a.json', 0)
    value, stats = run_ocr._reflection_chain(
        server, tmp_path, 'a', image, metadata, None, first_value, first_stats)
    value.update(status='ok', record_id='a', input_path='inputs/a.png',
                 calls=stats['calls'], raw_paths=stats['raw_paths'])
    common.write_json(common.result_path(tmp_path, 'gemma_no_ocr', 'a'), value)

    report = benchmark.build_benchmark(tmp_path)
    request = common.read_json(tmp_path / 'raw/gemma_no_ocr/a.reflection_feedback.request.json')
    fields = json.loads(request['messages'][1]['content'][1]['text'])

    assert report['paired_ids'] == ['a']
    assert report['stage_layout'] == 'frozen_reflection_v1'
    assert report['label_reflection']['teacher_assisted_train_only'] is False
    assert report['label_reflection']['reflection_mode'] == 'frozen_guideline_v1'
    assert report['label_reflection']['available']['without_ocr']['after']['accuracy'] == 1
    assert fields['current_guidelines'] == document
    assert 'actual_label' not in fields



