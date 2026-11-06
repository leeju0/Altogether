import hashlib
import json

import gemma
import benchmark
import common
import reflection
import run_ocr


def _types(*labels):
    return {
        'matched_types': [{'label': label, 'reason': f'{label} 근거'} for label in labels],
        'reason': '후보 근거', 'image_evidence': '이미지 근거', 'context_evidence': '문맥 근거',
    }


def _final(label='중복형'):
    return {'label': label, 'reason': '최종 근거',
            'image_evidence': '이미지 근거', 'context_evidence': '문맥 근거'}


def _metadata():
    data = {key: None for key in gemma.METADATA_FIELDS}
    data.update(alt_text='행사 안내', in_link=False)
    return data


def _stats(stage):
    return {'stage': stage, 'usage': {}, 'elapsed_seconds': .1,
            'raw_path': f'raw/gemma_no_ocr/a.{stage}.json'}


def _user_json(payload):
    return json.loads(payload['messages'][1]['content'][-1]['text'])


def test_final_reflection_document_is_separate_atomic_and_hash_checked(tmp_path):
    binary = reflection.read_document(tmp_path)
    candidate = reflection.read_type_document(tmp_path)
    final = reflection.read_final_document(tmp_path)

    assert final.startswith('# 최종 오류 유형 선택 Reflection 가이드라인')
    assert len({reflection.FILENAME, reflection.TYPE_FILENAME, reflection.FINAL_FILENAME}) == 3

    revised = '# 최종 오류 유형 선택 Reflection 가이드라인\n\n- 경계 기준\n'
    reflection.write_final_document(tmp_path, revised)
    assert reflection.read_final_document(tmp_path) == revised
    assert reflection.read_document(tmp_path) == binary
    assert reflection.read_type_document(tmp_path) == candidate
    sha = hashlib.sha256(revised.encode()).hexdigest()
    assert reflection.read_frozen_final_document(tmp_path, sha) == revised
    try:
        reflection.read_frozen_final_document(tmp_path, '0' * 64)
    except ValueError as exc:
        assert 'SHA-256' in str(exc)
    else:
        raise AssertionError('Frozen 최종 Reflection MD 해시 불일치를 거부해야 합니다.')


def test_final_reflection_config_contains_v11_prompts_and_mechanical_document_branch():
    config = gemma.default_config(binary_prompt='guideline_direct', label_reflection=True,
                                  type_reflection=True, final_reflection=True)

    for stage in ('final_reflection_feedback', 'final_reflection_fallback',
                  'final_reflection_document', 'final_reflection_document_compact'):
        prompt = config['prompts'][stage]
        assert '## 1.' in prompt and '## 3.' in prompt and '## 5.' in prompt
        assert '## 4.' not in prompt
    assert gemma.final_reflection_document_stage(config, 'x' * 5000) == 'final_reflection_document'
    assert gemma.final_reflection_document_stage(config, 'x' * 5001) == 'final_reflection_document_compact'
    # 경로 유효성 검사는 파일 존재만 요구하며 이 테스트에서 GPU를 로드하지 않는다.
    import sys
    for key in ('model_path', 'mmproj_path', 'server_path'):
        config[key] = sys.executable
    gemma.validate_config(config)


def test_final_reselection_uses_feedback_and_md_without_first_choice():
    config = gemma.default_config(binary_prompt='guideline_direct', label_reflection=True,
                                  type_reflection=True, final_reflection=True)
    prompt = config['prompts']['final_reflection_fallback']

    assert 'Reflection 피드백과 누적 최종 선택 가이드라인' in prompt
    assert 'initial_final_assessment' not in prompt
    assert '같은 대상·사건·기관을 설명하면서' in prompt
    assert '작성 지시문' in prompt
    assert '각 후보의 정의가 독립적으로 성립한 뒤에만 우선순위' in prompt
    assert '+final-selection-guided-v4' in config['prompt_version']


def test_final_reselection_payload_keeps_guidance_but_excludes_truth_and_first_choice(tmp_path):
    image = tmp_path / 'a.png'
    image.write_bytes(b'png')
    config = gemma.default_config(binary_prompt='guideline_direct', label_reflection=True,
                                  type_reflection=True, final_reflection=True)
    candidates = _types('중복형', '불충분형')
    data = {
        **_metadata(), '_binary_assessment': {'verdict': '부적절'},
        '_candidate_assessment': candidates, '_initial_final_assessment': _final(),
        '_final_reflection_feedback': {'feedback': '두 유형의 경계를 다시 확인하라.'},
        '_actual_label': '불충분형', '_final_reflection_document': '# 비밀 문서',
    }

    payload = gemma.request_payload('final_reflection_fallback', image, data, config)
    supplied = _user_json(payload)

    assert supplied['candidate_assessment'] == candidates
    assert 'initial_final_assessment' not in supplied
    assert supplied['reflection_feedback'] == {'feedback': '두 유형의 경계를 다시 확인하라.'}
    assert 'actual_label' not in supplied
    assert supplied['current_guidelines'] == '# 비밀 문서'
    assert payload['response_format']['json_schema']['name'] == 'final_error_type'
    assert payload['response_format']['json_schema']['schema']['properties']['label']['enum'] == [
        '중복형', '불충분형']


def test_frozen_final_feedback_payload_never_contains_evaluation_label(tmp_path):
    image = tmp_path / 'a.png'
    image.write_bytes(b'png')
    sha = 'a' * 64
    config = gemma.default_config(
        binary_prompt='guideline_direct', label_reflection=True, type_reflection=True,
        final_reflection=True, reflection_mode=gemma.REFLECTION_FROZEN,
        reflection_guideline_source_run='train-run', reflection_guideline_sha256='b' * 64,
        type_reflection_guideline_sha256='c' * 64, final_reflection_guideline_sha256=sha)
    data = {
        **_metadata(), '_binary_assessment': {'verdict': '부적절'},
        '_candidate_assessment': _types('중복형', '불충분형'),
        '_initial_final_assessment': _final(), '_actual_label': '불충분형',
        '_final_reflection_document': '# 고정 최종 기준',
    }

    payload = gemma.request_payload('final_reflection_feedback', image, data, config)
    supplied = _user_json(payload)

    assert 'actual_label' not in supplied
    assert supplied['current_guidelines'] == '# 고정 최종 기준'
    assert payload['response_format']['json_schema']['name'] == 'reflection_feedback_gate'


def test_train_final_reflection_skips_non_applicable_truth_missing_and_initial_correct(tmp_path):
    candidates = _types('중복형', '불충분형')

    class Server:
        config = {'reflection_mode': gemma.REFLECTION_TRAIN, 'final_reflection': True}

        def __getattr__(self, name):
            if name.startswith('infer_final_reflection'):
                raise AssertionError('생략 조건에서는 최종 Reflection을 호출하면 안 됩니다.')
            raise AttributeError(name)

    for truth, initial, reason in [
            ('적절', _final('중복형'), 'actual_label_appropriate'),
            ('무관형', _final('중복형'), 'actual_label_not_in_candidates'),
            ('중복형', _final('중복형'), 'initial_final_correct')]:
        final, calls, fields = run_ocr._final_reflection_chain(
            Server(), tmp_path, 'a', tmp_path / 'a.png', _metadata(), truth,
            {'verdict': '부적절'}, candidates, initial)
        assert final == initial
        assert calls == []
        assert fields['initial_final_decision'] == initial
        assert fields['final_reflection_skipped'] == reason


def test_final_reflection_skips_a_single_fixed_candidate(tmp_path):
    document = reflection.read_final_document(tmp_path)

    class Server:
        config = {'reflection_mode': gemma.REFLECTION_FROZEN, 'final_reflection': True,
                  'label_priority': list(gemma.ERROR_PRIORITY),
                  'final_reflection_guideline_sha256': hashlib.sha256(document.encode()).hexdigest()}

        def __getattr__(self, name):
            if name.startswith('infer_final_reflection'):
                raise AssertionError('후보가 하나면 최종 Reflection을 호출하면 안 됩니다.')
            raise AttributeError(name)

    initial = _final('중복형')
    final, calls, fields = run_ocr._final_reflection_chain(
        Server(), tmp_path, 'a', tmp_path / 'a.png', _metadata(), '불충분형',
        {'verdict': '부적절'}, _types('중복형'), initial)

    assert final == initial
    assert calls == []
    assert fields['final_reflection_skipped'] == 'single_candidate'
    assert fields['final_reflection_decision_source'] == 'single_candidate_v1'


def test_frozen_final_metadata_gate_handles_repeated_model_boundary_failures(tmp_path):
    cases = [
        ({'alt_text': '해당 이미지 아래 설명 추가', 'in_link': False, 'link_dest': ''},
         _types('무의미형', '불충분형'), '불충분형'),
        ({'alt_text': '미리보기', 'in_link': True, 'link_dest': '소식지 미리보기'},
         _types('무의미형', '불충분형'), '불충분형'),
        ({'alt_text': 'external_image', 'in_link': False, 'link_dest': ''},
         _types('파일명형', '무의미형'), '무의미형'),
        ({'alt_text': '중구청', 'in_link': True, 'link_dest': '중구청 대전 중구청 홈페이지'},
         _types('중복형', '불충분형'), '불충분형'),
        ({'alt_text': '콘텐츠(3)', 'in_link': False, 'link_dest': ''},
         _types('무의미형', '파일명형'), '무의미형'),
        ({'alt_text': '7페이지_영화필름 사진', 'in_link': False, 'link_dest': ''},
         _types('파일명형', '불충분형'), '불충분형'),
        ({'alt_text': '캡차이미지', 'in_link': False, 'link_dest': ''},
         _types('무의미형', '불충분형'), '불충분형'),
    ]
    for metadata, candidates, expected in cases:
        decision = gemma.final_metadata_gate(metadata, candidates)
        assert decision['label'] == expected

    assert gemma.final_metadata_gate(
        {'alt_text': '중구청', 'in_link': True, 'link_dest': '중구청 중구청'},
        _types('중복형', '불충분형')) is None
    assert gemma.final_metadata_gate(
        {'alt_text': '미리보기', 'in_link': True, 'link_dest': '소식지'},
        _types('무의미형')) is None
    assert gemma.final_metadata_gate(
        {'alt_text': 'Boramae Park img', 'in_link': True, 'link_dest': 'Boramae Park'},
        _types('파일명형', '무의미형', '불충분형'), _final('무의미형'))['label'] == '불충분형'
    assert gemma.final_metadata_gate(
        {'alt_text': '북스타그램 이미지', 'in_link': True, 'link_dest': '북스타그램'},
        _types('무의미형', '중복형', '불충분형'), _final('중복형')) is None


def test_frozen_final_metadata_gate_skips_reflection_calls(tmp_path):
    document = reflection.read_final_document(tmp_path)

    class Server:
        config = {'reflection_mode': gemma.REFLECTION_FROZEN, 'final_reflection': True,
                  'label_priority': list(gemma.ERROR_PRIORITY),
                  'final_reflection_guideline_sha256': hashlib.sha256(document.encode()).hexdigest()}

        def __getattr__(self, name):
            if name.startswith('infer_final_reflection'):
                raise AssertionError('구조 판정 후 Reflection을 호출하면 안 됩니다.')
            raise AttributeError(name)

    metadata = {**_metadata(), 'alt_text': '해당 이미지 아래 설명 추가'}
    final, calls, fields = run_ocr._final_reflection_chain(
        Server(), tmp_path, 'a', tmp_path / 'a.png', metadata, '불충분형',
        {'verdict': '부적절'}, _types('무의미형', '불충분형'), _final('무의미형'))

    assert final['label'] == '불충분형'
    assert calls == []
    assert fields['final_reflection_skipped'] == 'deterministic_final_metadata_gate'


def test_train_appropriate_truth_that_reaches_final_selector_is_auditable_and_cacheable(
        tmp_path, monkeypatch):
    image = tmp_path / 'inputs/a.png'
    image.parent.mkdir()
    image.write_bytes(b'png')
    config = gemma.default_config(binary_prompt='guideline_direct', label_reflection=True,
                                  type_reflection=True, final_reflection=True)
    answers = {
        'binary_verdict': [
            {'verdict': '부적절', 'reason': '최초 이진 근거',
             'image_evidence': '이미지 근거', 'context_evidence': '문맥 근거'},
            {'verdict': '부적절', 'reason': '재판정 이진 근거',
             'image_evidence': '이미지 근거', 'context_evidence': '문맥 근거'},
        ],
        'error_types': [_types('중복형', '불충분형')],
        'final_error_type': [_final('중복형')],
        'reflection_feedback': [{'feedback': '적절 여부를 다시 확인하라.'}],
        'reflection_document': [{
            'markdown': '# Reflection 판단 가이드라인\n\n- 적절 여부를 다시 확인한다.\n'}],
    }

    def respond(path, payload, timeout):
        contract = payload['response_format']['json_schema']['name']
        value = answers[contract].pop(0)
        return {'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps(value, ensure_ascii=False)}}]}

    server = gemma.GemmaServer(config, 'gpu:1', tmp_path)
    monkeypatch.setattr(server, 'http', respond)
    metadata = _metadata()
    binary, first = server.infer_binary(
        'classify_no_ocr', image, metadata, tmp_path / 'raw/gemma_no_ocr/a.json', 0)
    value, stats = run_ocr._reflection_chain(
        server, tmp_path, 'a', image, metadata, '적절', binary, first)

    assert value['label'] == '중복형'
    assert value['final_reflection_skipped'] == 'actual_label_appropriate'
    assert [call['stage'] for call in stats['calls']] == [
        'classify_no_ocr', 'reflection_feedback', 'reflection_fallback',
        'reflection_document', 'types_no_ocr', 'final_no_ocr']
    common.write_json(tmp_path / 'records.json', [
        {'record_id': 'a', 'label': '적절', **metadata}])
    saved = {**value, 'status': 'ok', 'record_id': 'a', 'input_path': 'inputs/a.png',
             'fingerprint': 'key', 'raw_paths': stats['raw_paths']}
    common.write_json(common.result_path(tmp_path, 'gemma_no_ocr', 'a'), saved)

    assert benchmark._single_audit_error(
        tmp_path, 'gemma_no_ocr', 'a', saved, config) is None
    assert run_ocr._cached_stage(tmp_path, 'gemma_no_ocr', 'a', 'key') == saved


def test_train_final_reflection_reselects_inside_fixed_candidates_and_updates_document(tmp_path):
    candidates = _types('중복형', '불충분형')
    initial = _final('중복형')
    revised = _final('불충분형')
    seen = []

    class Server:
        config = {'reflection_mode': gemma.REFLECTION_TRAIN, 'final_reflection': True,
                  'reflection_document_compact_threshold': 5000}

        def infer_final_reflection_feedback(self, image, data, path):
            seen.append(('feedback', data))
            return {'feedback': '핵심 정보 누락과 문맥 반복의 경계를 다시 확인하라.'}, _stats('final_reflection_feedback')

        def infer_final_reflection_fallback(self, image, data, path):
            seen.append(('fallback', data))
            assert data['_candidate_assessment'] == candidates
            assert '_actual_label' not in data
            assert data['_final_reflection_feedback']['feedback']
            assert data['_final_reflection_document'].startswith('# 최종 오류')
            return revised, _stats('final_reflection_fallback')

        def infer_final_reflection_document(self, image, data, path):
            seen.append(('document', data))
            return {'markdown': '# 최종 오류 유형 선택 Reflection 가이드라인\n\n- 후보 경계를 확인한다.\n'}, _stats('final_reflection_document')

    final, calls, fields = run_ocr._final_reflection_chain(
        Server(), tmp_path, 'a', tmp_path / 'a.png', _metadata(), '불충분형',
        {'verdict': '부적절'}, candidates, initial)

    assert final == revised
    assert [call['stage'] for call in calls] == [
        'final_reflection_feedback', 'final_reflection_fallback', 'final_reflection_document']
    assert seen[0][0] == 'feedback'
    assert {name for name, _ in seen[1:]} == {'fallback', 'document'}
    assert fields['final_reflection_decision'] == revised
    assert fields['final_reflection_document_status'] == 'ok'
    assert '후보 경계' in reflection.read_final_document(tmp_path)


def test_frozen_final_reflection_passes_or_reselects_without_truth(tmp_path):
    candidates = _types('중복형', '불충분형')
    initial = _final('중복형')
    document = '# 최종 오류 유형 선택 Reflection 가이드라인\n\n- 고정 기준\n'
    reflection.write_final_document(tmp_path, document)

    class Server:
        def __init__(self, action):
            self.action = action
            self.config = {
                'reflection_mode': gemma.REFLECTION_FROZEN, 'final_reflection': True,
                'final_reflection_guideline_sha256': hashlib.sha256(document.encode()).hexdigest(),
                'frozen_feedback_policy': gemma.FROZEN_FEEDBACK_GATE,
            }

        def infer_final_reflection_feedback(self, image, data, path):
            assert '_actual_label' not in data
            feedback = '' if self.action == '통과' else '후보 경계를 다시 확인하라.'
            return {'action': self.action, 'feedback': feedback, 'reason': '검토 근거'}, _stats('final_reflection_feedback')

        def infer_final_reflection_fallback(self, image, data, path):
            assert data['_candidate_assessment'] == candidates
            assert '_actual_label' not in data
            assert data['_final_reflection_feedback']['feedback'] == '후보 경계를 다시 확인하라.'
            assert data['_final_reflection_document'] == document
            return _final('불충분형'), _stats('final_reflection_fallback')

    passed, calls, fields = run_ocr._final_reflection_chain(
        Server('통과'), tmp_path, 'a', tmp_path / 'a.png', _metadata(), '실제 라벨을 보내지 않음',
        {'verdict': '부적절'}, candidates, initial)
    assert passed == initial
    assert [call['stage'] for call in calls] == ['final_reflection_feedback']
    assert fields['final_reflection_skipped'] == 'frozen_feedback_not_needed'

    revised, calls, fields = run_ocr._final_reflection_chain(
        Server('피드백 제공'), tmp_path, 'a', tmp_path / 'a.png', _metadata(), '실제 라벨을 보내지 않음',
        {'verdict': '부적절'}, candidates, initial)
    assert revised['label'] == '불충분형'
    assert [call['stage'] for call in calls] == [
        'final_reflection_feedback', 'final_reflection_fallback']
    assert fields['final_reflection_decision_source'] == gemma.REFLECTION_FROZEN


def test_full_chain_runs_initial_selector_before_final_reflection(tmp_path, monkeypatch):
    image = tmp_path / 'inputs/a.png'
    image.parent.mkdir()
    image.write_bytes(b'png')
    config = gemma.default_config(binary_prompt='guideline_direct', label_reflection=True,
                                  type_reflection=True, final_reflection=True)
    answers = {
        'binary_verdict': [{'verdict': '부적절', 'reason': '이진 근거',
                            'image_evidence': '이미지 근거', 'context_evidence': '문맥 근거'}],
        'error_types': [_types('중복형', '불충분형')],
        'final_error_type': [_final('중복형'), _final('불충분형')],
        'reflection_feedback': [{'feedback': '후보 경계를 다시 확인하라.'}],
        'reflection_document': [{
            'markdown': '# 최종 오류 유형 선택 Reflection 가이드라인\n\n- 후보 경계를 확인한다.\n'}],
    }
    contracts = []

    def respond(path, payload, timeout):
        contract = payload['response_format']['json_schema']['name']
        contracts.append(contract)
        value = answers[contract].pop(0)
        return {'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps(value, ensure_ascii=False)}}]}

    server = gemma.GemmaServer(config, 'gpu:1', tmp_path)
    monkeypatch.setattr(server, 'http', respond)
    metadata = _metadata()
    binary, first = server.infer_binary(
        'classify_no_ocr', image, metadata, tmp_path / 'raw/gemma_no_ocr/a.json', 0)
    value, stats = run_ocr._reflection_chain(
        server, tmp_path, 'a', image, metadata, '불충분형', binary, first)

    assert value['initial_final_decision']['label'] == '중복형'
    assert value['final_reflection_decision']['label'] == '불충분형'
    assert value['final_decision']['label'] == '불충분형'
    assert value['label'] == '불충분형'
    assert contracts.count('final_error_type') == 2
    assert [call['stage'] for call in stats['calls']] == [
        'classify_no_ocr', 'types_no_ocr', 'final_no_ocr',
        'final_reflection_feedback', 'final_reflection_fallback',
        'final_reflection_document']
    common.write_json(tmp_path / 'records.json', [
        {'record_id': 'a', 'label': '불충분형', **metadata}])
    audited = {**value, 'status': 'ok', 'record_id': 'a', 'input_path': 'inputs/a.png'}
    assert benchmark._single_audit_error(
        tmp_path, 'gemma_no_ocr', 'a', audited, config) is None


def test_final_reflection_report_separates_reachable_selector_and_full_scores(tmp_path):
    config = gemma.default_config(binary_prompt='guideline_direct', label_reflection=True,
                                  type_reflection=True, final_reflection=True)
    common.write_json(tmp_path / 'manifest.json', {
        'record_ids': ['a', 'b', 'c', 'd'], 'pipeline_mode': 'gemma_only',
        'models': [], 'gemma': config})
    common.write_json(tmp_path / 'records.json', [
        {'record_id': 'a', 'label': '불충분형'},
        {'record_id': 'b', 'label': '중복형'},
        {'record_id': 'c', 'label': '무관형'},
        {'record_id': 'd', 'label': '적절'},
    ])
    rows = [
        ('a', _types('중복형', '불충분형'), _final('중복형'), _final('불충분형')),
        ('b', _types('중복형', '불충분형'), _final('중복형'), _final('불충분형')),
        ('c', _types('중복형'), _final('중복형'), _final('중복형')),
    ]
    for rid, candidates, before, after in rows:
        common.write_json(common.result_path(tmp_path, 'gemma_no_ocr', rid), {
            'status': 'ok', 'record_id': rid, 'verdict': '부적절', 'label': after['label'],
            'matched_types': candidates['matched_types'], 'type_classification': candidates,
            'binary_decision': {'verdict': '부적절'}, 'reflection_decision': {'verdict': '부적절'},
            'initial_final_decision': before, 'final_decision': after,
            'final_reflection_decision': after,
        })
    common.write_json(common.result_path(tmp_path, 'gemma_no_ocr', 'd'), {
        'status': 'ok', 'record_id': 'd', 'verdict': '적절', 'label': '적절',
        'matched_types': [], 'binary_decision': {'verdict': '적절'},
        'reflection_decision': {'verdict': '적절'}, 'final_decision': None,
    })

    report = benchmark.build_benchmark(tmp_path, audit_requests=False)['final_reflection']
    metrics = report['available']['without_ocr']

    assert metrics['reachable_count'] == 2
    assert metrics['truth_missing_count'] == 1
    assert metrics['initial_correct_count'] == 1
    assert metrics['final_correct_count'] == 1
    assert metrics['corrected'] == 1
    assert metrics['harmed'] == 1
    assert metrics['initial_accuracy'] == metrics['final_accuracy'] == .5
    assert metrics['full_before']['count'] == metrics['full_after']['count'] == 4
    assert metrics['full_before']['binary_f1'] == metrics['full_after']['binary_f1']
