import hashlib
import json

import benchmark
import common
import gemma
import reflection
import run_ocr


def _types(*labels):
    return {
        'matched_types': [{'label': label, 'reason': f'{label} 근거'} for label in labels],
        'reason': '후보 근거', 'image_evidence': '이미지 근거', 'context_evidence': '문맥 근거',
    }


def _stats(stage):
    return {'stage': stage, 'usage': {}, 'elapsed_seconds': .1,
            'raw_path': f'raw/gemma_no_ocr/a.{stage}.json'}


def _binary(verdict):
    return {'verdict': verdict, 'reason': '이진 근거',
            'image_evidence': '이미지 근거', 'context_evidence': '문맥 근거'}


def test_type_reflection_document_is_separate_and_atomic(tmp_path):
    binary = reflection.read_document(tmp_path)
    subtype = reflection.read_type_document(tmp_path)

    assert binary.startswith('# Reflection 판단 가이드라인')
    assert subtype.startswith('# 오류 유형 후보 Reflection 가이드라인')
    assert reflection.FILENAME != reflection.TYPE_FILENAME

    revised = '# 오류 유형 후보 Reflection 가이드라인\n\n- 후보 기준\n'
    reflection.write_type_document(tmp_path, revised)
    assert reflection.read_type_document(tmp_path) == revised
    assert reflection.read_document(tmp_path) == binary


def test_train_type_reflection_runs_only_when_truth_is_missing(tmp_path):
    initial = _types('중복형')
    revised = _types('중복형', '불충분형')

    class Server:
        config = {'reflection_mode': gemma.REFLECTION_TRAIN,
                  'type_reflection': True,
                  'reflection_document_compact_threshold': 5000}

        def infer_type_reflection_feedback(self, *args):
            return {'feedback': '핵심 정보 누락 여부를 다시 확인하라.'}, _stats('type_reflection_feedback')

        def infer_type_reflection_fallback(self, *args):
            return revised, _stats('type_reflection_fallback')

        def infer_type_reflection_document(self, *args):
            return {'markdown': '# 오류 유형 후보 Reflection 가이드라인\n\n- 핵심 정보 누락 여부를 확인한다.\n'}, _stats('type_reflection_document')

    final, calls, fields = run_ocr._type_reflection_chain(
        Server(), tmp_path, 'a', tmp_path / 'a.png', {'alt_text': '행사'},
        '불충분형', {'verdict': '부적절'}, initial)

    assert final == revised
    assert [call['stage'] for call in calls] == [
        'type_reflection_feedback', 'type_reflection_fallback', 'type_reflection_document']
    assert fields['initial_type_classification'] == initial
    assert fields['type_reflection_decision'] == revised
    assert fields['type_reflection_feedback']['feedback'].startswith('핵심 정보')
    assert fields['type_reflection_document_status'] == 'ok'
    assert '핵심 정보' in reflection.read_type_document(tmp_path)


def test_train_type_reflection_skips_candidate_hit_and_appropriate_truth(tmp_path):
    initial = _types('불충분형')

    class Server:
        config = {'reflection_mode': gemma.REFLECTION_TRAIN, 'type_reflection': True}

        def __getattr__(self, name):
            if name.startswith('infer_type_reflection'):
                raise AssertionError('후보 정답 문제에는 후보 Reflection을 호출하면 안 됩니다.')
            raise AttributeError(name)

    for truth, reason in [('불충분형', 'initial_candidate_hit'), ('적절', 'actual_label_appropriate')]:
        final, calls, fields = run_ocr._type_reflection_chain(
            Server(), tmp_path, 'a', tmp_path / 'a.png', {'alt_text': '행사'},
            truth, {'verdict': '부적절'}, initial)
        assert final == initial
        assert calls == []
        assert fields['type_reflection_skipped'] == reason
        assert fields['initial_type_classification'] == initial


def test_frozen_type_reflection_uses_fixed_md_and_can_pass(tmp_path):
    initial = _types('중복형')
    document = '# 오류 유형 후보 Reflection 가이드라인\n\n- 고정 기준\n'
    reflection.write_type_document(tmp_path, document)

    class Server:
        config = {'reflection_mode': gemma.REFLECTION_FROZEN,
                  'type_reflection': True,
                  'type_candidate_policy': gemma.TYPE_CANDIDATE_RECALL,
                  'type_reflection_guideline_sha256': hashlib.sha256(document.encode()).hexdigest(),
                  'frozen_feedback_policy': gemma.FROZEN_FEEDBACK_GATE}

        def infer_type_reflection_feedback(self, *args):
            return {'action': '통과', 'feedback': '', 'reason': '적용 기준 없음'}, _stats('type_reflection_feedback')

        def infer_type_reflection_fallback(self, *args):
            raise AssertionError('통과이면 후보를 다시 생성하면 안 됩니다.')

    final, calls, fields = run_ocr._type_reflection_chain(
        Server(), tmp_path, 'a', tmp_path / 'a.png', {'alt_text': '행사'},
        None, {'verdict': '부적절'}, initial)

    assert final == initial
    assert [call['stage'] for call in calls] == ['type_reflection_feedback']
    assert fields['type_reflection_skipped'] == 'frozen_feedback_not_needed'
    assert fields['type_reflection_document_sha256'] == hashlib.sha256(document.encode()).hexdigest()


def test_frozen_type_reflection_only_expands_and_never_removes_initial_candidates(tmp_path):
    initial = _types('중복형', '불충분형')
    fallback = _types('무의미형')
    document = '# 오류 유형 후보 Reflection 가이드라인\n\n- 빠진 후보를 확인한다.\n'
    reflection.write_type_document(tmp_path, document)

    class Server:
        config = {'reflection_mode': gemma.REFLECTION_FROZEN,
                  'type_reflection': True,
                  'type_candidate_policy': gemma.TYPE_CANDIDATE_RECALL,
                  'type_reflection_guideline_sha256': hashlib.sha256(document.encode()).hexdigest(),
                  'frozen_feedback_policy': gemma.FROZEN_FEEDBACK_GATE}

        def infer_type_reflection_feedback(self, *args):
            return {'action': '피드백 제공', 'feedback': '빠진 후보를 확인하라.',
                    'reason': '누락 가능성'}, _stats('type_reflection_feedback')

        def infer_type_reflection_fallback(self, *args):
            return fallback, _stats('type_reflection_fallback')

    final, _, fields = run_ocr._type_reflection_chain(
        Server(), tmp_path, 'a', tmp_path / 'a.png', {'alt_text': '행사'},
        None, {'verdict': '부적절'}, initial)

    assert [item['label'] for item in final['matched_types']] == [
        '중복형', '불충분형', '무의미형']
    assert fields['type_reflection_decision'] == final


def test_type_reflection_prompts_define_a_candidate_expander_not_a_filter():
    config = gemma.default_config(
        binary_prompt='guideline_direct', label_reflection=True, type_reflection=True)

    feedback = config['prompts']['type_reflection_feedback']
    fallback = config['prompts']['type_reflection_fallback']
    assert '빠진 후보를 추가할 필요가 있는지' in feedback
    assert '잘못 포함된 후보를 제거' not in feedback
    assert 'initial_candidate_assessment의 후보를 모두 유지한다' in fallback
    assert '후보를 삭제하거나 교체하지 않는다' in fallback
    assert '+type-candidate-recall-v2' in config['prompt_version']
    assert config['type_candidate_policy'] == gemma.TYPE_CANDIDATE_RECALL


def test_frozen_candidate_only_stops_after_reflected_candidate_pool(tmp_path):
    sha = '0' * 64
    config = gemma.default_config(
        binary_prompt='guideline_direct', label_reflection=True,
        reflection_mode=gemma.REFLECTION_FROZEN,
        reflection_guideline_source_run='train-run',
        reflection_guideline_sha256=sha,
        candidate_only=True, type_reflection=True,
        type_reflection_guideline_sha256=sha,
    )
    server = gemma.GemmaServer(config, 'gpu:3', tmp_path)
    initial = _types('중복형')
    expanded = _types('중복형', '불충분형')

    def infer_once(stage, *args, **kwargs):
        assert stage == 'types_no_ocr'
        return initial, _stats(stage)

    def reflect(types, binary):
        assert types == initial
        assert binary['verdict'] == '부적절'
        return expanded, [_stats('type_reflection_feedback')], {
            'initial_type_classification': initial,
            'type_reflection_decision': expanded,
        }

    server._infer_once = infer_once
    value, stats = server.infer_after_binary(
        'classify_no_ocr', tmp_path / 'input.png', {'alt_text': '행사'},
        tmp_path / 'raw/gemma_no_ocr/a.json', _binary('부적절'),
        _stats('classify_no_ocr'), type_reflector=reflect)

    assert config['candidate_only'] is True
    assert not config.get('final_reflection')
    assert value['candidate_only'] is True
    assert value['label'] is None
    assert value['matched_types'] == expanded['matched_types']
    assert value['type_classification'] == expanded
    assert value['final_decision'] is None
    assert [call['stage'] for call in stats['calls']] == [
        'classify_no_ocr', 'types_no_ocr', 'type_reflection_feedback']


def test_frozen_candidate_only_chain_is_auditable_without_final_request(tmp_path, monkeypatch):
    image = tmp_path / 'inputs/a.png'
    image.parent.mkdir()
    image.write_bytes(b'png')
    binary_document = '# Reflection 판단 가이드라인\n\n- 이진 기준\n'
    type_document = '# 오류 유형 후보 Reflection 가이드라인\n\n- 후보 기준\n'
    binary_sha = hashlib.sha256(binary_document.encode()).hexdigest()
    type_sha = hashlib.sha256(type_document.encode()).hexdigest()
    config = gemma.default_config(
        binary_prompt='guideline_direct', label_reflection=True,
        reflection_mode=gemma.REFLECTION_FROZEN,
        reflection_guideline_source_run='train-run',
        reflection_guideline_sha256=binary_sha,
        candidate_only=True, type_reflection=True,
        type_reflection_guideline_sha256=type_sha,
    )
    reflection.write_document(tmp_path, binary_document)
    reflection.write_type_document(tmp_path, type_document)
    initial = _types('중복형')
    added = _types('불충분형')
    answers = {
        'visual_evidence': [{
            'role': '정보 전달', 'reason': '행사 이미지다.',
            'visible_facts': ['행사 정보'], 'context_quotes': [],
            'required_information': ['행사 정보'],
        }],
        'binary_verdict': [_binary('부적절'), _binary('적절')],
        'reflection_feedback_gate': [
            {'action': '통과', 'feedback': '', 'reason': '이진 재검토 불필요'},
            {'action': '피드백 제공', 'feedback': '누락 후보를 다시 확인하라.',
             'reason': '후보 누락 가능성'},
        ],
        'error_types': [initial, added],
    }

    def respond(path, payload, timeout):
        contract = payload['response_format']['json_schema']['name']
        value = answers[contract].pop(0)
        return {'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps(value, ensure_ascii=False)}}]}

    server = gemma.GemmaServer(config, 'gpu:3', tmp_path)
    monkeypatch.setattr(server, 'http', respond)
    metadata = {key: None for key in gemma.METADATA_FIELDS}
    metadata.update(alt_text='행사', in_link=False)
    binary, first = server.infer_binary(
        'classify_no_ocr', image, metadata, tmp_path / 'raw/gemma_no_ocr/a.json', 0)
    value, stats = run_ocr._reflection_chain(
        server, tmp_path, 'a', image, metadata, None, binary, first)

    assert value['label'] is None
    assert value['candidate_only'] is True
    assert [item['label'] for item in value['matched_types']] == ['중복형', '불충분형']
    assert not (tmp_path / 'raw/gemma_no_ocr/a.final.json').exists()

    common.write_json(tmp_path / 'manifest.json', {
        'record_ids': ['a'], 'pipeline_mode': 'gemma_only', 'models': [], 'gemma': config})
    common.write_json(tmp_path / 'records.json', [
        {'record_id': 'a', 'label': '불충분형', **metadata}])
    common.write_json(common.result_path(tmp_path, 'gemma_no_ocr', 'a'), {
        **value, 'status': 'ok', 'record_id': 'a', 'input_path': 'inputs/a.png',
        'calls': stats['calls'], 'raw_paths': stats['raw_paths']})

    report = benchmark.build_benchmark(tmp_path)
    assert report['pairing_excluded'] == {}
    assert report['type_reflection']['available']['without_ocr']['after_hit_rate'] == 1


def test_candidate_reflection_report_is_inclusion_rate_not_f1(tmp_path):
    config = gemma.default_config(binary_prompt='guideline_direct', label_reflection=True,
                                  type_reflection=True)
    common.write_json(tmp_path / 'manifest.json', {
        'record_ids': ['a', 'b', 'c'], 'pipeline_mode': 'gemma_only', 'models': [], 'gemma': config})
    common.write_json(tmp_path / 'records.json', [
        {'record_id': 'a', 'label': '불충분형'},
        {'record_id': 'b', 'label': '중복형'},
        {'record_id': 'c', 'label': '적절'},
    ])
    for rid, truth, before, after in [
            ('a', '불충분형', _types('중복형'), _types('중복형', '불충분형')),
            ('b', '중복형', _types('중복형'), _types('중복형')),
            ('c', '적절', _types('무관형'), _types('무관형'))]:
        common.write_json(common.result_path(tmp_path, 'gemma_no_ocr', rid), {
            'status': 'ok', 'record_id': rid, 'verdict': '부적절',
            'label': after['matched_types'][0]['label'],
            'matched_types': after['matched_types'], 'type_classification': after,
            'initial_type_classification': before,
            'type_reflection_decision': after,
            'binary_decision': {'verdict': '부적절'},
            'reflection_decision': {'verdict': '부적절'},
        })

    report = benchmark.build_benchmark(tmp_path, audit_requests=False)['type_reflection']
    metrics = report['available']['without_ocr']

    assert metrics == {
        'count': 2, 'before_hit_count': 1, 'before_hit_rate': .5,
        'after_hit_count': 2, 'after_hit_rate': 1.0,
        'after_miss_count': 0, 'before_mean_size': 1.0, 'after_mean_size': 1.5,
        'corrected': 1, 'harmed': 0,
    }
    assert all('f1' not in key.lower() for key in metrics)


def test_full_train_chain_reflects_candidates_then_calls_final_selector_once(tmp_path, monkeypatch):
    image = tmp_path / 'inputs/a.png'
    image.parent.mkdir()
    image.write_bytes(b'png')
    config = gemma.default_config(binary_prompt='guideline_direct', label_reflection=True,
                                  type_reflection=True)
    initial = _types('중복형')
    fallback = _types('불충분형')
    revised = _types('중복형', '불충분형')
    answers = {
        'binary_verdict': [_binary('부적절')],
        'error_types': [initial, fallback],
        'reflection_feedback': [{'feedback': '핵심 정보 누락 여부를 다시 확인하라.'}],
        'reflection_document': [{
            'markdown': '# 오류 유형 후보 Reflection 가이드라인\n\n- 핵심 정보 누락 여부를 확인한다.\n'}],
        'final_error_type': [{
            'label': '불충분형', 'reason': '최종 근거',
            'image_evidence': '이미지 근거', 'context_evidence': '문맥 근거'}],
    }
    calls = []

    def respond(path, payload, timeout):
        contract = payload['response_format']['json_schema']['name']
        calls.append(contract)
        value = answers[contract].pop(0)
        return {'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps(value, ensure_ascii=False)}}]}

    server = gemma.GemmaServer(config, 'gpu:1', tmp_path)
    monkeypatch.setattr(server, 'http', respond)
    metadata = {key: None for key in gemma.METADATA_FIELDS}
    metadata.update(alt_text='행사', in_link=False)
    binary, first = server.infer_binary(
        'classify_no_ocr', image, metadata, tmp_path / 'raw/gemma_no_ocr/a.json', 0)
    value, stats = run_ocr._reflection_chain(
        server, tmp_path, 'a', image, metadata, '불충분형', binary, first)

    assert value['initial_type_classification'] == initial
    assert value['type_classification'] == revised
    assert value['label'] == '불충분형'
    assert calls.count('final_error_type') == 1
    assert [call['stage'] for call in stats['calls']] == [
        'classify_no_ocr', 'types_no_ocr', 'type_reflection_feedback',
        'type_reflection_fallback', 'type_reflection_document', 'final_no_ocr']
    common.write_json(tmp_path / 'manifest.json', {
        'record_ids': ['a'], 'pipeline_mode': 'gemma_only', 'models': [], 'gemma': config})
    common.write_json(tmp_path / 'records.json', [
        {'record_id': 'a', 'label': '불충분형', **metadata}])
    common.write_json(common.result_path(tmp_path, 'gemma_no_ocr', 'a'), {
        **value, 'status': 'ok', 'record_id': 'a', 'input_path': 'inputs/a.png',
        'calls': stats['calls'], 'raw_paths': stats['raw_paths']})

    report = benchmark.build_benchmark(tmp_path)
    assert report['pairing_excluded'] == {}
    assert report['type_reflection']['available']['without_ocr']['after_hit_rate'] == 1
