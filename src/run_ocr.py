"""Run image + metadata Gemma classification. Legacy CLI filename; no OCR execution."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
import signal
import sys
import time
import traceback

from common import (DEFAULT_DATASET, acquire_worker_lock, create_run, load_manifest, load_records,
                    load_result, now, prepare_image, read_json, resolve_image, result_path,
                    run_elapsed_seconds, write_json)
from gemma import (DECORATIVE_STRUCTURED_GATE, EMPTY_ALT_LINK_GATE,
                   EMPTY_ALT_LINK_VISUAL_REVIEW, FROZEN_FEEDBACK_GATE,
                   FROZEN_EVIDENCE_PIPELINE,
                   LINK_STRUCTURE_GATE, SAME_LINK_ALT_REDUNDANCY_GATE,
                   ERROR_PRIORITY, LABELS, REFLECTION_ERRORS_ONLY,
                   REFLECTION_FROZEN, REFLECTION_TRAIN, TYPE_CANDIDATE_RECALL,
                   GemmaServer, INPUT_VERSION, METADATA_FIELDS, default_config,
                   decorative_observation_decision, enforce_final_candidate_pool,
                   expand_type_candidates, fingerprint, final_candidate_labels,
                   final_metadata_gate, frozen_coverage_feedback, link_structure_gate,
                   uses_empty_alt_link_visual_review, validate_config)
from benchmark import save_benchmark
from reflection import (read_document, read_final_document, read_frozen_document,
                        read_frozen_final_document, read_frozen_type_document,
                        read_type_document, write_document, write_final_document,
                        write_type_document)

def _input(run_dir, manifest, row):
    path = run_dir / 'inputs' / (row['record_id'] + '.png')
    meta_path = path.with_suffix('.json')
    if path.exists() and meta_path.exists():
        return path, read_json(meta_path)
    original = resolve_image(row.get('_dataset_dir', manifest['dataset_dir']), row)
    image, metadata = prepare_image(original, manifest['background'])
    metadata.update({'source_file': str(original), 'source_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
                     'width': image.width, 'height': image.height})
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    image.save(temp, format='PNG')
    os.replace(temp, path)
    write_json(meta_path, metadata)
    return path, metadata

def _cached_stage(run_dir, stage, rid, key):
    result = load_result(run_dir, stage, rid)
    raw_paths = result.get('raw_paths') if result else None
    if result and result.get('binary_decision'):
        if result.get('decorative_gate_selected'):
            expected = 3  # initial binary + alt-hidden router + structured observer
        else:
            expected = 1 if result.get('binary_only') or result.get('verdict') == '적절' else 2
            expected += bool(result.get('role_decision'))
            expected += bool(result.get('review_decision'))
            expected += bool(result.get('role_review_decision'))
            expected += bool(result.get('initial_type_classification')
                             and not result.get('type_reflection_decision'))
            expected += bool(result.get('final_decision')
                             and result.get('final_selection_source') != 'single_candidate_v1')
            if (result.get('reflection_decision')
                    and result.get('reflection_skipped') not in (
                        'deterministic_binary_gate', 'initial_binary_correct')):
                if result.get('reflection_decision_source') == REFLECTION_FROZEN:
                    if result.get('reflection_visual_evidence'):
                        expected += 2
                    expected += (0 if result.get('reflection_feedback_source') == FROZEN_EVIDENCE_PIPELINE
                                 else 1)
                    expected += result.get('reflection_skipped') != 'frozen_feedback_not_needed'
                else:
                    expected += 2 if result.get('reflection_document_status') == 'error' else 3
            if result.get('type_reflection_decision'):
                skipped = result.get('type_reflection_skipped')
                if skipped not in ('initial_candidate_hit', 'actual_label_appropriate'):
                    if result.get('type_reflection_decision_source') == REFLECTION_FROZEN:
                        expected += 1 if skipped == 'frozen_feedback_not_needed' else 2
                    else:
                        expected += 2 if result.get('type_reflection_document_status') == 'error' else 3
            if result.get('final_reflection_decision'):
                skipped = result.get('final_reflection_skipped')
                if skipped not in ('single_candidate', 'actual_label_appropriate',
                                   'actual_label_not_in_candidates', 'initial_final_correct'):
                    if result.get('final_reflection_decision_source') == REFLECTION_FROZEN:
                        expected += 1 if skipped == 'frozen_feedback_not_needed' else 2
                    else:
                        expected += 2 if result.get('final_reflection_document_status') == 'error' else 3
            expected += bool(result.get('decorative_router_decision'))
            expected += bool(result.get('decorative_observation'))
        if not isinstance(raw_paths, list) or len(raw_paths) != expected:
            return None
    if (result and result.get('status') == 'ok' and result.get('fingerprint') == key
            and raw_paths
            and all((run_dir / path).is_file() and (run_dir / path).with_suffix('.request.json').is_file()
                    for path in raw_paths)):
        return result
    return None

def _inference_audit(stats, default_path):
    """Keep pre-v6 test doubles and saved v5 runner contracts readable."""
    return stats.get('calls', [stats]), stats.get('raw_paths', [default_path])


def _gate_audit(run_dir, rid, metadata, decision, source=EMPTY_ALT_LINK_GATE):
    path = run_dir / 'raw' / 'gemma_no_ocr' / f'{rid}.gate.json'
    write_json(path.with_suffix('.request.json'), {
        'decision_source': source,
        'input': {key: metadata.get(key) for key in METADATA_FIELDS},
    })
    write_json(path, decision)
    return {'stage': 'binary_gate', 'usage': {}, 'elapsed_seconds': 0,
            'raw_path': str(path.relative_to(run_dir))}


def _gate_appropriate_result(config, decision, gate_stats, source=EMPTY_ALT_LINK_GATE):
    binary_only = bool(config.get('binary_only'))
    candidate_only = bool(config.get('candidate_only'))
    partial = binary_only or candidate_only
    return {'verdict': '적절', 'label': None if partial else '적절', 'matched_types': [],
            'assessment_reason': decision['reason'], 'reason': decision['reason'],
            'image_evidence': decision['image_evidence'],
            'context_evidence': decision['context_evidence'],
            'decision_rule': {'name': source,
                              'priority': list(config['label_priority'])},
            'binary_decision': decision, 'binary_decision_source': source,
            'binary_route': 'link_structure_gate', 'type_classification': None,
            'final_decision': None, 'priority_label': None if partial else '적절',
            **({'binary_only': True} if binary_only else {}),
            **({'candidate_only': True} if candidate_only else {}),
            'usage': {}, 'elapsed_seconds': 0,
            'calls': [gate_stats], 'raw_paths': [gate_stats['raw_path']]}


def _decorative_route(server, image_path, metadata, audit_path):
    """Run the validated alt-hidden router and conditional structured observer."""
    if not str(metadata.get('alt_text') or '').strip():
        return None
    router_path = audit_path.with_name(audit_path.stem + '.decorative_router.json')
    router, router_stats = server.infer_decorative_router(
        image_path, metadata, router_path, 1)
    observation = observer_stats = None
    if router['verdict'] == '적절':
        observer_path = audit_path.with_name(audit_path.stem + '.decorative_observer.json')
        observation, observer_stats = server.infer_decorative_observer(
            image_path, metadata, observer_path, 1)
    calls = [router_stats] + ([observer_stats] if observer_stats else [])
    return {
        'selected': bool(observation and decorative_observation_decision(observation, metadata)),
        'router': router,
        'observation': observation,
        'calls': calls,
    }


def _call_stats(calls):
    return {
        'usage': {key: sum(call.get('usage', {}).get(key, 0) for call in calls)
                  for key in ('prompt_tokens', 'completion_tokens')},
        'elapsed_seconds': round(sum(call.get('elapsed_seconds', 0) for call in calls), 3),
        'calls': calls,
        'raw_paths': [call['raw_path'] for call in calls],
    }


def _decorative_gate_result(config, initial, first_stats, route):
    """Build the terminal seven-class result for a structured decorative match."""
    router = route['router']
    observation = route['observation']
    reason = observation['reason']
    gate_binary = {
        'verdict': '부적절', 'reason': reason,
        'image_evidence': router['image_evidence'],
        'context_evidence': router['context_evidence'],
    }
    candidates = {
        'matched_types': [{'label': '장식오용형', 'reason': reason}],
        'reason': reason,
        'image_evidence': router['image_evidence'],
        'context_evidence': router['context_evidence'],
    }
    final = {
        'label': '장식오용형', 'reason': reason,
        'image_evidence': router['image_evidence'],
        'context_evidence': router['context_evidence'],
    }
    value = {
        'verdict': '부적절', 'label': '장식오용형',
        'matched_types': candidates['matched_types'],
        'assessment_reason': reason, 'reason': reason,
        'image_evidence': router['image_evidence'],
        'context_evidence': router['context_evidence'],
        'decision_rule': {'name': DECORATIVE_STRUCTURED_GATE,
                          'priority': list(config['label_priority'])},
        'binary_decision': initial,
        'reflection_decision': gate_binary,
        'reflection_feedback': None,
        'reflection_decision_source': DECORATIVE_STRUCTURED_GATE,
        'reflection_skipped': 'decorative_structured_gate',
        'type_classification': candidates,
        'final_decision': final,
        'priority_label': '장식오용형',
        'final_selection_source': DECORATIVE_STRUCTURED_GATE,
        'decorative_router_decision': router,
        'decorative_observation': observation,
        'decorative_gate_selected': True,
    }
    return value, _call_stats([first_stats, *route['calls']])


def _attach_decorative_route(value, stats, route):
    """Keep rejected route evidence and its calls on the unchanged main result."""
    if route is None:
        return value, stats
    value.update(decorative_router_decision=route['router'],
                 decorative_observation=route['observation'],
                 decorative_gate_selected=False)
    calls = list(stats.get('calls', [stats]))
    calls[1:1] = route['calls']
    return value, _call_stats(calls)


def _type_reflection_chain(server, run_dir, rid, image_path, metadata, actual_label,
                           binary, initial_types):
    """Reflect only the six-class candidate pool; the final selector stays unchanged."""
    run_dir = Path(run_dir)
    config = getattr(server, 'config', {})
    mode = config.get('reflection_mode', REFLECTION_TRAIN)
    document = read_type_document(run_dir)
    digest = hashlib.sha256(document.encode()).hexdigest()
    labels = {item['label'] for item in initial_types['matched_types']}
    fields = {
        'initial_type_classification': initial_types,
        'type_reflection_decision': initial_types,
        'type_reflection_feedback': None,
        'type_reflection_document_sha256': digest,
    }
    if mode == REFLECTION_TRAIN:
        if actual_label == '적절':
            fields.update(type_reflection_skipped='actual_label_appropriate',
                          type_reflection_decision_source='train_not_applicable_v1',
                          type_reflection_document_status='skipped')
            return initial_types, [], fields
        if actual_label not in ERROR_PRIORITY:
            raise ValueError('Train 후보 Reflection 실제 오류 유형이 잘못되었습니다.')
        if actual_label in labels:
            fields.update(type_reflection_skipped='initial_candidate_hit',
                          type_reflection_decision_source='train_initial_candidate_hit_v1',
                          type_reflection_document_status='skipped')
            return initial_types, [], fields

    base = run_dir / 'raw' / 'gemma_no_ocr' / f'{rid}'
    feedback_data = {
        **metadata,
        '_binary_assessment': binary,
        '_candidate_assessment': initial_types,
        '_type_reflection_document': document,
    }
    if mode == REFLECTION_TRAIN:
        feedback_data['_actual_label'] = actual_label
    feedback, feedback_stats = server.infer_type_reflection_feedback(
        image_path, feedback_data, base.with_suffix('.type_reflection_feedback.json'))
    fields['type_reflection_feedback'] = feedback

    if mode == REFLECTION_FROZEN:
        passed = (config.get('frozen_feedback_policy') == FROZEN_FEEDBACK_GATE
                  and feedback.get('action') == '통과')
        revised, calls = initial_types, [feedback_stats]
        if not passed:
            revised, fallback_stats = server.infer_type_reflection_fallback(image_path, {
                **metadata,
                '_binary_assessment': binary,
                '_candidate_assessment': initial_types,
                '_type_reflection_feedback': {'feedback': feedback['feedback']},
            }, base.with_suffix('.type_reflection_fallback.json'))
            if config.get('type_candidate_policy') == TYPE_CANDIDATE_RECALL:
                revised = expand_type_candidates(initial_types, revised)
            calls.append(fallback_stats)
        fields.update(
            type_reflection_decision=revised,
            type_reflection_decision_source=REFLECTION_FROZEN,
            **({'type_reflection_skipped': 'frozen_feedback_not_needed'} if passed else {}),
        )
        return revised, calls, fields

    with ThreadPoolExecutor(max_workers=2) as pool:
        fallback_future = pool.submit(server.infer_type_reflection_fallback, image_path, {
            **metadata,
            '_binary_assessment': binary,
            '_candidate_assessment': initial_types,
            '_type_reflection_feedback': feedback,
        }, base.with_suffix('.type_reflection_fallback.json'))
        document_future = pool.submit(server.infer_type_reflection_document, image_path, {
            '_type_reflection_document': document,
            '_type_reflection_feedback': feedback,
        }, base.with_suffix('.type_reflection_document.json'))
        revised, fallback_stats = fallback_future.result()
        if config.get('type_candidate_policy') == TYPE_CANDIDATE_RECALL:
            revised = expand_type_candidates(initial_types, revised)
        try:
            updated, document_stats = document_future.result()
            write_type_document(run_dir, updated['markdown'])
            document_status = 'ok'
            document_error = None
            document_hash = hashlib.sha256(updated['markdown'].encode()).hexdigest()
        except Exception as exc:
            document_stats = None
            document_status = 'error'
            document_error = f'{type(exc).__name__}: {exc}'
            document_hash = digest
    calls = [feedback_stats, fallback_stats]
    if document_stats is not None:
        calls.append(document_stats)
    fields.update(
        type_reflection_decision=revised,
        type_reflection_decision_source='type_label_reflection_v1',
        type_reflection_document_sha256=document_hash,
        type_reflection_document_status=document_status,
    )
    if document_error:
        fields['type_reflection_document_error'] = document_error
    return revised, calls, fields


def _final_reflection_chain(server, run_dir, rid, image_path, metadata, actual_label,
                            binary, candidates, initial_final):
    """Reflect the final choice without changing the fixed candidate pool."""
    run_dir = Path(run_dir)
    config = getattr(server, 'config', {})
    mode = config.get('reflection_mode', REFLECTION_TRAIN)
    document = (read_frozen_final_document(
        run_dir, config['final_reflection_guideline_sha256'])
        if mode == REFLECTION_FROZEN else read_final_document(run_dir))
    digest = hashlib.sha256(document.encode()).hexdigest()
    labels = {item['label'] for item in candidates['matched_types']}
    fields = {
        'initial_final_decision': initial_final,
        'final_reflection_feedback': None,
        'final_reflection_decision': initial_final,
        'final_reflection_document_sha256': digest,
    }
    if len(final_candidate_labels(candidates)) == 1:
        fields.update(final_reflection_skipped='single_candidate',
                      final_reflection_decision_source='single_candidate_v1',
                      final_reflection_document_status='skipped')
        return initial_final, [], fields
    if mode == REFLECTION_FROZEN:
        structured = final_metadata_gate(metadata, candidates, initial_final)
        if structured is not None:
            fields.update(final_reflection_skipped='deterministic_final_metadata_gate',
                          final_reflection_decision=structured,
                          final_reflection_decision_source='final_metadata_gate_v1',
                          final_reflection_document_status='frozen')
            return structured, [], fields
    if mode == REFLECTION_TRAIN:
        if actual_label == '적절':
            fields.update(final_reflection_skipped='actual_label_appropriate',
                          final_reflection_decision_source='train_not_applicable_v1',
                          final_reflection_document_status='skipped')
            return initial_final, [], fields
        if actual_label not in ERROR_PRIORITY:
            raise ValueError('Train 최종 선택 Reflection 실제 오류 유형이 잘못되었습니다.')
        if actual_label not in labels:
            fields.update(final_reflection_skipped='actual_label_not_in_candidates',
                          final_reflection_decision_source='train_truth_unreachable_v1',
                          final_reflection_document_status='skipped')
            return initial_final, [], fields
        if initial_final.get('label') == actual_label:
            fields.update(final_reflection_skipped='initial_final_correct',
                          final_reflection_decision_source='train_initial_final_correct_v1',
                          final_reflection_document_status='skipped')
            return initial_final, [], fields

    base = run_dir / 'raw' / 'gemma_no_ocr' / rid
    feedback_data = {
        **metadata,
        '_binary_assessment': binary,
        '_candidate_assessment': candidates,
        '_initial_final_assessment': initial_final,
        '_final_reflection_document': document,
    }
    if mode == REFLECTION_TRAIN:
        feedback_data['_actual_label'] = actual_label
    feedback, feedback_stats = server.infer_final_reflection_feedback(
        image_path, feedback_data, base.with_suffix('.final_reflection_feedback.json'))
    fields['final_reflection_feedback'] = feedback

    fallback_data = {
        **metadata,
        '_binary_assessment': binary,
        '_candidate_assessment': candidates,
        '_initial_final_assessment': initial_final,
        '_final_reflection_feedback': {'feedback': feedback['feedback']},
        '_final_reflection_document': document,
    }
    if mode == REFLECTION_FROZEN:
        passed = (config.get('frozen_feedback_policy') == FROZEN_FEEDBACK_GATE
                  and feedback.get('action') == '통과')
        revised, calls = initial_final, [feedback_stats]
        if not passed:
            revised, fallback_stats = server.infer_final_reflection_fallback(
                image_path, fallback_data, base.with_suffix('.final_reflection_fallback.json'))
            calls.append(fallback_stats)
        revised, recovery = enforce_final_candidate_pool(
            revised, candidates, config.get('label_priority', list(ERROR_PRIORITY)))
        fields.update(
            final_reflection_decision=revised,
            final_reflection_decision_source=REFLECTION_FROZEN,
            final_reflection_document_status='frozen',
            **({'final_reflection_contract_recovery': recovery} if recovery else {}),
            **({'final_reflection_skipped': 'frozen_feedback_not_needed'} if passed else {}),
        )
        return revised, calls, fields

    document_data = {
        '_final_reflection_document': document,
        '_final_reflection_feedback': feedback,
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        fallback_future = pool.submit(
            server.infer_final_reflection_fallback, image_path, fallback_data,
            base.with_suffix('.final_reflection_fallback.json'))
        document_future = pool.submit(
            server.infer_final_reflection_document, image_path, document_data,
            base.with_suffix('.final_reflection_document.json'))
        revised, fallback_stats = fallback_future.result()
        revised, recovery = enforce_final_candidate_pool(
            revised, candidates, config.get('label_priority', list(ERROR_PRIORITY)))
        try:
            updated, document_stats = document_future.result()
            write_final_document(run_dir, updated['markdown'])
            document_status = 'ok'
            document_error = None
            document_hash = hashlib.sha256(updated['markdown'].encode()).hexdigest()
        except Exception as exc:
            document_stats = None
            document_status = 'error'
            document_error = f'{type(exc).__name__}: {exc}'
            document_hash = digest
    calls = [feedback_stats, fallback_stats]
    if document_stats is not None:
        calls.append(document_stats)
    fields.update(
        final_reflection_decision=revised,
        final_reflection_decision_source='final_label_reflection_v1',
        final_reflection_document_sha256=document_hash,
        final_reflection_document_status=document_status,
        **({'final_reflection_contract_recovery': recovery} if recovery else {}),
    )
    if document_error:
        fields['final_reflection_document_error'] = document_error
    return revised, calls, fields


def _reflection_chain(server, run_dir, rid, image_path, metadata, actual_label,
                      initial, first_stats, initial_source=None):
    """Apply one train-label or frozen-guideline feedback cycle."""
    run_dir = Path(run_dir)
    current_document = read_document(run_dir)
    base = run_dir / 'raw' / 'gemma_no_ocr' / f'{rid}.json'
    config = getattr(server, 'config', {})
    reflection_mode = config.get('reflection_mode', REFLECTION_TRAIN)
    type_reflector = (lambda candidates, binary: _type_reflection_chain(
        server, run_dir, rid, image_path, metadata, actual_label, binary, candidates)
        ) if config.get('type_reflection') else None
    final_reflector = (lambda initial_final, candidates, binary: _final_reflection_chain(
        server, run_dir, rid, image_path, metadata, actual_label, binary,
        candidates, initial_final)
        ) if config.get('final_reflection') else None
    deterministic_gate = (
        initial_source == SAME_LINK_ALT_REDUNDANCY_GATE
        or (config.get('reflection_gate_policy') == 'deterministic_final_v1'
            and initial_source == EMPTY_ALT_LINK_GATE)
    )
    if deterministic_gate:
        value, stats = server.infer_after_binary(
            'classify_no_ocr', image_path, metadata, base, initial, first_stats,
            initial_source, 0, type_reflector, final_reflector)
        value.update(
            binary_decision=initial,
            reflection_decision=initial,
            reflection_feedback=None,
            reflection_decision_source='deterministic_gate_v1',
            reflection_skipped='deterministic_binary_gate',
            reflection_document_sha256=hashlib.sha256(current_document.encode()).hexdigest(),
        )
        return value, stats
    if (reflection_mode == REFLECTION_TRAIN
            and config.get('reflection_training_route') == REFLECTION_ERRORS_ONLY):
        actual_verdict = '적절' if actual_label == '적절' else '부적절'
        if actual_label not in LABELS or initial.get('verdict') not in ('적절', '부적절'):
            raise ValueError('Train Reflection 이진 비교 입력이 잘못되었습니다.')
        if initial['verdict'] == actual_verdict:
            value, stats = server.infer_after_binary(
                'classify_no_ocr', image_path, metadata, base, initial, first_stats,
                initial_source, 0, type_reflector, final_reflector)
            value.update(
                binary_decision=initial,
                reflection_decision=initial,
                reflection_feedback=None,
                reflection_decision_source='train_initial_binary_correct_v1',
                reflection_skipped='initial_binary_correct',
                reflection_document_sha256=hashlib.sha256(current_document.encode()).hexdigest(),
                reflection_document_status='skipped',
            )
            return value, stats
    feedback_input = {
        **metadata,
        '_initial_assessment': initial,
        '_reflection_document': current_document,
    }
    if reflection_mode == REFLECTION_TRAIN:
        feedback_input['_actual_label'] = actual_label
    visual_evidence = coverage_audit = None
    visual_stats = coverage_stats = None
    feedback_source = None
    if (reflection_mode == REFLECTION_FROZEN
            and config.get('frozen_evidence_pipeline') == FROZEN_EVIDENCE_PIPELINE):
        visual_evidence, visual_stats = server.infer_reflection_visual_evidence(
            image_path, metadata,
            base.with_name(f'{rid}.reflection_visual_evidence.json'))
        coverage_audit, coverage_stats = server.infer_reflection_coverage_audit(
            image_path, {**metadata, '_initial_assessment': visual_evidence},
            base.with_name(f'{rid}.reflection_coverage_audit.json'))
    if coverage_audit and coverage_audit['verdict'] == '부적절':
        feedback = frozen_coverage_feedback(coverage_audit)
        feedback_stats = None
        feedback_source = FROZEN_EVIDENCE_PIPELINE
    else:
        feedback, feedback_stats = server.infer_reflection_feedback(
            image_path, feedback_input, base.with_name(f'{rid}.reflection_feedback.json'))
        if visual_evidence is not None:
            feedback_source = REFLECTION_FROZEN
    if reflection_mode == REFLECTION_FROZEN:
        passed = (config.get('frozen_feedback_policy') == FROZEN_FEEDBACK_GATE
                  and feedback.get('action') == '통과')
        current, current_stats = initial, first_stats
        fallback_stats = None
        if not passed:
            fallback_feedback = ({'feedback': feedback['feedback']}
                                 if config.get('frozen_feedback_policy') == FROZEN_FEEDBACK_GATE
                                 else feedback)
            current, fallback_stats = server.infer_reflection_fallback(image_path, {
                **metadata,
                '_initial_assessment': initial,
                '_reflection_feedback': fallback_feedback,
                '_reflection_document': current_document,
            }, base.with_name(f'{rid}.reflection_fallback.json'))
            current_stats = fallback_stats
        value, after_stats = server.infer_after_binary(
            'classify_no_ocr', image_path, metadata, base, current, current_stats,
            REFLECTION_FROZEN, 0, type_reflector, final_reflector)
        calls = [first_stats]
        calls.extend(call for call in (visual_stats, coverage_stats, feedback_stats) if call is not None)
        if fallback_stats is not None:
            calls.append(fallback_stats)
        calls.extend(after_stats['calls'][1:])
        value.update(
            binary_decision=initial,
            reflection_decision=current,
            reflection_feedback=feedback,
            reflection_decision_source=REFLECTION_FROZEN,
            reflection_document_sha256=hashlib.sha256(current_document.encode()).hexdigest(),
            **({'reflection_visual_evidence': visual_evidence,
                'reflection_coverage_audit': coverage_audit,
                'reflection_feedback_source': feedback_source}
               if visual_evidence is not None else {}),
            **({'reflection_skipped': 'frozen_feedback_not_needed'} if passed else {}),
        )
        if initial_source:
            value['binary_decision_source'] = initial_source
        else:
            value.pop('binary_decision_source', None)
        return value, {
            'usage': {key: sum(call.get('usage', {}).get(key, 0) for call in calls)
                      for key in ('prompt_tokens', 'completion_tokens')},
            'elapsed_seconds': round(sum(call.get('elapsed_seconds', 0) for call in calls), 3),
            'calls': calls,
            'raw_paths': [call['raw_path'] for call in calls],
        }
    # Both calls depend on the generated feedback but not on each other. Keep
    # records sequential for cumulative-MD consistency while using both llama
    # slots concurrently inside each record.
    with ThreadPoolExecutor(max_workers=2) as pool:
        fallback_future = pool.submit(server.infer_reflection_fallback, image_path, {
            **metadata,
            '_initial_assessment': initial,
            '_reflection_feedback': feedback,
        }, base.with_name(f'{rid}.reflection_fallback.json'))
        document_future = pool.submit(server.infer_reflection_document, image_path, {
            '_reflection_document': current_document,
            '_reflection_feedback': feedback,
        }, base.with_name(f'{rid}.reflection_document.json'))
        fallback, fallback_stats = fallback_future.result()
        try:
            revised, document_stats = document_future.result()
            write_document(run_dir, revised['markdown'])
            document_status = 'ok'
            document_error = None
            document_hash = hashlib.sha256(revised['markdown'].encode()).hexdigest()
        except Exception as exc:
            # MD maintenance is auxiliary. Preserve the last valid document and
            # keep this record's binary/subtype predictions usable.
            document_stats = None
            document_status = 'error'
            document_error = f'{type(exc).__name__}: {exc}'
            document_hash = hashlib.sha256(current_document.encode()).hexdigest()
    value, after_stats = server.infer_after_binary(
        'classify_no_ocr', image_path, metadata, base, fallback, fallback_stats,
        'label_reflection_v1', 0, type_reflector, final_reflector)
    calls = [first_stats, feedback_stats, after_stats['calls'][0]]
    if document_stats is not None:
        calls.append(document_stats)
    calls.extend(after_stats['calls'][1:])
    value.update(
        binary_decision=initial,
        reflection_decision=fallback,
        reflection_feedback=feedback,
        reflection_decision_source='label_reflection_v1',
        reflection_document_sha256=document_hash,
        reflection_document_status=document_status,
    )
    if document_error:
        value['reflection_document_error'] = document_error
    if initial_source:
        value['binary_decision_source'] = initial_source
    else:
        value.pop('binary_decision_source', None)
    return value, {
        'usage': {key: sum(call.get('usage', {}).get(key, 0) for call in calls)
                  for key in ('prompt_tokens', 'completion_tokens')},
        'elapsed_seconds': round(sum(call.get('elapsed_seconds', 0) for call in calls), 3),
        'calls': calls,
        'raw_paths': [call['raw_path'] for call in calls],
    }

def _model_identity(config):
    return {key: {'path': config[key], 'size': Path(config[key]).stat().st_size,
                  'mtime_ns': Path(config[key]).stat().st_mtime_ns}
            for key in ('model_path', 'mmproj_path', 'server_path')}


def _result_config(config):
    """Runtime concurrency does not change prompts, model weights, or predictions."""
    return {key: value for key, value in config.items()
            if key not in ('execution_mode', 'parallel_slots')}


def _slot_futures(items, slots, work, stop_check):
    """Run at most one problem per llama.cpp slot and reuse that slot for its whole task."""
    iterator = iter(items)
    pool = ThreadPoolExecutor(max_workers=slots)
    pending = {}
    try:
        for slot_id in range(slots):
            item = next(iterator, None)
            if item is not None:
                pending[pool.submit(work, item, slot_id)] = (item, slot_id)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                item, slot_id = pending.pop(future)
                yield item, slot_id, future
                stop_check()
                next_item = next(iterator, None)
                if next_item is not None:
                    pending[pool.submit(work, next_item, slot_id)] = (next_item, slot_id)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _execute_record_parallel(run_dir, manifest, config, identity, rows, stack, state, update, stop_check):
    slots = int(config['parallel_slots'])
    state.update(execution_mode='record_parallel_v1', parallel_slots=slots)
    plans = []
    update(phase='preparing')
    for rid in manifest['record_ids']:
        stop_check()
        try:
            path, _ = _input(run_dir, manifest, rows[rid])
            metadata = {key: rows[rid].get(key) for key in METADATA_FIELDS}
            key = fingerprint({'variant': 'without_ocr', 'input_version': INPUT_VERSION,
                               'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                               'metadata': metadata, 'config': _result_config(config),
                               'model_files': identity})
            if _cached_stage(run_dir, 'gemma_no_ocr', rid, key):
                state['completed'] += 1
                update(record_id=rid)
                continue
            gate, gate_source = link_structure_gate(
                metadata, config.get('binary_gate'),
                legacy_readable_link=config.get('guideline_mode') == 'sections_1_3_v1')
            visual_review = bool(config.get('guideline_mode') in (
                                     'sections_1_3_v1_0', 'sections_1_3_5_v1_1')
                                 and config.get('binary_gate') in (
                                     EMPTY_ALT_LINK_GATE, LINK_STRUCTURE_GATE)
                                 and uses_empty_alt_link_visual_review(metadata))
            if gate and gate['verdict'] == '적절':
                gate_stats = _gate_audit(run_dir, rid, metadata, gate, gate_source)
                value = _gate_appropriate_result(config, gate, gate_stats, gate_source)
                value.update(status='ok', record_id=rid, input_path=f'inputs/{rid}.png',
                             fingerprint=key, finished_at=now())
                write_json(result_path(run_dir, 'gemma_no_ocr', rid), value)
                state['completed'] += 1
                update(record_id=rid)
                continue
            plans.append({'rid': rid, 'path': path, 'metadata': metadata, 'fingerprint': key,
                          'relative': f'raw/gemma_no_ocr/{rid}.json', 'gate': gate,
                          'gate_source': gate_source, 'visual_review': visual_review})
        except Exception as exc:
            write_json(result_path(run_dir, 'gemma_no_ocr', rid), {
                'status': 'error', 'record_id': rid, 'error': f'{type(exc).__name__}: {exc}',
                'finished_at': now()})
            state['completed'] += 1
            state['errors'] += 1
            update(record_id=rid)

    if not plans:
        return
    update(phase='gemma_loading')
    server = stack.enter_context(GemmaServer(config, manifest['device'], run_dir))
    update(phase='alt_classification')

    def classify(plan, slot_id):
        if plan['gate']:
            binary = plan['gate']
            source = plan['gate_source']
            first = _gate_audit(run_dir, plan['rid'], plan['metadata'], binary, source)
        else:
            binary_stage = ('empty_alt_link_visual_review' if plan['visual_review']
                            else 'classify_no_ocr')
            binary, first = server.infer_binary(
                binary_stage, plan['path'], plan['metadata'], run_dir / plan['relative'], slot_id)
            source = EMPTY_ALT_LINK_VISUAL_REVIEW if plan['visual_review'] else None
        value, stats = server.infer_after_binary(
            'classify_no_ocr', plan['path'], plan['metadata'], run_dir / plan['relative'],
            binary, first, source, slot_id)
        calls, raw_paths = _inference_audit(stats, plan['relative'])
        value.update(calls=calls, raw_paths=raw_paths, status='ok', record_id=plan['rid'],
                     input_path=f"inputs/{plan['rid']}.png", fingerprint=plan['fingerprint'],
                     finished_at=now())
        return value

    for plan, slot_id, future in _slot_futures(plans, slots, classify, stop_check):
        rid = plan['rid']
        try:
            value = future.result()
        except Exception as exc:
            value = {'status': 'error', 'record_id': rid, 'fingerprint': plan['fingerprint'],
                     'error': f'{type(exc).__name__}: {exc}', 'finished_at': now()}
            state['errors'] += 1
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        write_json(result_path(run_dir, 'gemma_no_ocr', rid), value)
        state['completed'] += 1
        update(record_id=rid, slot_id=slot_id)
        print(f"{state['completed']}/{state['total']} slot {slot_id} {rid}: "
              f"{value.get('label') or value.get('verdict') or value['status']}", flush=True)


def execute_gemma(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest = load_manifest(run_dir)
    state = {'state': 'running', 'pid': os.getpid(), 'model': 'gemma', 'pipeline_mode': 'gemma_only',
             'completed': 0, 'total': len(manifest['record_ids']), 'errors': 0, 'started_at': now()}
    def update(**values):
        state.update(values, updated_at=now())
        write_json(run_dir / 'status.json', state)
    def stop_check():
        if (run_dir / 'STOP').exists():
            raise InterruptedError('중지 요청')
    update(phase='preparing')
    try:
        if manifest.get('pipeline_mode') != 'gemma_only' or manifest.get('models'):
            raise ValueError('과거 OCR 실험은 열람만 지원합니다. Gemma 단독 새 실행을 만드세요.')
        config = manifest['gemma']
        if config.get('input_mode') != 'image_metadata':
            raise ValueError('이미지·메타데이터 전용 새 실행을 만드세요.')
        validate_config(config)
        stop_check()
        identity = _model_identity(config)
        rows = {r['record_id']: r for r in read_json(run_dir / 'records.json')}
        if config.get('label_reflection'):
            if config.get('reflection_mode', REFLECTION_TRAIN) == REFLECTION_FROZEN:
                read_frozen_document(run_dir, config['reflection_guideline_sha256'])
                if config.get('type_reflection'):
                    read_frozen_type_document(
                        run_dir, config['type_reflection_guideline_sha256'])
                if config.get('final_reflection'):
                    read_frozen_final_document(
                        run_dir, config['final_reflection_guideline_sha256'])
            else:
                unlabeled = [rid for rid in manifest['record_ids'] if rows[rid].get('label') not in LABELS]
                if unlabeled:
                    raise ValueError(f'Train Reflection은 정답이 있는 데이터에서만 실행할 수 있습니다: {len(unlabeled)}개 라벨 없음')
                read_document(run_dir)
                if config.get('type_reflection'):
                    read_type_document(run_dir)
                if config.get('final_reflection'):
                    read_final_document(run_dir)
        with ExitStack() as stack:
            if config.get('execution_mode') == 'record_parallel_v1':
                _execute_record_parallel(run_dir, manifest, config, identity, rows, stack,
                                         state, update, stop_check)
                save_benchmark(run_dir)
                update(state='completed_with_errors' if state['errors'] else 'completed', phase='finished')
                return 0
            server = None
            for rid in manifest['record_ids']:
                stop_check()
                update(record_id=rid, phase='preparing')
                try:
                    path, _ = _input(run_dir, manifest, rows[rid])
                    metadata = {key: rows[rid].get(key) for key in METADATA_FIELDS}
                    key = fingerprint({'variant': 'without_ocr', 'input_version': INPUT_VERSION,
                                       'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                       'metadata': metadata, 'config': _result_config(config),
                                       'model_files': identity})
                    if _cached_stage(run_dir, 'gemma_no_ocr', rid, key):
                        update(completed=state['completed'] + 1)
                        continue
                except Exception as exc:
                    write_json(result_path(run_dir, 'gemma_no_ocr', rid), {
                        'status': 'error', 'record_id': rid, 'error': f'{type(exc).__name__}: {exc}', 'finished_at': now()})
                    update(completed=state['completed'] + 1, errors=state['errors'] + 1)
                    continue
                stop_check()
                update(phase='alt_classification')
                try:
                    relative = f'raw/gemma_no_ocr/{rid}.json'
                    gate, gate_source = link_structure_gate(
                        metadata, config.get('binary_gate'),
                        legacy_readable_link=config.get('guideline_mode') == 'sections_1_3_v1')
                    visual_review = bool(config.get('guideline_mode') in (
                                             'sections_1_3_v1_0', 'sections_1_3_5_v1_1')
                                         and config.get('binary_gate') in (
                                             EMPTY_ALT_LINK_GATE, LINK_STRUCTURE_GATE)
                                         and uses_empty_alt_link_visual_review(metadata))
                    gate_stats = (_gate_audit(run_dir, rid, metadata, gate, gate_source)
                                  if gate else None)
                    if config.get('label_reflection'):
                        if server is None:
                            update(phase='gemma_loading')
                            server = stack.enter_context(GemmaServer(config, manifest['device'], run_dir))
                            update(phase='alt_classification')
                        if gate:
                            initial, first, source = gate, gate_stats, gate_source
                            decorative_route = None
                        elif (config.get('decorative_gate')
                              and str(metadata.get('alt_text') or '').strip()):
                            with ThreadPoolExecutor(max_workers=2) as pool:
                                binary_future = pool.submit(
                                    server.infer_binary, 'classify_no_ocr', path, metadata,
                                    run_dir / relative, 0)
                                decorative_future = pool.submit(
                                    _decorative_route, server, path, metadata, run_dir / relative)
                                initial, first = binary_future.result()
                                decorative_route = decorative_future.result()
                            source = None
                        else:
                            binary_stage = ('empty_alt_link_visual_review' if visual_review
                                            else 'classify_no_ocr')
                            initial, first = server.infer_binary(
                                binary_stage, path, metadata, run_dir / relative, 0)
                            source = EMPTY_ALT_LINK_VISUAL_REVIEW if visual_review else None
                            decorative_route = None
                        if decorative_route and decorative_route['selected']:
                            value, stats = _decorative_gate_result(
                                config, initial, first, decorative_route)
                        else:
                            update(phase='reflection_feedback')
                            value, stats = _reflection_chain(
                                server, run_dir, rid, path, metadata, rows[rid].get('label'),
                                initial, first, source)
                            value, stats = _attach_decorative_route(
                                value, stats, decorative_route)
                        update(phase='alt_classification')
                        calls, raw_paths = _inference_audit(stats, relative)
                        value.update(calls=calls, raw_paths=raw_paths)
                    elif gate and gate['verdict'] == '적절':
                        value = _gate_appropriate_result(
                            config, gate, gate_stats, gate_source)
                    else:
                        if server is None:
                            update(phase='gemma_loading')
                            server = stack.enter_context(GemmaServer(config, manifest['device'], run_dir))
                            update(phase='alt_classification')
                        if gate:
                            value, stats = server.infer_from_binary(
                                'classify_no_ocr', path, metadata, run_dir / relative,
                                gate, gate_stats, gate_source)
                        elif visual_review:
                            binary, first = server.infer_binary(
                                'empty_alt_link_visual_review', path, metadata,
                                run_dir / relative, 0)
                            value, stats = server.infer_after_binary(
                                'classify_no_ocr', path, metadata, run_dir / relative,
                                binary, first, EMPTY_ALT_LINK_VISUAL_REVIEW, 0)
                        else:
                            value, stats = server.infer('classify_no_ocr', path, metadata, run_dir / relative)
                        calls, raw_paths = _inference_audit(stats, relative)
                        value.update(calls=calls, raw_paths=raw_paths)
                    value.update(status='ok', record_id=rid, input_path=f'inputs/{rid}.png', fingerprint=key,
                                 finished_at=now())
                except (InterruptedError, KeyboardInterrupt):
                    raise
                except Exception as exc:
                    value = {'status': 'error', 'record_id': rid, 'error': f'{type(exc).__name__}: {exc}', 'finished_at': now()}
                    state['errors'] += 1
                    traceback.print_exc()
                write_json(result_path(run_dir, 'gemma_no_ocr', rid), value)
                update(completed=state['completed'] + 1)
                print(f"{state['completed']}/{state['total']} {rid}: "
                      f"{value.get('label') or value.get('verdict') or value['status']}", flush=True)
        save_benchmark(run_dir)
        update(state='completed_with_errors' if state['errors'] else 'completed', phase='finished')
        return 0
    except (InterruptedError, KeyboardInterrupt):
        update(state='stopped', phase='stopped')
        save_benchmark(run_dir)
        return 0
    except Exception as exc:
        update(state='failed', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        return 1


def execute_pipeline(run_dir, lock_fd=None):
    # Keep elapsed time across resume attempts; the descriptor is held by the caller.
    run_dir = Path(run_dir)
    elapsed = run_elapsed_seconds(run_dir)
    timing = {'elapsed_seconds': elapsed, 'started_at': now(), 'running': True, 'estimated': False}
    write_json(run_dir / 'timing.json', timing)
    started = time.monotonic()
    try:
        return execute_gemma(run_dir)
    finally:
        timing.update(elapsed_seconds=elapsed + max(0, time.monotonic() - started),
                      running=False, finished_at=now())
        write_json(run_dir / 'timing.json', timing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='Resume an image + metadata run')
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--limit', type=int, help='First N records; default: all records')
    parser.add_argument('--ids', nargs='+', help='Explicit record IDs')
    parser.add_argument('--background', choices=['auto_gray', 'white', 'dark'], default='auto_gray')
    parser.add_argument('--device', default='gpu:2', help='Physical NVIDIA GPU index; no CPU inference')
    parser.add_argument('--lock-fd', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--gemma-max-tokens', type=int, default=4096)
    parser.add_argument('--gemma-context', type=int, default=24576)
    parser.add_argument('--binary-prompt', choices=('guideline_direct', 'evidence_split', 'role_split', 'criteria', 'basic'),
                        default='guideline_direct')
    parser.add_argument('--review-policy', choices=('appropriate_v1', 'evidence_uncertain_v1'), default='appropriate_v1',
                        help='v12 fallback: review every initial appropriate, or only model-flagged uncertainty')
    parser.add_argument('--recheck-appropriate', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--label-reflection', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--empty-alt-prompt', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--compare-review-role', action=argparse.BooleanOptionalAction, default=None,
                        help='Compare fallback with/without the shared role evidence; defaults on for v12 appropriate_v1')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    def terminated(signum, frame):
        raise KeyboardInterrupt('작업 중지 신호')
    signal.signal(signal.SIGTERM, terminated)
    try:
        lock = os.fdopen(args.lock_fd, 'a') if args.lock_fd is not None else acquire_worker_lock()
        with lock:
            run_dir = args.run
            if run_dir is None:
                rows = load_records(args.dataset)
                ids = args.ids or [r['record_id'] for r in rows[:args.limit]]
                config = default_config(args.gemma_max_tokens, args.gemma_context, binary_prompt=args.binary_prompt,
                                        recheck_appropriate=args.recheck_appropriate, empty_alt_prompt=args.empty_alt_prompt,
                                        review_policy=args.review_policy if args.binary_prompt == 'evidence_split' else None,
                                        role_observation=args.binary_prompt == 'evidence_split' and args.review_policy == 'appropriate_v1',
                                        label_reflection=args.label_reflection,
                                        compare_review_role=(args.compare_review_role if args.compare_review_role is not None else
                                            args.binary_prompt == 'evidence_split' and args.review_policy == 'appropriate_v1'
                                            and args.recheck_appropriate and args.empty_alt_prompt))
                run_dir = create_run(args.dataset, ids, [], args.background, args.device,
                                     gemma_config=config, pipeline_mode='gemma_only')
            elif load_manifest(run_dir).get('pipeline_mode') != 'gemma_only':
                raise ValueError('과거 OCR 실험은 열람만 지원합니다. Gemma 단독 새 실행을 만드세요.')
            (run_dir / 'STOP').unlink(missing_ok=True)
            print(f'RUN_DIR={run_dir.resolve()}', flush=True)
            return execute_pipeline(run_dir, lock.fileno())
    except Exception as exc:
        print(f'Cannot start Gemma: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
