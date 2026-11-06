"""Gemma F1, fallback comparison, and historical experiment reports. No inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from common import load_manifest, load_result, now, read_json, write_json
from gemma import (DECORATIVE_STRUCTURED_GATE, DETERMINISTIC_BINARY_GATE_SOURCES,
                   EMPTY_ALT_LINK_GATE, ERROR_PRIORITY,
                   EMPTY_ALT_LINK_VISUAL_REVIEW, FROZEN_FEEDBACK_GATE,
                   LABELS, LINK_STRUCTURE_GATE, METADATA_FIELDS,
                   REFLECTION_ERRORS_ONLY, REFLECTION_FROZEN, REFLECTION_TRAIN,
                   SAME_LINK_ALT_REDUNDANCY_GATE, SCHEMAS, TYPE_CANDIDATE_RECALL, _response_contract,
                   audit_request, binary_prompt_for_input, candidate_priority_decision,
                   decorative_observation_decision,
                   enforce_final_candidate_pool, expand_type_candidates, fingerprint, final_candidate_labels,
                   frozen_coverage_feedback, FROZEN_EVIDENCE_PIPELINE,
                   link_structure_gate, final_reflection_document_stage,
                   needs_appropriate_review, parse_response,
                   reflection_document_stage,
                   request_payload, type_reflection_document_stage)
from role_review_report import build_role_review_report, role_review_csv, show_role_review_report

VARIANTS = {'with_ocr': 'gemma', 'without_ocr': 'gemma_no_ocr'}
VARIANT_NAMES = {'with_ocr': '정제 OCR 사용', 'without_ocr': 'OCR 미사용'}
FALLBACK_NAMES = {f'{variant}_{mode}': f'{name} · 폴백 {suffix}'
                  for variant, name in VARIANT_NAMES.items()
                  for mode, suffix in [('no_fallback', '없음'), ('fallback', '있음')]}
BINARY_LABELS = ('적절', '부적절')


def variants_for(manifest):
    """Keep saved OCR ablations intact while new runs have one Gemma result."""
    return {'without_ocr': 'gemma_no_ocr'} if manifest.get('pipeline_mode') == 'gemma_only' else VARIANTS


def variant_names_for(manifest):
    if (manifest.get('gemma') or {}).get('compare_review_role'):
        return {'without_ocr': 'Gemma'}
    return {'without_ocr': 'Gemma'} if manifest.get('pipeline_mode') == 'gemma_only' else VARIANT_NAMES


def fallback_names_for(variant_names):
    return {f'{variant}_{mode}': f'{name} · 폴백 {suffix}'
            for variant, name in variant_names.items()
            for mode, suffix in [('no_fallback', '없음'), ('fallback', '있음')]}


def score_labels(truth, predictions, labels=LABELS):
    if len(truth) != len(predictions) or not set(truth + predictions) <= set(labels):
        raise ValueError('같은 개수의 유효한 정답/예측 라벨이 필요합니다.')
    if not truth:
        return None
    per_class = {}
    matrix = [[sum(a == actual and b == predicted for a, b in zip(truth, predictions))
               for predicted in labels] for actual in labels]
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        support, predicted = sum(matrix[i]), sum(row[i] for row in matrix)
        per_class[label] = {'support': support, 'correct_count': tp,
                            'precision': tp / predicted if predicted else 0.0,
                            'recall': tp / support if support else 0.0,
                            'f1': 2 * tp / (support + predicted) if support + predicted else 0.0}
    positive = set(labels) - {'적절'}
    binary_tp = sum(actual in positive and predicted in positive for actual, predicted in zip(truth, predictions))
    binary_actual = sum(actual in positive for actual in truth)
    binary_predicted = sum(predicted in positive for predicted in predictions)
    binary_f1 = 2 * binary_tp / (binary_actual + binary_predicted) if binary_actual + binary_predicted else 0.0
    macro_f1 = sum(row['f1'] for row in per_class.values()) / len(labels)
    return {'count': len(truth), 'macro_f1': macro_f1,
            'binary_f1': binary_f1, 'combined_score': 0.5 * (macro_f1 + binary_f1),
            'accuracy': sum(a == b for a, b in zip(truth, predictions)) / len(truth),
            'per_class': per_class, 'confusion_matrix': matrix}


def _validate_label_map(labels, name='labels'):
    if not isinstance(labels, dict):
        raise ValueError(f'{name}는 record_id: label 사전이어야 합니다.')
    for record_id, label in labels.items():
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f'{name}에 빈 record_id가 있습니다.')
        if label not in LABELS:
            raise ValueError(f'{name}의 {record_id!r} 라벨이 유효하지 않습니다.')


def read_label_csv(data):
    """Read UTF-8 CSV labels; metadata columns are allowed but IDs must be unique."""
    if not isinstance(data, bytes):
        raise ValueError('CSV 바이트 데이터가 필요합니다.')
    try:
        text = data.decode('utf-8-sig')
        reader = csv.DictReader(io.StringIO(text, newline=''), strict=True)
        headers = reader.fieldnames
        if not headers or len(headers) != len(set(headers)) or not {'record_id', 'label'} <= set(headers):
            raise ValueError('CSV에는 중복 없이 record_id,label 헤더가 필요합니다.')
        labels = {}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'CSV {reader.line_num}행의 열 수가 헤더와 다릅니다.')
            record_id, label = row.get('record_id'), row.get('label')
            if not record_id or not record_id.strip():
                raise ValueError('CSV에 빈 record_id가 있습니다.')
            if record_id in labels:
                raise ValueError(f'CSV에 중복 record_id가 있습니다: {record_id}')
            if label not in LABELS:
                raise ValueError(f'CSV의 {record_id!r} 라벨이 비었거나 유효하지 않습니다.')
            labels[record_id] = label
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f'CSV 형식이 올바르지 않습니다: {exc}') from exc
    if not labels:
        raise ValueError('CSV에 라벨 행이 없습니다.')
    return labels


def label_csv(labels):
    """Export validated labels in a spreadsheet-friendly UTF-8 BOM CSV."""
    _validate_label_map(labels)
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=['record_id', 'label'])
    writer.writeheader()
    writer.writerows({'record_id': record_id, 'label': label} for record_id, label in labels.items())
    return out.getvalue().encode('utf-8-sig')


def compare_label_csvs(truth, predictions):
    """Score CSV labels by ID; matching IDs alone do not prove identical model or prompt conditions."""
    _validate_label_map(truth, 'truth')
    if not isinstance(predictions, dict) or not predictions or not set(predictions) <= set(VARIANTS):
        raise ValueError('예측 방식은 with_ocr 또는 without_ocr만 사용할 수 있습니다.')
    for variant, labels in predictions.items():
        _validate_label_map(labels, variant)
        unknown = set(labels) - set(truth)
        if unknown:
            raise ValueError(f'{variant} CSV에 정답 CSV에 없는 record_id가 있습니다: {sorted(unknown)[0]}')
    if not any(predictions.values()):
        raise ValueError('최소 한 CSV에는 예측 라벨이 있어야 합니다.')

    single_run = set(predictions) == {'without_ocr'}
    variants = {'without_ocr': 'gemma_no_ocr'} if single_run else VARIANTS
    variant_names = {'without_ocr': 'Gemma'} if single_run else VARIANT_NAMES
    paired_ids = [record_id for record_id in truth if all(record_id in labels for labels in predictions.values())]
    coverage, available = {}, {}
    for variant, labels in predictions.items():
        ids = [record_id for record_id in truth if record_id in labels]
        coverage[variant] = {'ok': len(ids), 'error': 0, 'pending': len(truth) - len(ids), 'labeled_ok': len(ids)}
        available[variant] = score_labels([truth[record_id] for record_id in ids], [labels[record_id] for record_id in ids])
    paired = {variant: score_labels([truth[record_id] for record_id in paired_ids],
                                    [labels[record_id] for record_id in paired_ids])
              for variant, labels in predictions.items()}
    both = {'with_ocr', 'without_ocr'} <= set(predictions)
    delta = (paired['with_ocr']['macro_f1'] - paired['without_ocr']['macro_f1']
             if both and paired_ids else None)
    rows = []
    for record_id, label in truth.items():
        row = {'record_id': record_id, 'truth': label, 'source': 'csv'}
        for variant in variants:
            value = predictions.get(variant, {}).get(record_id)
            row[variant] = value
            row[variant + '_status'] = 'ok' if value is not None else 'pending'
        rows.append(row)
    return {'schema_version': 1, 'source': 'csv', 'generated_at': now(), 'labels': list(LABELS), 'zero_division': 0,
            'selected_count': len(truth), 'labeled_count': len(truth), 'unlabeled_count': 0,
            'input_mode': 'image_metadata' if single_run else None,
            'variants': variants, 'variant_names': variant_names,
            'coverage': coverage, 'available': available, 'paired_ids': paired_ids, 'paired': paired,
            'candidate_sets': None,
            'pairing_excluded': {}, 'delta_macro_f1': delta,
            'delta_definition': 'with_ocr - without_ocr' if not single_run else None,
            'full_comparison_complete': len(paired_ids) == len(truth), 'predictions': rows}


def request_signature(audit):
    """Compare audited requests after removing ONLY the OCR input and image path."""
    value = json.loads(json.dumps(audit))
    user = value['messages'][1]['content']
    user[0].pop('path', None)  # compare exact PNG sha256 instead of local location
    fields = json.loads(user[1]['text'])
    fields.pop('cleaned_ocr_text', None)
    user[1]['text'] = fields
    return fingerprint(value)


def _valid_result(result):
    return result and result.get('status') == 'ok' and result.get('label') in LABELS


def _valid_binary_result(result):
    return (result and result.get('status') == 'ok'
            and (result.get('binary_decision') or {}).get('verdict') in BINARY_LABELS
            and result.get('verdict') in BINARY_LABELS)


def _valid_candidate_result(result):
    if (not _valid_binary_result(result) or not result.get('candidate_only')
            or result.get('label') is not None or result.get('final_decision') is not None):
        return False
    if result['verdict'] == '적절':
        return not result.get('matched_types') and result.get('type_classification') is None
    types = result.get('type_classification')
    return (isinstance(types, dict) and isinstance(types.get('matched_types'), list)
            and bool(types['matched_types']) and result.get('matched_types') == types['matched_types'])


def _pairing_error(audits):
    if any(audit is None for audit in audits):
        return '요청 기록 없음'
    if 'cleaned_ocr_text' in json.loads(audits[1]['messages'][1]['content'][1]['text']):
        return 'OCR 미사용 요청에 OCR 필드가 포함됨'
    if request_signature(audits[0]) != request_signature(audits[1]):
        return '입력·프롬프트·생성 설정 불일치'
    return None


def _first_stage_report(run_dir, manifest, truth, results, variants, paired_ids=None):
    config = manifest.get('gemma') or {}
    if config.get('classification_mode') not in ('two_stage_verdict_types_v1',
                                                  'guideline_verdict_candidates_final_v1'):
        return None
    verdicts, coverage = {}, {}
    for variant, model in variants.items():
        saved = {}
        for rid, result in results[variant].items():
            binary = (result or {}).get('binary_decision') or {}
            verdict = binary.get('verdict')
            if verdict not in BINARY_LABELS:
                # The first response is published before the optional subtype call.
                # Reuse its parser so truncated/invalid outputs never become predictions.
                try:
                    raw = read_json(run_dir / 'raw' / model / (rid + '.json'))
                    verdict = parse_response(raw, 'classify', config)['verdict']
                except (KeyError, TypeError, IndexError, ValueError):
                    continue
            saved[rid] = verdict
        verdicts[variant] = saved
        coverage[variant] = {'completed': len(saved),
                             'evaluated': sum(truth[rid] in LABELS for rid in saved),
                             'unavailable': len(truth) - len(saved)}
    excluded = {}
    if paired_ids is None:
        if len(variants) == 1:
            paired_ids = [rid for rid, label in truth.items()
                          if label in LABELS and rid in next(iter(verdicts.values()))]
        else:
            paired_ids = []
            for rid, label in truth.items():
                if label not in LABELS or not all(rid in saved for saved in verdicts.values()):
                    continue
                try:
                    audits = [read_json(run_dir / 'raw' / model / (rid + '.request.json')) for model in variants.values()]
                    error = _pairing_error(audits)
                    if not error and config.get('role_stage'):
                        error = next((problem for audit in audits
                                      if (problem := _role_audit_error(run_dir, rid, audit, config))), None)
                except (KeyError, TypeError, IndexError, ValueError):
                    error = '요청 기록 형식 오류'
                if error:
                    excluded[rid] = error
                else:
                    paired_ids.append(rid)
    report = {'labels': list(BINARY_LABELS), 'coverage': coverage, 'paired_ids': paired_ids,
              'pairing_excluded': excluded, 'verdicts': verdicts}
    for scope in ('available', 'paired'):
        report[scope] = {}
        for variant, saved in verdicts.items():
            ids = paired_ids if scope == 'paired' else [rid for rid in saved if truth[rid] in LABELS]
            metrics = score_labels(['적절' if truth[rid] == '적절' else '부적절' for rid in ids],
                                   [saved[rid] for rid in ids], labels=BINARY_LABELS)
            if metrics:
                metrics.pop('combined_score')  # No seven-class competition score at this stage.
            report[scope][variant] = metrics
    return report


def _role_audit_error(run_dir, rid, binary_audit, config):
    path = run_dir / 'raw' / 'gemma_role' / (rid + '.json')
    audit = read_json(path.with_suffix('.request.json'))
    raw = read_json(path)
    if audit is None or raw is None:
        return '역할 판정 요청·응답 기록 없음'
    role = parse_response(raw, 'role', config)
    fields = json.loads(binary_audit['messages'][1]['content'][1]['text'])
    if config.get('evidence_stage') and fields.get('visual_evidence') != role:
        return '관찰 근거와 이진 입력 불일치'
    if binary_audit['messages'][0]['content'] != binary_prompt_for_input(config, {**fields, '_image_role': role.get('role')}):
        return '역할별 이진 판정 기준 불일치'
    expected = json.loads(json.dumps(binary_audit))
    expected['messages'][0]['content'] = config['prompts']['role']
    fields = json.loads(expected['messages'][1]['content'][1]['text'])
    fields.pop('alt_text', None)
    fields.pop('cleaned_ocr_text', None)
    fields.pop('visual_evidence', None)
    expected['messages'][1]['content'][1]['text'] = json.dumps(fields, ensure_ascii=False)
    contract = _response_contract('role', config)
    expected['response_format'] = {'type': 'json_schema', 'json_schema': {
        'name': contract, 'strict': True, 'schema': SCHEMAS[contract]}}
    if audit != expected:
        return '역할 판정 입력·프롬프트·생성 설정 불일치'
    return None


def _review_audit_error(run_dir, model, rid, binary_audit, config):
    path = run_dir / 'raw' / model / (rid + '.recheck.json')
    audit, raw = read_json(path.with_suffix('.request.json')), read_json(path)
    if audit is None or raw is None:
        return '적절 재검토 요청·응답 기록 없음'
    expected = json.loads(json.dumps(binary_audit))
    expected['messages'][0]['content'] = config['prompts']['recheck']
    if config.get('guideline_mode'):
        fields = json.loads(expected['messages'][1]['content'][1]['text'])
        fields['initial_assessment'] = parse_response(
            read_json(run_dir / 'raw' / model / (rid + '.json')), 'classify', config)
        expected['messages'][1]['content'][1]['text'] = json.dumps(fields, ensure_ascii=False)
        expected['id_slot'] = 1
    elif config.get('evidence_stage'):
        fields = json.loads(expected['messages'][1]['content'][1]['text'])
        if config.get('review_policy') == 'appropriate_v1':
            fields.pop('visual_evidence', None)
        else:
            fields['initial_assessment'] = parse_response(read_json(run_dir / 'raw' / model / (rid + '.json')), 'classify', config)
        expected['messages'][1]['content'][1]['text'] = json.dumps(fields, ensure_ascii=False)
    contract = _response_contract('recheck', config)
    expected['response_format'] = {'type': 'json_schema', 'json_schema': {
        'name': contract, 'strict': True, 'schema': SCHEMAS[contract]}}
    if audit != expected:
        return '적절 재검토 입력·프롬프트·생성 설정 불일치'
    parse_response(raw, 'recheck', config)
    return None


def _appropriate_review_report(run_dir, config, truth, first, variants):
    if not config.get('recheck_appropriate') or first is None:
        return None
    after, excluded, reviewed = {}, {}, {}
    for variant, model in variants.items():
        after[variant], excluded[variant] = {}, {}
        reviewed[variant] = set()
        for rid, verdict in first['verdicts'][variant].items():
            if verdict == '적절' or config.get('review_policy') == 'evidence_uncertain_v1':
                try:
                    audit = read_json(run_dir / 'raw' / model / (rid + '.request.json'))
                    fields = json.loads(audit['messages'][1]['content'][1]['text'])
                    decision = (parse_response(read_json(run_dir / 'raw' / model / (rid + '.json')), 'classify', config)
                                if config.get('evidence_stage') else None)
                    if not needs_appropriate_review(config, fields, verdict, decision):
                        after[variant][rid] = verdict
                        continue
                    error = _review_audit_error(run_dir, model, rid, audit, config)
                    if error:
                        excluded[variant][rid] = error
                        continue
                    verdict = parse_response(read_json(run_dir / 'raw' / model / (rid + '.recheck.json')),
                                             'recheck', config)['verdict']
                    reviewed[variant].add(rid)
                except (KeyError, TypeError, IndexError, ValueError):
                    excluded[variant][rid] = '적절 재검토 기록 형식 오류'
                    continue
            after[variant][rid] = verdict
    paired = [rid for rid in first['paired_ids'] if all(rid in after[v] for v in variants)]
    def metrics(variant, ids):
        if not ids:
            return None
        actual = ['적절' if truth[rid] == '적절' else '부적절' for rid in ids]
        before = [first['verdicts'][variant][rid] for rid in ids]
        final = [after[variant][rid] for rid in ids]
        return {'count': len(ids), 'reviewed': len(reviewed[variant].intersection(ids)),
                'changed': sum(a != b for a, b in zip(before, final)),
                'corrected': sum(a != y and b == y for a, b, y in zip(before, final, actual)),
                'harmed': sum(a == y and b != y for a, b, y in zip(before, final, actual)),
                'before': score_labels(actual, before, labels=BINARY_LABELS),
                'after': score_labels(actual, final, labels=BINARY_LABELS)}
    return {'paired_ids': paired, 'excluded': excluded, 'verdicts': after,
            'available': {v: metrics(v, [rid for rid in after[v] if truth[rid] in LABELS]) for v in variants},
            'paired': {v: metrics(v, paired) for v in variants}}


def _label_reflection_report(config, truth, results, variants):
    if not config.get('label_reflection'):
        return None
    verdicts = {}
    for variant, saved in results.items():
        verdicts[variant] = {
            rid: result['reflection_decision']['verdict']
            for rid, result in saved.items()
            if _valid_binary_result(result)
            and (result.get('binary_decision') or {}).get('verdict') in BINARY_LABELS
            and (result.get('reflection_decision') or {}).get('verdict') in BINARY_LABELS
        }
    paired_ids = [rid for rid, label in truth.items()
                  if label in LABELS and all(rid in verdicts[variant] for variant in variants)]

    def metrics(variant, ids):
        if not ids:
            return None
        actual = ['적절' if truth[rid] == '적절' else '부적절' for rid in ids]
        before = [results[variant][rid]['binary_decision']['verdict'] for rid in ids]
        after = [verdicts[variant][rid] for rid in ids]
        return {
            'count': len(ids),
            'deterministic_gate_preserved': sum(
                results[variant][rid].get('reflection_skipped') == 'deterministic_binary_gate'
                for rid in ids),
            'decorative_gate_selected': sum(
                results[variant][rid].get('decorative_gate_selected') is True
                for rid in ids),
            'initial_correct_skipped': sum(
                results[variant][rid].get('reflection_skipped') == 'initial_binary_correct'
                for rid in ids),
            'frozen_feedback_passed': sum(
                results[variant][rid].get('reflection_skipped') == 'frozen_feedback_not_needed'
                for rid in ids),
            'feedback_applied': sum(
                isinstance(results[variant][rid].get('reflection_feedback'), dict)
                and results[variant][rid]['reflection_feedback'].get('action') != '통과'
                for rid in ids),
            'changed': sum(a != b for a, b in zip(before, after)),
            'corrected': sum(a != y and b == y for a, b, y in zip(before, after, actual)),
            'harmed': sum(a == y and b != y for a, b, y in zip(before, after, actual)),
            'before': score_labels(actual, before, labels=BINARY_LABELS),
            'after': score_labels(actual, after, labels=BINARY_LABELS),
        }

    reflection_mode = config.get('reflection_mode', REFLECTION_TRAIN)
    return {
        'reflection_mode': reflection_mode,
        'teacher_assisted_train_only': reflection_mode == REFLECTION_TRAIN,
        'paired_ids': paired_ids,
        'available': {variant: metrics(
            variant, [rid for rid in verdicts[variant]
                      if truth[rid] in LABELS])
            for variant in variants},
        'paired': {variant: metrics(variant, paired_ids) for variant in variants},
    }


def _type_reflection_report(config, truth, results, variants):
    """Measure only whether the true improper type is present before/after Reflection."""
    if not config.get('type_reflection'):
        return None
    eligible = {}
    for variant, saved in results.items():
        eligible[variant] = []
        for rid, result in saved.items():
            valid = (_valid_candidate_result(result) if config.get('candidate_only')
                     else _valid_result(result))
            if truth.get(rid) not in ERROR_PRIORITY or not valid:
                continue
            before = (result or {}).get('initial_type_classification')
            after = (result or {}).get('type_classification')
            if (not isinstance(before, dict) or not isinstance(after, dict)
                    or not isinstance(before.get('matched_types'), list)
                    or not isinstance(after.get('matched_types'), list)):
                continue
            eligible[variant].append(rid)
    paired_ids = [rid for rid, label in truth.items()
                  if label in ERROR_PRIORITY and all(rid in eligible[v] for v in variants)]

    def metrics(variant, ids):
        if not ids:
            return None
        before_hits, after_hits, before_sizes, after_sizes = [], [], [], []
        for rid in ids:
            result = results[variant][rid]
            before = {item.get('label') for item in result['initial_type_classification']['matched_types']
                      if isinstance(item, dict)}
            after = {item.get('label') for item in result['type_classification']['matched_types']
                     if isinstance(item, dict)}
            before_hits.append(truth[rid] in before)
            after_hits.append(truth[rid] in after)
            before_sizes.append(len(before))
            after_sizes.append(len(after))
        count = len(ids)
        return {'count': count,
                'before_hit_count': sum(before_hits),
                'before_hit_rate': sum(before_hits) / count,
                'after_hit_count': sum(after_hits),
                'after_hit_rate': sum(after_hits) / count,
                'after_miss_count': count - sum(after_hits),
                'before_mean_size': sum(before_sizes) / count,
                'after_mean_size': sum(after_sizes) / count,
                'corrected': sum(not before and after for before, after in zip(before_hits, after_hits)),
                'harmed': sum(before and not after for before, after in zip(before_hits, after_hits))}

    return {
        'reflection_mode': config.get('reflection_mode', REFLECTION_TRAIN),
        'paired_ids': paired_ids,
        'available': {variant: metrics(variant, ids) for variant, ids in eligible.items()},
        'paired': {variant: metrics(variant, paired_ids) for variant in variants},
    }


def _final_reflection_report(config, truth, results, variants, paired_ids):
    """Score fixed-pool final selection separately from full seven-class output."""
    if not config.get('final_reflection'):
        return None

    def metrics(variant, ids):
        completed = []
        reachable = []
        truth_missing = 0
        selector_bypassed = 0
        for rid in ids:
            result = results[variant].get(rid)
            actual = truth.get(rid)
            if actual not in LABELS or not _valid_result(result):
                continue
            if result.get('decorative_gate_selected'):
                after = result.get('label')
                if after in LABELS:
                    completed.append((actual, after, after))
                    selector_bypassed += 1
                continue
            before = ('적절' if result.get('verdict') == '적절' else
                      (result.get('initial_final_decision') or {}).get('label'))
            after = result.get('label')
            if before not in LABELS or after not in LABELS:
                continue
            completed.append((actual, before, after))
            if actual in ERROR_PRIORITY:
                candidates = set(candidate_labels(result) or [])
                if actual in candidates:
                    reachable.append((actual, before, after))
                else:
                    truth_missing += 1
        if not completed:
            return None
        actual = [row[0] for row in reachable]
        before = [row[1] for row in reachable]
        after = [row[2] for row in reachable]
        count = len(reachable)
        initial_correct = sum(a == b for a, b in zip(actual, before))
        final_correct = sum(a == b for a, b in zip(actual, after))
        full_actual = [row[0] for row in completed]
        full_before = [row[1] for row in completed]
        full_after = [row[2] for row in completed]
        return {
            'reachable_count': count,
            'truth_missing_count': truth_missing,
            'selector_bypassed_count': selector_bypassed,
            'initial_correct_count': initial_correct,
            'final_correct_count': final_correct,
            'initial_accuracy': initial_correct / count if count else None,
            'final_accuracy': final_correct / count if count else None,
            'corrected': sum(a != b and a == c for a, b, c in zip(actual, before, after)),
            'harmed': sum(a == b and a != c for a, b, c in zip(actual, before, after)),
            'full_before': score_labels(full_actual, full_before),
            'full_after': score_labels(full_actual, full_after),
        }

    available_ids = {variant: list(results[variant]) for variant in variants}
    return {
        'reflection_mode': config.get('reflection_mode', REFLECTION_TRAIN),
        'teacher_assisted_train_only': config.get('reflection_mode', REFLECTION_TRAIN) == REFLECTION_TRAIN,
        'paired_ids': list(paired_ids),
        'available': {variant: metrics(variant, ids)
                      for variant, ids in available_ids.items()},
        'paired': {variant: metrics(variant, paired_ids) for variant in variants},
    }


def _fallback_comparison(truth, results, first, paired_ids, variants, fallback_names):
    """No-review output follows the original gate; no extra inference or truth-based selection."""
    predictions = []
    path_results = {key: {} for key in fallback_names}
    for rid, label in truth.items():
        row = {'record_id': rid, 'truth': label}
        for variant in variants:
            result = results[variant][rid]
            final = result['label'] if _valid_result(result) else None
            initial = first['verdicts'][variant].get(rid)
            row[variant + '_no_fallback'] = ('적절' if initial == '적절' else
                (result.get('initial_label') or (final if final != '적절' else None))
                if initial == '부적절' and result else None)
            row[variant + '_fallback'] = final
            # Explicit first-stage appropriate stops before subtype classification.
            # A reviewed bad->good result needs the separately saved initial types.
            before = None
            if initial == '적절':
                before = {'status': 'ok', 'verdict': '적절', 'label': '적절', 'matched_types': []}
            elif initial == '부적절' and result:
                types = (result.get('initial_type_classification')
                         if final == '적절' else result)
                if isinstance(types, dict):
                    before = {'status': 'ok', 'verdict': '부적절',
                              'label': row[variant + '_no_fallback'],
                              'matched_types': types.get('matched_types')}
            path_results[variant + '_no_fallback'][rid] = before
            path_results[variant + '_fallback'][rid] = result
        predictions.append(row)
    common_ids = set(paired_ids)
    paired = [row for row in predictions if row['record_id'] in common_ids
              and all(row[key] in LABELS for key in fallback_names)]
    def scores(rows, key):
        chosen = [row for row in rows if row['truth'] in LABELS and row[key] in LABELS]
        return score_labels([row['truth'] for row in chosen], [row[key] for row in chosen])
    candidates = _candidate_sets(truth, path_results, paired_ids, list(truth))
    candidate_ids = set(candidates['paired_ids'])
    for row in predictions:
        rid = row['record_id']
        row['fallback_candidate_paired'] = rid in candidate_ids
        for key, saved in path_results.items():
            labels = candidate_labels(saved[rid])
            hit = truth[rid] in labels if truth[rid] in LABELS and labels is not None else None
            row[key + '_candidates'] = labels
            row[key + '_candidate_hit'] = hit
            row[key + '_selection_miss'] = hit and row[key] != truth[rid] if hit is not None else None
            row[key + '_candidate_oracle_label'] = (truth[rid] if hit else row[key]) if hit is not None else None
    return {'paired_ids': [row['record_id'] for row in paired], 'predictions': predictions,
            'candidate_sets': candidates,
            'available': {key: scores(predictions, key) for key in fallback_names},
            'paired': {key: scores(paired, key) for key in fallback_names},
            'completed': {key: sum(row[key] in LABELS for row in predictions) for key in fallback_names}}


def _final_request_matches(actual, expected):
    """Accept the historical six-label schema while auditing every other field."""
    if actual == expected:
        return True
    legacy = json.loads(json.dumps(expected))
    try:
        legacy['response_format']['json_schema']['schema'] = SCHEMAS['final_error_type']
    except (KeyError, TypeError):
        return False
    return actual == legacy


def _audit_final_selection(final_path, expected_audit, types, result, config, prefix):
    """Reconstruct single-candidate, constrained-schema, and recovery selection."""
    labels = final_candidate_labels(types)
    priority = candidate_priority_decision(types, config['label_priority'])
    source = result.get('final_selection_source')
    if source == 'single_candidate_v1':
        if len(labels) != 1 or result.get('final_contract_recovery') is not None:
            return None, f'{prefix} 단일 후보 확정 기록 불일치'
        return priority, None
    audit = read_json(final_path.with_suffix('.request.json'))
    if audit is None or not final_path.is_file():
        return None, f'{prefix} 요청·응답 기록 없음'
    if not _final_request_matches(audit, expected_audit):
        return None, f'{prefix} 입력·프롬프트·생성 설정 불일치'
    attempted = parse_response(read_json(final_path), 'final', config)
    final, recovery = enforce_final_candidate_pool(
        attempted, types, config['label_priority'])
    expected_source = ('candidate_priority_contract_fallback_v1'
                       if recovery else 'candidate_schema_v1')
    # Old completed runs predate the source marker and used the broad schema.
    if source is not None and source != expected_source:
        return None, f'{prefix} 선택 방식 기록 불일치'
    if result.get('final_contract_recovery') != recovery:
        return None, f'{prefix} 후보 계약 복구 기록 불일치'
    return final, None


def _two_stage_audit_error(run_dir, model, rid, result, binary_audit, config):
    """Each executed second call must use the same inputs and saved type instructions."""
    decision = result.get('binary_decision') or {}
    verdict = decision.get('verdict')
    if verdict not in ('적절', '부적절'):
        return '1단계 판정 기록 불일치'
    fields = json.loads(binary_audit['messages'][1]['content'][1]['text'])
    if needs_appropriate_review(config, fields, verdict, result.get('binary_decision')):
        error = _review_audit_error(run_dir, model, rid, binary_audit, config)
        if error:
            return error
        review = parse_response(read_json(run_dir / 'raw' / model / (rid + '.recheck.json')), 'recheck', config)
        if review != result.get('review_decision'):
            return '적절 재검토 결과 불일치'
        decision = review
        verdict = review['verdict']
    elif result.get('review_decision'):
        return '재검토 대상이 아닌 문제에 재검토 결과가 있음'
    if config.get('evidence_stage') and result['binary_decision']['verdict'] == '부적절' and verdict == '적절':
        path = run_dir / 'raw' / model / (rid + '.initial_types.json')
        audit = read_json(path.with_suffix('.request.json'))
        expected = json.loads(json.dumps(binary_audit))
        expected['messages'][0]['content'] = config['prompts']['types']
        expected['response_format'] = {'type': 'json_schema', 'json_schema': {
            'name': 'error_types', 'strict': True, 'schema': SCHEMAS['error_types']}}
        if audit != expected or not path.is_file():
            return '최초 부적절 유형 요청·응답 기록 불일치'
        initial_types = parse_response(read_json(path), 'types', config)
        candidates = {v['label'] for v in initial_types['matched_types']}
        label = next(label for label in config['label_priority'] if label in candidates)
        if initial_types != result.get('initial_type_classification') or label != result.get('initial_label'):
            return '폴백 전 최초 라벨 기록 불일치'
    if verdict != result.get('verdict'):
        return '재검토 후 이진 판정 기록 불일치'
    if config.get('binary_only'):
        for suffix in ('types', 'final', 'initial_types'):
            path = run_dir / 'raw' / model / f'{rid}.{suffix}.json'
            if path.exists() or path.with_suffix('.request.json').exists():
                return '이진분류 전용 실행에 오류 유형 기록 존재'
        return (None if result.get('label') is None and not result.get('matched_types')
                and result.get('type_classification') is None
                and result.get('final_decision') is None
                else '이진분류 전용 결과 형식 불일치')
    if (verdict == '적절') != (result.get('label') == '적절'):
        return '1단계와 최종 판정 불일치'
    if config.get('role_stage'):
        error = _role_audit_error(run_dir, rid, binary_audit, config)
        if error:
            return error
        role = parse_response(read_json(run_dir / 'raw/gemma_role' / (rid + '.json')), 'role', config)
        if role != result.get('role_decision'):
            return '역할 판정 결과 불일치'
    if verdict == '적절':
        return None  # No subtype request is expected when the binary gate passes.
    path = run_dir / 'raw' / model / (rid + '.types.json')
    audit = read_json(path.with_suffix('.request.json'))
    if audit is None or not path.is_file():
        return '2단계 요청·응답 기록 없음'
    expected = json.loads(json.dumps(binary_audit))
    expected['messages'][0]['content'] = config['prompts']['types']
    if config.get('guideline_mode'):
        type_fields = json.loads(expected['messages'][1]['content'][1]['text'])
        type_fields['binary_assessment'] = decision
        expected['messages'][1]['content'][1]['text'] = json.dumps(type_fields, ensure_ascii=False)
    expected['response_format'] = {'type': 'json_schema', 'json_schema': {
        'name': 'error_types', 'strict': True, 'schema': SCHEMAS['error_types']}}
    if audit != expected:
        return '2단계 입력·프롬프트·생성 설정 불일치'
    if config.get('classification_mode') == 'guideline_verdict_candidates_final_v1':
        types = parse_response(read_json(path), 'types', config)
        if types != result.get('type_classification'):
            return '오류 유형 후보 결과 불일치'
        final_path = run_dir / 'raw' / model / (rid + '.final.json')
        final_expected = json.loads(json.dumps(binary_audit))
        final_expected['messages'][0]['content'] = config['prompts']['final']
        final_fields = json.loads(final_expected['messages'][1]['content'][1]['text'])
        final_fields['binary_assessment'] = decision
        final_fields['candidate_assessment'] = types
        final_expected['messages'][1]['content'][1]['text'] = json.dumps(final_fields, ensure_ascii=False)
        final_schema = json.loads(json.dumps(SCHEMAS['final_error_type']))
        final_schema['properties']['label']['enum'] = final_candidate_labels(types)
        final_expected['response_format'] = {'type': 'json_schema', 'json_schema': {
            'name': 'final_error_type', 'strict': True, 'schema': final_schema}}
        final, error = _audit_final_selection(
            final_path, final_expected, types, result, config, '최종 오류 유형')
        if error:
            return error
        candidates = {item['label'] for item in types['matched_types']}
        priority = next(label for label in config['label_priority'] if label in candidates)
        if (final != result.get('final_decision')
                or result.get('label') != final['label'] or result.get('priority_label') != priority):
            return '최종 오류 유형 결과 불일치'
    return None


def _single_audit_error(run_dir, model, rid, result, config):
    """Validate the saved request chain without inventing an OCR comparison arm."""
    try:
        if config.get('label_reflection'):
            return _label_reflection_audit_error(run_dir, model, rid, result, config)
        if result.get('binary_decision_source') in DETERMINISTIC_BINARY_GATE_SOURCES:
            return _gate_audit_error(run_dir, model, rid, result, config)
        audit = read_json(run_dir / 'raw' / model / (rid + '.request.json'))
        if audit is None:
            return '요청 기록 없음'
        if config.get('evidence_stage') or config.get('guideline_mode'):
            initial = parse_response(read_json(run_dir / 'raw' / model / (rid + '.json')), 'classify', config)
            if initial != result.get('binary_decision'):
                return '최초 이진 응답과 저장 판정 불일치'
        if config.get('classification_mode') in ('two_stage_verdict_types_v1',
                                                  'guideline_verdict_candidates_final_v1'):
            error = _two_stage_audit_error(run_dir, model, rid, result, audit, config)
            if error:
                return error
        if config.get('role_stage'):
            return _role_audit_error(run_dir, rid, audit, config)
    except (KeyError, TypeError, IndexError, ValueError):
        return '요청 기록 형식 오류'
    return None


def _label_reflection_audit_error(run_dir, model, rid, result, config):
    image_path = run_dir / result['input_path']
    rows = {row['record_id']: row for row in read_json(run_dir / 'records.json')}
    row = rows.get(rid) or {}
    metadata = {key: row.get(key) for key in METADATA_FIELDS}
    actual_label = row.get('label')
    initial = result.get('binary_decision')
    reflection_mode = config.get('reflection_mode', REFLECTION_TRAIN)
    if ((reflection_mode == REFLECTION_TRAIN and actual_label not in LABELS)
            or not isinstance(initial, dict)):
        return '라벨 Reflection 학습 입력 불일치'

    if result.get('binary_decision_source') in DETERMINISTIC_BINARY_GATE_SOURCES:
        gate_path = run_dir / 'raw' / model / f'{rid}.gate.json'
        gate_request, gate_decision = read_json(gate_path.with_suffix('.request.json')), read_json(gate_path)
        expected, expected_source = link_structure_gate(
            metadata, config.get('binary_gate'),
            legacy_readable_link=config.get('guideline_mode') == 'sections_1_3_v1')
        if (not isinstance(gate_request, dict)
                or gate_request.get('decision_source') != expected_source
                or gate_request.get('input') != metadata or gate_decision != expected or initial != expected):
            return 'Reflection 최초 구조 게이트 기록 불일치'
    else:
        initial_path = run_dir / 'raw' / model / f'{rid}.json'
        initial_stage = ('empty_alt_link_visual_review'
                         if result.get('binary_decision_source') == EMPTY_ALT_LINK_VISUAL_REVIEW
                         else 'classify_no_ocr')
        expected = audit_request(request_payload(initial_stage, image_path, metadata, config), image_path)
        if read_json(initial_path.with_suffix('.request.json')) != expected or not initial_path.is_file():
            return 'Reflection 최초 이진 요청·응답 기록 불일치'
        if parse_response(read_json(initial_path), 'classify_no_ocr', config) != initial:
            return 'Reflection 최초 이진 결과 불일치'

    decorative_selected = False
    if (config.get('decorative_gate') and str(metadata.get('alt_text') or '').strip()
            and result.get('binary_decision_source') not in DETERMINISTIC_BINARY_GATE_SOURCES):
        router_path = run_dir / 'raw' / model / f'{rid}.decorative_router.json'
        observer_path = run_dir / 'raw' / model / f'{rid}.decorative_observer.json'
        if (read_json(router_path.with_suffix('.request.json')) != audit_request(
                request_payload('decorative_router', image_path, metadata, config), image_path)
                or not router_path.is_file()):
            return '장식오용형 후보 라우터 입력 불일치'
        router = parse_response(read_json(router_path), 'decorative_router', config)
        if router != result.get('decorative_router_decision'):
            return '장식오용형 후보 라우터 결과 불일치'
        observation = None
        if router['verdict'] == '적절':
            if (read_json(observer_path.with_suffix('.request.json')) != audit_request(
                    request_payload('decorative_observer', image_path, metadata, config), image_path)
                    or not observer_path.is_file()):
                return '장식오용형 구조화 관찰 입력 불일치'
            observation = parse_response(read_json(observer_path), 'decorative_observer', config)
        elif observer_path.exists() or observer_path.with_suffix('.request.json').exists():
            return '장식오용형 라우터 제외 문제에 구조화 관찰 기록 존재'
        decorative_selected = bool(
            observation and decorative_observation_decision(observation, metadata))
        if (observation != result.get('decorative_observation')
                or result.get('decorative_gate_selected') is not decorative_selected):
            return '장식오용형 구조화 관찰 결과 불일치'

    if result.get('decorative_gate_selected'):
        if not decorative_selected:
            return '장식오용형 전용 게이트 적용 조건 불일치'
        candidates = result.get('type_classification') or {}
        final = result.get('final_decision') or {}
        if (result.get('verdict') != '부적절' or result.get('label') != '장식오용형'
                or result.get('reflection_decision_source') != DECORATIVE_STRUCTURED_GATE
                or result.get('reflection_skipped') != 'decorative_structured_gate'
                or [item.get('label') for item in candidates.get('matched_types', [])]
                != ['장식오용형'] or final.get('label') != '장식오용형'):
            return '장식오용형 전용 게이트 최종 결과 불일치'
        for suffix in ('reflection_visual_evidence', 'reflection_coverage_audit',
                       'reflection_feedback', 'reflection_fallback', 'reflection_document',
                       'types', 'final', 'type_reflection_feedback',
                       'final_reflection_feedback'):
            path = run_dir / 'raw' / model / f'{rid}.{suffix}.json'
            if path.exists() or path.with_suffix('.request.json').exists():
                return '장식오용형 전용 게이트 종료 뒤 추가 호출 기록 존재'
        return None

    skipped = result.get('reflection_skipped')
    gate_final = skipped == 'deterministic_binary_gate'
    initial_correct = skipped == 'initial_binary_correct'
    if gate_final or initial_correct:
        if gate_final:
            source = result.get('binary_decision_source')
            valid_gate = (source == EMPTY_ALT_LINK_GATE
                          and config.get('reflection_gate_policy') == 'deterministic_final_v1')
            valid_gate = valid_gate or source == SAME_LINK_ALT_REDUNDANCY_GATE
            if not valid_gate or result.get('reflection_decision_source') != 'deterministic_gate_v1':
                return 'Reflection 결정적 구조 게이트 기록 불일치'
        actual_verdict = '적절' if actual_label == '적절' else '부적절'
        if initial_correct and (
                config.get('reflection_training_route') != REFLECTION_ERRORS_ONLY
                or reflection_mode != REFLECTION_TRAIN
                or initial.get('verdict') != actual_verdict
                or result.get('reflection_decision_source') != 'train_initial_binary_correct_v1'):
            return 'Reflection 최초 정답 생략 기록 불일치'
        if (result.get('reflection_decision') != initial
                or result.get('reflection_feedback') is not None):
            return 'Reflection 생략 판정 기록 불일치'
        for suffix in ('reflection_visual_evidence', 'reflection_coverage_audit',
                       'reflection_feedback', 'reflection_fallback', 'reflection_document'):
            path = run_dir / 'raw' / model / f'{rid}.{suffix}.json'
            if path.exists() or path.with_suffix('.request.json').exists():
                return 'Reflection 생략 대상에 Reflection 기록 존재'
        fallback = initial
    else:
        coverage = None
        synthetic_feedback = False
        if (reflection_mode == REFLECTION_FROZEN
                and config.get('frozen_evidence_pipeline') == FROZEN_EVIDENCE_PIPELINE):
            visual_path = run_dir / 'raw' / model / f'{rid}.reflection_visual_evidence.json'
            visual_audit = read_json(visual_path.with_suffix('.request.json'))
            if (visual_audit != audit_request(
                    request_payload('reflection_visual_evidence', image_path, metadata, config), image_path)
                    or not visual_path.is_file()):
                return 'Frozen 시각 증거 추출 입력 불일치'
            visual = parse_response(read_json(visual_path), 'reflection_visual_evidence', config)
            if visual != result.get('reflection_visual_evidence'):
                return 'Frozen 시각 증거 추출 결과 불일치'
            coverage_path = run_dir / 'raw' / model / f'{rid}.reflection_coverage_audit.json'
            coverage_data = {**metadata, '_initial_assessment': visual}
            if (read_json(coverage_path.with_suffix('.request.json')) != audit_request(
                    request_payload('reflection_coverage_audit', image_path, coverage_data, config), image_path)
                    or not coverage_path.is_file()):
                return 'Frozen 시각 증거 대조 입력 불일치'
            coverage = parse_response(read_json(coverage_path), 'reflection_coverage_audit', config)
            if coverage != result.get('reflection_coverage_audit'):
                return 'Frozen 시각 증거 대조 결과 불일치'
            synthetic_feedback = coverage['verdict'] == '부적절'

        feedback_path = run_dir / 'raw' / model / f'{rid}.reflection_feedback.json'
        if synthetic_feedback:
            current_document = (run_dir / 'reflection_guidelines.md').read_text(encoding='utf-8')
            feedback = frozen_coverage_feedback(coverage)
            if (feedback_path.exists() or feedback_path.with_suffix('.request.json').exists()
                    or result.get('reflection_feedback_source') != FROZEN_EVIDENCE_PIPELINE
                    or feedback != result.get('reflection_feedback')):
                return 'Frozen 시각 증거 피드백 기록 불일치'
        else:
            feedback_audit = read_json(feedback_path.with_suffix('.request.json'))
            if not isinstance(feedback_audit, dict) or not feedback_path.is_file():
                return 'Reflection 피드백 요청·응답 기록 없음'
            feedback_fields = json.loads(feedback_audit['messages'][1]['content'][-1]['text'])
            current_document = feedback_fields.get('current_guidelines')
            feedback_data = {**metadata, '_initial_assessment': initial,
                             '_reflection_document': current_document}
            if reflection_mode == REFLECTION_TRAIN:
                feedback_data['_actual_label'] = actual_label
            if feedback_audit != audit_request(
                    request_payload('reflection_feedback', image_path, feedback_data, config), image_path):
                return 'Reflection 피드백 입력 불일치'
            feedback = parse_response(read_json(feedback_path), 'reflection_feedback', config)
            if (feedback != result.get('reflection_feedback')
                    or (coverage is not None
                        and result.get('reflection_feedback_source') != REFLECTION_FROZEN)):
                return 'Reflection 피드백 결과 불일치'
        if (reflection_mode == REFLECTION_FROZEN
                and hashlib.sha256((current_document or '').encode()).hexdigest()
                != config.get('reflection_guideline_sha256')):
            return 'Frozen Reflection 요청의 가이드라인 스냅샷 불일치'

        fallback_path = run_dir / 'raw' / model / f'{rid}.reflection_fallback.json'
        frozen_pass = (reflection_mode == REFLECTION_FROZEN
                       and config.get('frozen_feedback_policy') == FROZEN_FEEDBACK_GATE
                       and feedback.get('action') == '통과')
        if frozen_pass:
            if (fallback_path.exists() or fallback_path.with_suffix('.request.json').exists()
                    or result.get('reflection_skipped') != 'frozen_feedback_not_needed'
                    or result.get('reflection_decision') != initial):
                return 'Frozen Reflection 통과 기록 불일치'
            fallback = initial
        else:
            fallback_feedback = ({'feedback': feedback['feedback']}
                                 if (reflection_mode == REFLECTION_FROZEN
                                     and config.get('frozen_feedback_policy') == FROZEN_FEEDBACK_GATE)
                                 else feedback)
            fallback_data = {**metadata, '_initial_assessment': initial,
                             '_reflection_feedback': fallback_feedback}
            if reflection_mode == REFLECTION_FROZEN:
                fallback_data['_reflection_document'] = current_document
            if (read_json(fallback_path.with_suffix('.request.json')) != audit_request(
                    request_payload('reflection_fallback', image_path, fallback_data, config), image_path)
                    or not fallback_path.is_file()):
                return 'Reflection 재판정 입력 불일치'
            fallback = parse_response(read_json(fallback_path), 'reflection_fallback', config)
            if fallback != result.get('reflection_decision'):
                return 'Reflection 재판정 결과 불일치'

        document_path = run_dir / 'raw' / model / f'{rid}.reflection_document.json'
        if reflection_mode == REFLECTION_FROZEN:
            if (document_path.exists() or document_path.with_suffix('.request.json').exists()
                    or result.get('reflection_decision_source') != REFLECTION_FROZEN
                    or hashlib.sha256(current_document.encode()).hexdigest()
                    != result.get('reflection_document_sha256')):
                return 'Frozen Reflection 가이드라인 고정 기록 불일치'
        else:
            document_data = {'_reflection_document': current_document, '_reflection_feedback': feedback}
            document_stage = reflection_document_stage(config, current_document)
            document_status = result.get('reflection_document_status', 'ok')
            if document_status == 'error':
                if (not isinstance(result.get('reflection_document_error'), str)
                        or not result['reflection_document_error'].strip()
                        or hashlib.sha256(current_document.encode()).hexdigest()
                        != result.get('reflection_document_sha256')):
                    return 'Reflection 문서 수정 실패 기록 불일치'
                request_path = document_path.with_suffix('.request.json')
                if (request_path.exists() and read_json(request_path) != audit_request(
                        request_payload(document_stage, image_path, document_data, config), image_path)):
                    return 'Reflection 문서 수정 실패 입력 불일치'
            else:
                if (read_json(document_path.with_suffix('.request.json')) != audit_request(
                        request_payload(document_stage, image_path, document_data, config), image_path)
                        or not document_path.is_file()):
                    return 'Reflection 문서 수정 입력 불일치'
                document = parse_response(read_json(document_path), document_stage, config)['markdown']
                if hashlib.sha256(document.encode()).hexdigest() != result.get('reflection_document_sha256'):
                    return 'Reflection 문서 수정 결과 불일치'

    if config.get('binary_only'):
        for suffix in ('types', 'final', 'initial_types'):
            path = run_dir / 'raw' / model / f'{rid}.{suffix}.json'
            if path.exists() or path.with_suffix('.request.json').exists():
                return '이진분류 전용 Reflection 실행에 오류 유형 기록 존재'
        return (None if result.get('verdict') == fallback['verdict']
                and result.get('label') is None and not result.get('matched_types')
                and result.get('type_classification') is None
                and result.get('final_decision') is None
                else '이진분류 전용 Reflection 결과 불일치')
    if config.get('candidate_only') and fallback['verdict'] == '적절':
        for suffix in ('types', 'final', 'type_reflection_feedback',
                       'type_reflection_fallback', 'final_reflection_feedback',
                       'final_reflection_fallback'):
            path = run_dir / 'raw' / model / f'{rid}.{suffix}.json'
            if path.exists() or path.with_suffix('.request.json').exists():
                return '후보풀 전용 적절 판정에 후보·최종 선택 기록 존재'
        return (None if result.get('candidate_only') is True
                and result.get('label') is None and not result.get('matched_types')
                and result.get('type_classification') is None
                and result.get('final_decision') is None
                else '후보풀 전용 적절 결과 불일치')
    if fallback['verdict'] == '적절':
        return None if result.get('label') == '적절' and not result.get('matched_types') else 'Reflection 적절 최종 결과 불일치'
    types_path = run_dir / 'raw' / model / f'{rid}.types.json'
    type_data = {**metadata, '_binary_assessment': fallback}
    if (read_json(types_path.with_suffix('.request.json')) != audit_request(
            request_payload('types_no_ocr', image_path, type_data, config), image_path)
            or not types_path.is_file()):
        return 'Reflection 후 오류 유형 입력 불일치'
    types = parse_response(read_json(types_path), 'types_no_ocr', config)
    if config.get('type_reflection'):
        types, error = _type_reflection_audit_error(
            run_dir, model, rid, result, config, image_path, metadata,
            actual_label, fallback, types)
        if error:
            return error
    elif types != result.get('type_classification'):
        return 'Reflection 후 오류 유형 결과 불일치'
    if config.get('candidate_only'):
        for suffix in ('final', 'final_reflection_feedback', 'final_reflection_fallback'):
            path = run_dir / 'raw' / model / f'{rid}.{suffix}.json'
            if path.exists() or path.with_suffix('.request.json').exists():
                return '후보풀 전용 실행에 최종 선택 기록 존재'
        return (None if result.get('candidate_only') is True
                and result.get('label') is None
                and result.get('matched_types') == types.get('matched_types')
                and result.get('type_classification') == types
                and result.get('final_decision') is None
                else '후보풀 전용 후보 결과 불일치')
    final_path = run_dir / 'raw' / model / f'{rid}.final.json'
    final_data = {**type_data, '_candidate_assessment': types}
    final_expected = audit_request(
        request_payload('final_no_ocr', image_path, final_data, config), image_path)
    initial_final, error = _audit_final_selection(
        final_path, final_expected, types, result, config, 'Reflection 후 최종 유형')
    if error:
        return error
    if config.get('final_reflection'):
        final, error = _final_reflection_audit_error(
            run_dir, model, rid, result, config, image_path, metadata,
            actual_label, fallback, types, initial_final)
        if error:
            return error
    else:
        final = initial_final
    if final != result.get('final_decision') or result.get('label') != final['label']:
        return 'Reflection 후 최종 유형 결과 불일치'
    return None


def _type_reflection_audit_error(run_dir, model, rid, result, config, image_path,
                                 metadata, actual_label, binary, initial_types):
    if result.get('initial_type_classification') != initial_types:
        return initial_types, '후보 Reflection 최초 후보 기록 불일치'
    mode = config.get('reflection_mode', REFLECTION_TRAIN)
    labels = {item['label'] for item in initial_types['matched_types']}
    skipped = result.get('type_reflection_skipped')
    expected_skip = (('actual_label_appropriate' if actual_label == '적절' else
                      'initial_candidate_hit' if actual_label in labels else None)
                     if mode == REFLECTION_TRAIN else None)
    suffixes = ('type_reflection_feedback', 'type_reflection_fallback',
                'type_reflection_document')
    if expected_skip:
        if (skipped != expected_skip or result.get('type_reflection_decision') != initial_types
                or result.get('type_reflection_feedback') is not None):
            return initial_types, '후보 Reflection 생략 기록 불일치'
        for suffix in suffixes:
            path = run_dir / 'raw' / model / f'{rid}.{suffix}.json'
            if path.exists() or path.with_suffix('.request.json').exists():
                return initial_types, '후보 Reflection 생략 대상에 호출 기록 존재'
        return initial_types, None

    feedback_path = run_dir / 'raw' / model / f'{rid}.type_reflection_feedback.json'
    feedback_audit = read_json(feedback_path.with_suffix('.request.json'))
    if not isinstance(feedback_audit, dict) or not feedback_path.is_file():
        return initial_types, '후보 Reflection 피드백 요청·응답 기록 없음'
    fields = json.loads(feedback_audit['messages'][1]['content'][-1]['text'])
    document = fields.get('current_guidelines')
    feedback_data = {**metadata, '_binary_assessment': binary,
                     '_candidate_assessment': initial_types,
                     '_type_reflection_document': document}
    if mode == REFLECTION_TRAIN:
        feedback_data['_actual_label'] = actual_label
    elif hashlib.sha256((document or '').encode()).hexdigest() != config.get(
            'type_reflection_guideline_sha256'):
        return initial_types, 'Frozen 후보 Reflection MD 스냅샷 불일치'
    if feedback_audit != audit_request(
            request_payload('type_reflection_feedback', image_path, feedback_data, config), image_path):
        return initial_types, '후보 Reflection 피드백 입력 불일치'
    feedback = parse_response(read_json(feedback_path), 'type_reflection_feedback', config)
    if feedback != result.get('type_reflection_feedback'):
        return initial_types, '후보 Reflection 피드백 결과 불일치'

    fallback_path = run_dir / 'raw' / model / f'{rid}.type_reflection_fallback.json'
    passed = (mode == REFLECTION_FROZEN and feedback.get('action') == '통과')
    if passed:
        if (fallback_path.exists() or fallback_path.with_suffix('.request.json').exists()
                or skipped != 'frozen_feedback_not_needed'):
            return initial_types, 'Frozen 후보 Reflection 통과 기록 불일치'
        revised = initial_types
    else:
        fallback_feedback = ({'feedback': feedback['feedback']}
                             if mode == REFLECTION_FROZEN else feedback)
        fallback_data = {**metadata, '_binary_assessment': binary,
                         '_candidate_assessment': initial_types,
                         '_type_reflection_feedback': fallback_feedback}
        if (read_json(fallback_path.with_suffix('.request.json')) != audit_request(
                request_payload('type_reflection_fallback', image_path, fallback_data, config), image_path)
                or not fallback_path.is_file()):
            return initial_types, '후보 Reflection 재생성 입력 불일치'
        revised = parse_response(read_json(fallback_path), 'type_reflection_fallback', config)
        if config.get('type_candidate_policy') == TYPE_CANDIDATE_RECALL:
            revised = expand_type_candidates(initial_types, revised)

    document_path = run_dir / 'raw' / model / f'{rid}.type_reflection_document.json'
    if mode == REFLECTION_FROZEN:
        if (document_path.exists() or document_path.with_suffix('.request.json').exists()
                or hashlib.sha256(document.encode()).hexdigest()
                != result.get('type_reflection_document_sha256')):
            return revised, 'Frozen 후보 Reflection MD 고정 기록 불일치'
    else:
        document_data = {'_type_reflection_document': document,
                         '_type_reflection_feedback': feedback}
        document_stage = type_reflection_document_stage(config, document)
        status = result.get('type_reflection_document_status', 'ok')
        if status == 'error':
            if (not result.get('type_reflection_document_error')
                    or hashlib.sha256(document.encode()).hexdigest()
                    != result.get('type_reflection_document_sha256')):
                return revised, '후보 Reflection MD 수정 실패 기록 불일치'
        else:
            if (read_json(document_path.with_suffix('.request.json')) != audit_request(
                    request_payload(document_stage, image_path, document_data, config), image_path)
                    or not document_path.is_file()):
                return revised, '후보 Reflection MD 수정 입력 불일치'
            updated = parse_response(read_json(document_path), document_stage, config)['markdown']
            if hashlib.sha256(updated.encode()).hexdigest() != result.get(
                    'type_reflection_document_sha256'):
                return revised, '후보 Reflection MD 수정 결과 불일치'
    if (revised != result.get('type_reflection_decision')
            or revised != result.get('type_classification')):
        return revised, '후보 Reflection 재판정 결과 불일치'
    return revised, None


def _final_reflection_audit_error(run_dir, model, rid, result, config, image_path,
                                  metadata, actual_label, binary, candidates, initial_final):
    """Reconstruct the final-selection Reflection chain and fixed candidate pool."""
    if result.get('initial_final_decision') != initial_final:
        return initial_final, '최종 선택 Reflection 최초 선택 기록 불일치'
    mode = config.get('reflection_mode', REFLECTION_TRAIN)
    labels = {item['label'] for item in candidates['matched_types']}
    skipped = result.get('final_reflection_skipped')
    expected_skip = ('single_candidate'
                     if result.get('final_selection_source') == 'single_candidate_v1' else None)
    if expected_skip is None and mode == REFLECTION_TRAIN:
        expected_skip = ('actual_label_appropriate' if actual_label == '적절' else
                         'actual_label_not_in_candidates' if actual_label not in labels else
                         'initial_final_correct' if initial_final['label'] == actual_label else None)
    suffixes = ('final_reflection_feedback', 'final_reflection_fallback',
                'final_reflection_document')
    if expected_skip:
        if (skipped != expected_skip
                or result.get('final_reflection_decision') != initial_final
                or result.get('final_reflection_feedback') is not None):
            return initial_final, '최종 선택 Reflection 생략 기록 불일치'
        for suffix in suffixes:
            path = run_dir / 'raw' / model / f'{rid}.{suffix}.json'
            if path.exists() or path.with_suffix('.request.json').exists():
                return initial_final, '최종 선택 Reflection 생략 대상에 호출 기록 존재'
        return initial_final, None

    feedback_path = run_dir / 'raw' / model / f'{rid}.final_reflection_feedback.json'
    feedback_audit = read_json(feedback_path.with_suffix('.request.json'))
    if not isinstance(feedback_audit, dict) or not feedback_path.is_file():
        return initial_final, '최종 선택 Reflection 피드백 요청·응답 기록 없음'
    fields = json.loads(feedback_audit['messages'][1]['content'][-1]['text'])
    document = fields.get('current_guidelines')
    feedback_data = {
        **metadata, '_binary_assessment': binary,
        '_candidate_assessment': candidates,
        '_initial_final_assessment': initial_final,
        '_final_reflection_document': document,
    }
    if mode == REFLECTION_TRAIN:
        feedback_data['_actual_label'] = actual_label
    elif hashlib.sha256((document or '').encode()).hexdigest() != config.get(
            'final_reflection_guideline_sha256'):
        return initial_final, 'Frozen 최종 선택 Reflection MD 스냅샷 불일치'
    if feedback_audit != audit_request(
            request_payload('final_reflection_feedback', image_path, feedback_data, config), image_path):
        return initial_final, '최종 선택 Reflection 피드백 입력 불일치'
    feedback = parse_response(read_json(feedback_path), 'final_reflection_feedback', config)
    if feedback != result.get('final_reflection_feedback'):
        return initial_final, '최종 선택 Reflection 피드백 결과 불일치'

    fallback_path = run_dir / 'raw' / model / f'{rid}.final_reflection_fallback.json'
    passed = (mode == REFLECTION_FROZEN and feedback.get('action') == '통과')
    if passed:
        if (fallback_path.exists() or fallback_path.with_suffix('.request.json').exists()
                or skipped != 'frozen_feedback_not_needed'):
            return initial_final, 'Frozen 최종 선택 Reflection 통과 기록 불일치'
        revised = initial_final
    else:
        fallback_data = {
            **metadata, '_binary_assessment': binary,
            '_candidate_assessment': candidates,
            '_initial_final_assessment': initial_final,
            '_final_reflection_feedback': {'feedback': feedback['feedback']},
            '_final_reflection_document': document,
        }
        expected_fallback = audit_request(
            request_payload('final_reflection_fallback', image_path, fallback_data, config), image_path)
        if (not _final_request_matches(
                read_json(fallback_path.with_suffix('.request.json')), expected_fallback)
                or not fallback_path.is_file()):
            return initial_final, '최종 선택 Reflection 재선택 입력 불일치'
        attempted = parse_response(read_json(fallback_path), 'final_reflection_fallback', config)
        revised, recovery = enforce_final_candidate_pool(
            attempted, candidates, config['label_priority'])
        if result.get('final_reflection_contract_recovery') != recovery:
            return revised, '최종 선택 Reflection 후보 계약 복구 기록 불일치'

    document_path = run_dir / 'raw' / model / f'{rid}.final_reflection_document.json'
    if mode == REFLECTION_FROZEN:
        if (document_path.exists() or document_path.with_suffix('.request.json').exists()
                or result.get('final_reflection_decision_source') != REFLECTION_FROZEN
                or hashlib.sha256(document.encode()).hexdigest()
                != result.get('final_reflection_document_sha256')):
            return revised, 'Frozen 최종 선택 Reflection 가이드라인 고정 기록 불일치'
    else:
        document_data = {'_final_reflection_document': document,
                         '_final_reflection_feedback': feedback}
        document_stage = final_reflection_document_stage(config, document)
        status = result.get('final_reflection_document_status', 'ok')
        if status == 'error':
            if (not result.get('final_reflection_document_error')
                    or hashlib.sha256(document.encode()).hexdigest()
                    != result.get('final_reflection_document_sha256')):
                return revised, '최종 선택 Reflection MD 수정 실패 기록 불일치'
        else:
            if (read_json(document_path.with_suffix('.request.json')) != audit_request(
                    request_payload(document_stage, image_path, document_data, config), image_path)
                    or not document_path.is_file()):
                return revised, '최종 선택 Reflection MD 수정 입력 불일치'
            updated = parse_response(read_json(document_path), document_stage, config)['markdown']
            if hashlib.sha256(updated.encode()).hexdigest() != result.get(
                    'final_reflection_document_sha256'):
                return revised, '최종 선택 Reflection MD 수정 결과 불일치'
    if revised != result.get('final_reflection_decision'):
        return revised, '최종 선택 Reflection 재판정 결과 불일치'
    return revised, None


def _gate_audit_error(run_dir, model, rid, result, config):
    if config.get('binary_gate') not in (EMPTY_ALT_LINK_GATE, LINK_STRUCTURE_GATE):
        return '저장 설정에 없는 링크 구조 게이트 결과'
    gate_path = run_dir / 'raw' / model / (rid + '.gate.json')
    gate_request = read_json(gate_path.with_suffix('.request.json'))
    decision = read_json(gate_path)
    if (not isinstance(gate_request, dict)
            or gate_request.get('decision_source') not in DETERMINISTIC_BINARY_GATE_SOURCES):
        return '링크 구조 게이트 입력 기록 없음'
    fields = gate_request.get('input')
    if not isinstance(fields, dict) or set(fields) != set(METADATA_FIELDS):
        return '링크 구조 게이트 입력 기록 불일치'
    expected, expected_source = link_structure_gate(
        fields, config.get('binary_gate'),
        legacy_readable_link=config.get('guideline_mode') == 'sections_1_3_v1')
    if (expected is None or gate_request.get('decision_source') != expected_source
            or decision != expected or result.get('binary_decision') != expected
            or result.get('binary_decision_source') != expected_source):
        return '링크 구조 게이트 판정 기록 불일치'
    if result.get('verdict') != decision['verdict'] or result.get('review_decision'):
        return '링크 구조 게이트 최종 이진 판정 불일치'
    if config.get('binary_only'):
        for suffix in ('types', 'final'):
            path = run_dir / 'raw' / model / f'{rid}.{suffix}.json'
            if path.exists() or path.with_suffix('.request.json').exists():
                return '이진분류 전용 구조 게이트에 오류 유형 기록 존재'
        return (None if result.get('label') is None and not result.get('matched_types')
                and result.get('type_classification') is None
                and result.get('final_decision') is None
                else '이진분류 전용 구조 게이트 결과 불일치')
    if decision['verdict'] == '적절':
        if (result.get('label') != '적절' or result.get('matched_types')
                or result.get('type_classification') is not None or result.get('final_decision') is not None):
            return '빈 alt 링크 적절 판정 결과 불일치'
        return None

    image_path = run_dir / result['input_path']
    type_path = run_dir / 'raw' / model / (rid + '.types.json')
    type_data = {**fields, '_binary_assessment': decision}
    expected_type_audit = audit_request(
        request_payload('types_no_ocr', image_path, type_data, config), image_path)
    type_audit = read_json(type_path.with_suffix('.request.json'))
    if config.get('execution_mode') == 'record_parallel_v1' and isinstance(type_audit, dict):
        slot_id = type_audit.get('id_slot')
        if slot_id is not None and (type(slot_id) is not int
                                    or not 0 <= slot_id < int(config['parallel_slots'])):
            return '게이트 후 병렬 슬롯 기록 불일치'
        if slot_id is not None:
            expected_type_audit['id_slot'] = slot_id
    if type_audit != expected_type_audit or not type_path.is_file():
        return '게이트 후 오류 유형 요청·응답 기록 불일치'
    types = parse_response(read_json(type_path), 'types', config)
    if types != result.get('type_classification'):
        return '게이트 후 오류 유형 후보 결과 불일치'
    final_path = run_dir / 'raw' / model / (rid + '.final.json')
    final_data = {**type_data, '_candidate_assessment': types}
    expected_final_audit = audit_request(
        request_payload('final_no_ocr', image_path, final_data, config), image_path)
    if config.get('execution_mode') == 'record_parallel_v1' and slot_id is not None:
        expected_final_audit['id_slot'] = slot_id
    final, error = _audit_final_selection(
        final_path, expected_final_audit, types, result, config, '게이트 후 최종 오류 유형')
    if error:
        return error
    candidates = {item['label'] for item in types['matched_types']}
    priority = next(label for label in config['label_priority'] if label in candidates)
    if (final != result.get('final_decision')
            or result.get('label') != final['label'] or result.get('priority_label') != priority):
        return '게이트 후 최종 오류 유형 결과 불일치'
    return None


def candidate_labels(result):
    """Return an auditable candidate set, or None when the saved output cannot prove one."""
    if not (_valid_result(result) or _valid_candidate_result(result)) or 'matched_types' not in result:
        return None
    matches = result['matched_types']
    if not isinstance(matches, list):
        return None
    labels = []
    for item in matches:
        if (not isinstance(item, dict) or set(item) != {'label', 'reason'} or item.get('label') not in LABELS[1:]
                or not isinstance(item.get('reason'), str) or not item['reason'].strip()
                or item['label'] in labels):
            return None
        labels.append(item['label'])
    verdict = result.get('verdict')
    if 'verdict' in result:
        if verdict == '적절' and not labels and (result['label'] == '적절'
                                                  or result.get('candidate_only')):
            return ['적절']
        if verdict == '부적절' and labels and (result['label'] in labels
                                                or result.get('candidate_only')):
            return labels
        return None
    # v4 output has no verdict: only a nonempty, internally consistent set is recoverable.
    return labels if labels and result['label'] in labels else None


def _candidate_metrics(truth, results, ids):
    if not ids:
        return None
    per_class = {label: {'support': 0, 'final_correct_count': 0, 'hit_count': 0,
                         'selection_miss_count': 0, 'candidate_miss_count': 0}
                 for label in LABELS}
    final_correct_count = hit_count = selection_miss_count = candidate_miss_count = 0
    sizes, multilabel_count = [], 0
    actual, priority, oracle = [], [], []
    for rid in ids:
        label, result = truth[rid], results[rid]
        candidates = candidate_labels(result)
        final_correct, hit = result['label'] == label, label in candidates
        actual.append(label)
        priority.append(result['label'])
        # Evaluation only: grant a correct answer when the truth is recoverable
        # from the set; retain the actual wrong prediction on a candidate miss.
        oracle.append(label if hit else result['label'])
        row = per_class[label]
        row['support'] += 1
        row['final_correct_count'] += final_correct
        row['hit_count'] += hit
        row['selection_miss_count'] += hit and not final_correct
        row['candidate_miss_count'] += not hit
        final_correct_count += final_correct
        hit_count += hit
        selection_miss_count += hit and not final_correct
        candidate_miss_count += not hit
        sizes.append(len(candidates))
        multilabel_count += len(candidates) > 1
    for row in per_class.values():
        support = row['support']
        row['final_recall'] = row['final_correct_count'] / support if support else None
        row['hit_rate'] = row['hit_count'] / support if support else None
    count = len(ids)
    return {'count': count, 'final_correct_count': final_correct_count,
            'final_accuracy': final_correct_count / count, 'hit_count': hit_count,
            'hit_rate': hit_count / count, 'selection_miss_count': selection_miss_count,
            'candidate_miss_count': candidate_miss_count, 'mean_size': sum(sizes) / count,
            'multilabel_count': multilabel_count, 'per_class': per_class,
            'priority': score_labels(actual, priority), 'oracle': score_labels(actual, oracle)}


def _candidate_sets(truth, results, paired_ids, ids):
    eligible = {variant: [rid for rid in ids if truth[rid] in LABELS and _valid_result(saved[rid])
                          and candidate_labels(saved[rid]) is not None]
                for variant, saved in results.items()}
    paired = [rid for rid in paired_ids if all(rid in eligible[variant] for variant in results)]
    available = {variant: _candidate_metrics(truth, saved, eligible[variant])
                 for variant, saved in results.items()}
    return {'oracle_definition': 'truth_if_in_candidates_else_final; evaluation_only',
            'available': available,
            'paired': {variant: _candidate_metrics(truth, saved, paired) for variant, saved in results.items()},
            'paired_ids': paired,
            'coverage': {variant: {'eligible': len(eligible[variant]),
                                   'unavailable': sum(truth[rid] in LABELS and bool(_valid_result(saved[rid]))
                                                      for rid in ids) - len(eligible[variant])}
                         for variant, saved in results.items()}}


def _binary_stage_comparison(truth, results, paired_ids, ids):
    """Compare the last binary verdict with the binary direction of the final label."""
    def verdict(result):
        for value in (result.get('verdict'),
                      (result.get('reflection_decision') or {}).get('verdict'),
                      (result.get('binary_decision') or {}).get('verdict')):
            if value in BINARY_LABELS:
                return value
        return None

    eligible = {variant: [rid for rid in ids if truth[rid] in LABELS
                          and _valid_result(saved[rid]) and verdict(saved[rid])]
                for variant, saved in results.items()}
    common = [rid for rid in paired_ids if all(rid in eligible[variant] for variant in results)]

    def metrics(saved, selected):
        if not selected:
            return None
        actual = ['적절' if truth[rid] == '적절' else '부적절' for rid in selected]
        return {
            'pre_candidate': score_labels(actual, [verdict(saved[rid]) for rid in selected], BINARY_LABELS),
            'final_label': score_labels(
                actual, ['적절' if saved[rid]['label'] == '적절' else '부적절' for rid in selected], BINARY_LABELS),
        }

    return {
        'paired_ids': common,
        'available': {variant: metrics(saved, eligible[variant]) for variant, saved in results.items()},
        'paired': {variant: metrics(saved, common) for variant, saved in results.items()},
    }


def build_benchmark(run_dir, *, audit_requests=True, record_ids=None, evaluation_name=None):
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    config = manifest.get('gemma') or {}
    binary_only = bool(config.get('binary_only'))
    candidate_only = bool(config.get('candidate_only'))
    valid_result = (_valid_binary_result if binary_only else
                    _valid_candidate_result if candidate_only else _valid_result)
    variants = variants_for(manifest)
    variant_names = variant_names_for(manifest)
    fallback_names = fallback_names_for(variant_names)
    single_run = len(variants) == 1
    ids = list(manifest['record_ids'] if record_ids is None else record_ids)
    if len(ids) != len(set(ids)) or not set(ids) <= set(manifest['record_ids']):
        raise ValueError('평가할 record_id가 실행 대상과 일치하지 않습니다.')
    rows = {row['record_id']: row for row in read_json(run_dir / 'records.json')}
    truth = {rid: rows.get(rid, {}).get('label') for rid in ids}
    labeled = [rid for rid in ids if truth[rid] in LABELS]
    results = {variant: {rid: load_result(run_dir, model, rid) for rid in ids}
               for variant, model in variants.items()}
    coverage, available = {}, {}
    for variant, saved in results.items():
        good = [rid for rid in ids if valid_result(saved[rid])]
        evaluated = [rid for rid in good if truth[rid] in LABELS]
        pending = sum(saved[rid] is None for rid in ids)
        coverage[variant] = {'ok': len(good), 'error': len(ids) - len(good) - pending,
                             'pending': pending, 'labeled_ok': len(evaluated)}
        available[variant] = (None if binary_only or candidate_only else
                              score_labels([truth[rid] for rid in evaluated],
                                           [saved[rid]['label'] for rid in evaluated]))
    paired_ids, excluded = [], {}
    for rid in labeled:
        if not all(valid_result(results[v][rid]) for v in variants):
            continue
        if not audit_requests:
            paired_ids.append(rid)
            continue
        if single_run:
            variant, model = next(iter(variants.items()))
            error = _single_audit_error(run_dir, model, rid, results[variant][rid], config)
            if error:
                excluded[rid] = error
            else:
                paired_ids.append(rid)
            continue
        try:
            audits = [read_json(run_dir / 'raw' / model / (rid + '.request.json')) for model in variants.values()]
            error = _pairing_error(audits)
            if error:
                excluded[rid] = error
            else:
                errors = []
                if config.get('classification_mode') in ('two_stage_verdict_types_v1',
                                                          'guideline_verdict_candidates_final_v1'):
                    errors = [_two_stage_audit_error(run_dir, model, rid, results[variant][rid], audit, config)
                              for (variant, model), audit in zip(variants.items(), audits)]
                if any(errors):
                    excluded[rid] = next(error for error in errors if error)
                else:
                    paired_ids.append(rid)
        except (KeyError, TypeError, IndexError, ValueError):
            excluded[rid] = '요청 기록 형식 오류'
    paired = {variant: (None if binary_only or candidate_only else score_labels(
        [truth[rid] for rid in paired_ids], [saved[rid]['label'] for rid in paired_ids]))
              for variant, saved in results.items()}
    delta = (paired['with_ocr']['macro_f1'] - paired['without_ocr']['macro_f1']
             if not binary_only and not candidate_only and not single_run and paired_ids else None)
    candidate_sets = None if binary_only or candidate_only else _candidate_sets(
        truth, results, paired_ids, ids)
    binary_stage_comparison = None if binary_only or candidate_only else _binary_stage_comparison(
        truth, results, paired_ids, ids)
    predictions = [{'record_id': rid, 'truth': truth[rid],
                    **{variant: (saved[rid]['verdict'] if binary_only or candidate_only else saved[rid]['label'])
                       if valid_result(saved[rid]) else None
                       for variant, saved in results.items()},
                    **{variant + '_candidates': (None if binary_only else candidate_labels(saved[rid]))
                       for variant, saved in results.items()},
                    **{variant + '_candidate_hit': (truth[rid] in candidate_labels(saved[rid])
                                                    if not binary_only and truth[rid] in LABELS
                                                    and candidate_labels(saved[rid]) is not None else None)
                       for variant, saved in results.items()},
                    **{variant + '_initial_candidates': (
                        [item.get('label') for item in (saved[rid].get('initial_type_classification') or {}).get(
                            'matched_types', []) if isinstance(item, dict)]
                        if not binary_only and isinstance((saved[rid] or {}).get('initial_type_classification'), dict)
                        else None)
                       for variant, saved in results.items()},
                    **{variant + '_selection_miss': (truth[rid] in candidate_labels(saved[rid])
                                                     and saved[rid]['label'] != truth[rid]
                                                     if not binary_only and truth[rid] in LABELS
                                                     and candidate_labels(saved[rid]) is not None else None)
                       for variant, saved in results.items()},
                    **{variant + '_status': ('ok' if valid_result(saved[rid]) else 'error') if saved[rid] else 'pending'
                       for variant, saved in results.items()}} for rid in ids]
    for row in predictions:
        for variant in variants:
            hit = row[variant + '_candidate_hit']
            row[variant + '_candidate_oracle_label'] = (
                (row['truth'] if hit else row[variant]) if hit is not None else None)
            initial = row.get(variant + '_initial_candidates')
            row[variant + '_initial_candidate_hit'] = (
                row['truth'] in initial if row['truth'] in ERROR_PRIORITY and initial is not None else None)
    first_stage = _first_stage_report(run_dir, manifest, truth, results, variants)
    label_reflection = _label_reflection_report(manifest.get('gemma') or {}, truth, results, variants)
    type_reflection = _type_reflection_report(
        manifest.get('gemma') or {}, truth, results, variants)
    final_reflection = _final_reflection_report(
        manifest.get('gemma') or {}, truth, results, variants, paired_ids)
    fallback = (_fallback_comparison(truth, results, first_stage, paired_ids, variants, fallback_names)
                if (manifest.get('gemma') or {}).get('recheck_appropriate') and first_stage else None)
    if fallback:
        for row, extra in zip(predictions, fallback['predictions']):
            row.update(extra)
    return {'schema_version': 1, 'generated_at': now(), 'labels': list(LABELS), 'zero_division': 0,
            'input_mode': 'image_metadata' if single_run else None,
            'evaluation_name': evaluation_name,
            'binary_only': binary_only,
            'candidate_only': candidate_only,
            'stage_layout': (('frozen_reflection_v1'
                              if (manifest.get('gemma') or {}).get('reflection_mode') == REFLECTION_FROZEN
                              else 'label_reflection_v1')
                             if (manifest.get('gemma') or {}).get('label_reflection')
                             else 'guideline_all_stages_v1'
                             if (manifest.get('gemma') or {}).get('classification_mode') == 'guideline_verdict_candidates_final_v1'
                             else 'guideline_direct_v1' if (manifest.get('gemma') or {}).get('guideline_mode') else None),
            'variants': variants, 'variant_names': variant_names,
            'request_audit': 'complete' if audit_requests else 'skipped_for_live_dashboard',
            'selected_count': len(ids), 'labeled_count': len(labeled), 'unlabeled_count': len(ids) - len(labeled),
            'coverage': coverage, 'available': available, 'paired_ids': paired_ids, 'paired': paired,
            'candidate_sets': candidate_sets,
            'binary_stage_comparison': binary_stage_comparison,
            'first_stage': first_stage,
            'label_reflection': label_reflection,
            'type_reflection': type_reflection,
            'final_reflection': final_reflection,
            'appropriate_review': _appropriate_review_report(run_dir, manifest.get('gemma') or {}, truth, first_stage, variants),
            'fallback_comparison': fallback,
            'role_review_comparison': (None if binary_only else build_role_review_report(
                run_dir, manifest.get('gemma') or {}, rows,
                results.get('without_ocr', {}), paired_ids, excluded)),
            'pairing_excluded': excluded,
            'delta_macro_f1': delta, 'delta_definition': 'with_ocr - without_ocr' if not single_run else None,
            'full_comparison_complete': bool(labeled) and len(paired_ids) == len(labeled) == len(ids),
            'predictions': predictions}


def report_csv(report, predictions=False):
    out = io.StringIO(newline='')
    variants = report.get('variants') or VARIANTS
    variant_names = report.get('variant_names') or VARIANT_NAMES
    fallback_names = fallback_names_for(variant_names)
    if predictions:
        rows = []
        fields = ['record_id', 'truth'] + list(variants) + [variant + '_status' for variant in variants] + ['source']
        if report.get('fallback_comparison'):
            fields += list(fallback_names)
            if report['fallback_comparison'].get('candidate_sets'):
                fields += ['fallback_candidate_paired']
                fields += [key + suffix for key in fallback_names for suffix in (
                    '_candidates', '_candidate_hit', '_selection_miss', '_candidate_oracle_label')]
        for variant in variants:
            fields += [variant + '_candidates', variant + '_candidate_hit', variant + '_selection_miss',
                       variant + '_candidate_oracle_label', variant + '_initial_candidates',
                       variant + '_initial_candidate_hit']
        for row in report['predictions']:
            rows.append({key: (json.dumps(value, ensure_ascii=False) if key.endswith('_candidates') and value is not None else value)
                         for key, value in row.items()})
    else:
        rows = []
        fields = ['scope', 'variant', 'count', 'macro_f1', 'binary_f1', 'combined_score', 'accuracy'] + [f'f1_{label}' for label in LABELS] + [
            'candidate_count', 'candidate_final_accuracy', 'candidate_hit_rate', 'selection_miss_count',
            'candidate_miss_count', 'mean_candidate_size']
        candidate_f1_fields = {
            f'candidate_{criterion}_{metric}': (criterion, metric)
            for criterion in ('priority', 'oracle') for metric in ('macro_f1', 'binary_f1', 'combined_score')}
        fields += list(candidate_f1_fields)
        for scope in ('paired', 'available'):
            for variant, metrics in report[scope].items():
                candidate = (report.get('candidate_sets') or {}).get(scope, {}).get(variant)
                rows.append({'scope': scope, 'variant': variant, 'count': metrics['count'] if metrics else 0,
                             'macro_f1': metrics['macro_f1'] if metrics else None,
                             'binary_f1': metrics['binary_f1'] if metrics else None,
                             'combined_score': metrics['combined_score'] if metrics else None,
                             'accuracy': metrics['accuracy'] if metrics else None,
                             **{f'f1_{label}': metrics['per_class'][label]['f1'] if metrics else None for label in LABELS},
                             'candidate_count': candidate['count'] if candidate else 0,
                             'candidate_final_accuracy': candidate['final_accuracy'] if candidate else None,
                             'candidate_hit_rate': candidate['hit_rate'] if candidate else None,
                             'selection_miss_count': candidate['selection_miss_count'] if candidate else 0,
                             'candidate_miss_count': candidate['candidate_miss_count'] if candidate else 0,
                             'mean_candidate_size': candidate['mean_size'] if candidate else None,
                             **{name: (candidate or {}).get(criterion, {}).get(metric)
                                for name, (criterion, metric) in candidate_f1_fields.items()}})
        if report.get('fallback_comparison'):
            for scope in ('paired', 'available'):
                for key, metrics in report['fallback_comparison'][scope].items():
                    rows.append({'scope': 'fallback_' + scope, 'variant': key,
                                 'count': metrics['count'] if metrics else 0,
                                 **{field: metrics[field] if metrics else None
                                    for field in ('macro_f1', 'binary_f1', 'combined_score', 'accuracy')},
                                 **{f'f1_{label}': metrics['per_class'][label]['f1'] if metrics else None for label in LABELS}})
            candidate_comparison = report['fallback_comparison'].get('candidate_sets')
            if candidate_comparison:
                for key, candidate in candidate_comparison['paired'].items():
                    for criterion in ('priority', 'oracle'):
                        metrics = candidate[criterion] if candidate else None
                        rows.append({'scope': 'fallback_candidates_paired_' + criterion, 'variant': key,
                                     'count': metrics['count'] if metrics else 0,
                                     **{field: metrics[field] if metrics else None
                                        for field in ('macro_f1', 'binary_f1', 'combined_score', 'accuracy')},
                                     **{f'f1_{label}': metrics['per_class'][label]['f1'] if metrics else None
                                        for label in LABELS}})
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode('utf-8-sig')


def save_benchmark(run_dir, report=None):
    run_dir = Path(run_dir)
    report = report or build_benchmark(run_dir)
    write_json(run_dir / 'benchmark.json', report)
    exports = [('benchmark.csv', report_csv(report)), ('benchmark_predictions.csv', report_csv(report, True))]
    if report.get('role_review_comparison') is not None:
        exports.append(('role_review_predictions.csv', role_review_csv(report['role_review_comparison'])))
    for filename, data in exports:
        temp = run_dir / (filename + '.tmp')
        temp.write_bytes(data)
        temp.replace(run_dir / filename)
    return report


def show_fallback_comparison(report):
    import streamlit as st
    fallback = report.get('fallback_comparison')
    if not fallback:
        return
    if fallback.get('candidate_sets'):
        from candidate_view import show_fallback_candidate_scores
        show_fallback_candidate_scores(report)
        return
    names = report.get('variant_names') or VARIANT_NAMES
    fallback_names = fallback_names_for(names)
    single_run = len(names) == 1
    st.subheader('Gemma 폴백 비교' if single_run else 'OCR × 폴백 · 4조건 비교')
    st.caption(f"{'두 조건 모두 완료된' if single_run else '네 조건 모두 완료된 동일'} 문제 {len(fallback['paired_ids']):,} / 정답 공개 {report['labeled_count']:,}개 · "
               '폴백 없음은 저장된 최초 판정으로 계산합니다. 1차 추론을 중복 실행하지 않습니다.')
    scope = 'paired' if fallback['paired_ids'] else 'available'
    if scope == 'available':
        st.caption('아직 공통 완료 문제가 없어 조건별 가용 표본을 표시합니다. 조건 사이 표본이 다를 수 있습니다.')
    elif len(fallback['paired_ids']) < report['selected_count']:
        st.caption('실행 중이거나 실패·대기·비교 제외 항목이 있는 잠정 점수입니다.')
    st.dataframe([{'조건': name, '완료 문제': fallback['completed'][key],
                   '평가 문제': m['count'] if m else 0,
                   '이진 F1': m['binary_f1'] if m else None, '7범주 macro-F1': m['macro_f1'] if m else None,
                   '종합 점수': m['combined_score'] if m else None, '정확도': m['accuracy'] if m else None}
                  for key, name in fallback_names.items() for m in [fallback[scope][key]]], hide_index=True)
    with st.expander(f"{'2' if single_run else '4'}조건 · 라벨별 F1"):
        st.dataframe([{'라벨': label, **{name: fallback[scope][key]['per_class'][label]['f1']
                         if fallback[scope][key] else None for key, name in fallback_names.items()}}
                      for label in LABELS], hide_index=True)


def show_benchmark(run_dir, report=None):
    import streamlit as st
    report = report or build_benchmark(run_dir)
    download_key = f"benchmark:{Path(run_dir).name}:{report.get('evaluation_name') or 'all'}"
    if report.get('first_stage'):
        from stage_view import show_stage_scores
        show_stage_scores(report)
    if report.get('binary_only') or report.get('candidate_only'):
        st.download_button('벤치마크 JSON', json.dumps(report, ensure_ascii=False, indent=2),
                           file_name='benchmark.json', mime='application/json', key=f'{download_key}:json')
        st.download_button('벤치마크 CSV', report_csv(report), file_name='benchmark.csv', mime='text/csv',
                           key=f'{download_key}:csv')
        st.download_button('항목별 예측 CSV', report_csv(report, True),
                           file_name='benchmark_predictions.csv', mime='text/csv',
                           key=f'{download_key}:predictions')
        return
    details = st.expander('전체 라벨별 성능 · 처리 현황') if report.get('first_stage') else st.container()
    with details:
        names = report.get('variant_names') or VARIANT_NAMES
        direct = report.get('input_mode') == 'image_metadata'
        st.subheader('Gemma 성능' if direct else 'OCR 유무 벤치마크')
        st.caption(f"동일 표본 비교: {len(report['paired_ids']):,} / 정답 공개 {report['labeled_count']:,}개 · 전체 {report['selected_count']:,}개")
        if report['pairing_excluded']:
            st.warning(f"동일 입력·설정을 확인하지 못한 {len(report['pairing_excluded'])}개는 비교 점수에서 제외했습니다. JSON에 사유를 기록합니다.")
        st.dataframe([{'방식': names[v], '성공': c['ok'], '실패': c['error'], '대기': c['pending']}
                      for v, c in report['coverage'].items()], hide_index=True)
        if not report['labeled_count']:
            st.info('정답 라벨이 공개되지 않아 F1을 계산하지 않습니다.')
        elif report['paired_ids']:
            if not report['full_comparison_complete']:
                st.caption('잠정 점수: 판정이 완료된 표본의 성능입니다. 실패·대기 항목은 위에 별도 표시합니다.')
            st.dataframe([{'방식': names[v], '7범주 macro-F1': m['macro_f1'], '부적절 이진 F1': m['binary_f1'],
                           '결합 점수 (0.5:0.5)': m['combined_score'], '정확도': m['accuracy']}
                          for v, m in report['paired'].items()], hide_index=True)
            if not direct:
                st.metric('macro-F1 차이 (정제 OCR 사용 − 미사용)', f"{report['delta_macro_f1']:+.4f}")
            st.dataframe([{'라벨': label,
                           '실제 문제 수': next(iter(report['paired'].values()))['per_class'][label]['support'],
                           **{column: value for v, m in report['paired'].items()
                              for column, value in (
                                  (f'{names[v]} 맞힌 수', m['per_class'][label]['correct_count']),
                                  (f'{names[v]} F1', m['per_class'][label]['f1']))}}
                          for label in LABELS], hide_index=True)
            st.caption('맞힌 수는 실제 정답이 해당 라벨인 문제 중 예측까지 맞은 개수입니다. '
                       'F1은 정밀도와 재현율의 조화평균이며, 단순 정답률과는 다릅니다.')
        else:
            st.caption('판정 결과를 기다리는 중입니다.' if direct else '두 방식이 모두 완료된 표본을 기다리는 중입니다.')
        with st.expander('방식별 가용 표본 점수 (서로 다른 표본일 수 있음)', expanded=not report['paired_ids']):
            st.dataframe([{'방식': names[v], '표본 수': m['count'] if m else 0,
                           '7범주 macro-F1': m['macro_f1'] if m else None,
                           '부적절 이진 F1': m['binary_f1'] if m else None,
                           '결합 점수 (0.5:0.5)': m['combined_score'] if m else None}
                          for v, m in report['available'].items()], hide_index=True)
        st.caption('7개 라벨 고정 · 분모가 0인 클래스 F1은 0 · '
                   f"{report.get('evaluation_name') or 'train 개발 데이터'} 결과입니다.")
        from candidate_view import show_candidate_scores
        if not report.get('first_stage'):
            show_candidate_scores(report)
    st.download_button('벤치마크 JSON', json.dumps(report, ensure_ascii=False, indent=2),
                       file_name='benchmark.json', mime='application/json', key=f'{download_key}:json')
    st.download_button('벤치마크 CSV', report_csv(report), file_name='benchmark.csv', mime='text/csv',
                       key=f'{download_key}:csv')
    st.download_button('항목별 예측 CSV', report_csv(report, True), file_name='benchmark_predictions.csv',
                       mime='text/csv', key=f'{download_key}:predictions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    report = save_benchmark(parser.parse_args().run)
    print(json.dumps({key: report[key] for key in ('coverage', 'delta_macro_f1', 'full_comparison_complete')}, ensure_ascii=False, indent=2))
