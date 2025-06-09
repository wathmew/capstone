import json
import argparse

def load_entries(path):
    entries = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            entries[entry['doc_id']] = entry
    return entries


def compare_labels(e1, e2):
    diffs = []
    # Compare entity mention texts
    ems1 = e1.get('entity_mentions', [])
    ems2 = e2.get('entity_mentions', [])
    if len(ems1) != len(ems2):
        diffs.append(('entity_count_mismatch', len(ems1), len(ems2)))
    # Compare corresponding texts (assumes same ordering)
    for i, (m1, m2) in enumerate(zip(ems1, ems2)):
        text1 = m1.get('text')
        text2 = m2.get('text')
        if text1 != text2:
            diffs.append(('entity_text', text1, text2))

    # Compare event trigger texts
    evs1 = e1.get('event_mentions', [])
    evs2 = e2.get('event_mentions', [])
    if len(evs1) != len(evs2):
        diffs.append(('event_count_mismatch', len(evs1), len(evs2)))
    for i, (ev1, ev2) in enumerate(zip(evs1, evs2)):
        trig1 = ev1.get('trigger', {}).get('text')
        trig2 = ev2.get('trigger', {}).get('text')
        if trig1 != trig2:
            diffs.append(('event_trigger_text', trig1, trig2))

    return diffs


def main():
    parser = argparse.ArgumentParser(description='Compare only Chinese text of entity and event mentions between two JSONL files')
    parser.add_argument('file1', help='First JSONL file path')
    parser.add_argument('file2', help='Second JSONL file path')
    parser.add_argument('-o', '--output', help='Redirect output to this file', default=None)
    args = parser.parse_args()

    out_f = open(args.output, 'w', encoding='utf-8') if args.output else None
    def write(line):
        if out_f:
            out_f.write(line + '\n')
        else:
            print(line)

    entries1 = load_entries(args.file1)
    entries2 = load_entries(args.file2)
    common_ids = set(entries1.keys()) & set(entries2.keys())
    total_entries = len(common_ids)
    write(f"Total documents compared: {total_entries}")

    total_diffs = 0
    for doc_id in sorted(common_ids):
        diffs = compare_labels(entries1[doc_id], entries2[doc_id])
        if diffs:
            write(f"Differences in document: {doc_id}")
            for dtype, val1, val2 in diffs:
                write(f"  {dtype}: {args.file1}: '{val1}', {args.file2}: '{val2}'")
            total_diffs += len(diffs)

    write(f"Total differences: {total_diffs}")

    if out_f:
        out_f.close()

if __name__ == '__main__':
    main()
