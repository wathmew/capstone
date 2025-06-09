import json
import argparse
import re
from typing import Dict

def load_jsonl(path: str) -> Dict[str, dict]:
    """Load JSONL and return a dict from doc_id to entry."""
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            data[obj['doc_id']] = obj
    return data

def strip_lang_suffix(doc_id: str) -> str:
    """Remove language suffix from doc_id (e.g., '_zh', '_es')."""
    return re.sub(r'_[a-z]{2}$', '', doc_id)

def pair_event_mentions(eng_data: Dict[str, dict], non_eng_data: Dict[str, dict]):
    """Yield aligned event mentions as structured matching tuples."""
    for non_eng_doc_id, non_eng_entry in non_eng_data.items():
        base_doc_id = strip_lang_suffix(non_eng_doc_id)
        eng_entry = eng_data.get(base_doc_id)
        if not eng_entry:
            continue

        eng_events = {ev['id']: ev for ev in eng_entry.get('event_mentions', [])}
        for non_ev in non_eng_entry.get('event_mentions', []):
            ev_id = non_ev['id']
            eng_ev = eng_events.get(ev_id)
            if not eng_ev:
                continue

            eng_trigger = eng_ev['trigger']['text']
            non_eng_trigger = non_ev['trigger']['text']
            ev_type = eng_ev['event_type']

            eng_args = {arg['role']: arg['text'] for arg in eng_ev.get('arguments', [])}
            non_eng_args = {arg['role']: arg['text'] for arg in non_ev.get('arguments', [])}

            # Only keep pairs where the role exists in both
            shared_roles = set(eng_args) & set(non_eng_args)
            arg_pairs = [(eng_args[r], non_eng_args[r]) for r in shared_roles]

            yield {
                'event_type': ev_type,
                'trigger_pair': (eng_trigger, non_eng_trigger),
                'argument_pairs': arg_pairs
            }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--english', required=True, help='Path to English JSONL file')
    parser.add_argument('-t', '--translated', required=True, help='Path to non-English JSONL file')
    parser.add_argument('-o', '--output', required=True, help='Path to output JSONL file')
    args = parser.parse_args()

    eng_data = load_jsonl(args.english)
    non_eng_data = load_jsonl(args.translated)

    with open(args.output, 'w', encoding='utf-8') as f:
        for row in pair_event_mentions(eng_data, non_eng_data):
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()
