import json
import re
import sys
import os
from nltk.tokenize.treebank import TreebankWordDetokenizer

def detokenize_with_regex(tokens):

    detokenizer = TreebankWordDetokenizer()
    sentence = detokenizer.detokenize(tokens)
    sentence = re.sub(r'\s+([.,!?;:])', r'\1', sentence)
    sentence = re.sub(r'\(\s+', '(', sentence)
    sentence = re.sub(r'\s+\)', ')', sentence)
    return sentence

def decode_unicode(text):
    return text.encode('utf-8').decode('unicode_escape')

def reformat_data(data):
    formatted = {
        "doc_id": data["doc_key"],
        "wnd_id": f"{data['doc_key']}_0",
        "tokens": [token for sentence in data["sentences"] for token in sentence],
        "entity_mentions": [],
        "event_mentions": []
    }
    
    formatted["sentence"] = detokenize_with_regex(formatted["tokens"])
    
    for idx, (start, end, types) in enumerate(data.get("ent_spans", [])):
        formatted["entity_mentions"].append({
            "id": f"{formatted['wnd_id']}-E{idx}",
            "start": start,
            "end": end + 1,
            "entity_type": types[0][0] if types else "UNK",
            "mention_type": "UNK",
            "text": " ".join(formatted["tokens"][start:end + 1])
        })
    
    for idx, (trigger_start, trigger_end, event_types) in enumerate(data.get("evt_triggers", [])):
        event_mention = {
            "event_type": event_types[0][0] if event_types else "UNK",
            "id": f"{formatted['wnd_id']}-EV{idx}",
            "trigger": {
                "start": trigger_start,
                "end": trigger_end + 1,
                "text": " ".join(formatted["tokens"][trigger_start:trigger_end + 1])
            },
            "arguments": []
        }
        
        for (evt_start, evt_end), (arg_start, arg_end), role in data.get("gold_evt_links", []):
            if evt_start == trigger_start and evt_end == trigger_end:
                entity_text = " ".join(formatted["tokens"][arg_start:arg_end + 1])
                event_mention["arguments"].append({
                    "entity_id": f"{formatted['wnd_id']}-E{idx}",
                    "text": entity_text,
                    "role": role
                })
        
        formatted["event_mentions"].append(event_mention)
    
    return formatted

data = {
    "language_id": "eng",
    "doc_key": "nw_RC000462ebb18ca0b29222d5e557fa31072af8337e3a0910dca8b5b62f",
    "sentences": [
        [
            "Transportation", "officials", "are", "urging", "carpool", "and", "teleworking", "as", "options", "to", "combat", "an", "expected", "flood", "of", "drivers", "on", "the", "road", "."
        ],
        ["(", "Paul", "Duggan", ")"],
        [
            "--", "A", "Baltimore", "prosecutor", "accused", "a", "police", "detective", "of", "“", "sabotaging", "”", "investigations", "related", "to", "the", "death", "of", "Freddie", "Gray", ",", "accusing", "him", "of", "fabricating", "notes", "to", "suggest", "that", "the", "state", "’s", "medical", "examiner", "believed", "the", "manner", "of", "death", "was", "an", "accident", "rather", "than", "a", "homicide", "."
        ],
        [
        "The",
        "heated",
        "exchange",
        "came",
        "in",
        "the",
        "chaotic",
        "sixth",
        "day",
        "of",
        "the",
        "trial",
        "of",
        "Baltimore",
        "Officer",
        "Caesar",
        "Goodson",
        "Jr.",
        ",",
        "who",
        "drove",
        "the",
        "police",
        "van",
        "in",
        "which",
        "Gray",
        "suffered",
        "a",
        "fatal",
        "spine",
        "injury",
        "in",
        "2015",
        "."
      ],
      [
        "(",
        "Derek",
        "Hawkins",
        "and",
        "Lynh",
        "Bui",
        ")"
      ]
    ],
    "evt_triggers": [[69, 69, [["life.die.deathcausedbyviolentevents", 1.0]]]],
    "rel_triggers": [],
    "ent_spans": [
        [42, 43, [["evt090arg02victim", 1.0]]],
        [85, 88, [["evt090arg01killer", 1.0]]],
        [26, 26, [["evt090arg04place", 1.0]]]
    ],
    "gold_evt_links": [
        [[69, 69], [85, 88], "evt090arg01killer"],
        [[69, 69], [42, 43], "evt090arg02victim"],
        [[69, 69], [26, 26], "evt090arg04place"]
    ]
}

#formatted_data = reformat_data(data)
#print(json.dumps(formatted_data, indent=4))

#with open("test_RAMS_to_CLaP.txt", "w", encoding="utf-8") as file:
#    json.dump(formatted_data, file, indent=4)

#print("Reformatted data in test_RAMS_to_CLaP.txt")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file.jsonl>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_converted.jsonl"
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if line.strip():
                data = json.loads(line)
                transformed = reformat_data(data)
                outfile.write(json.dumps(transformed) + "\n")
    
    print(f"Converted file saved as: {output_file}")