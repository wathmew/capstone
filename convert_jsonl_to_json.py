import json

def convert_jsonl_to_json(input_file, output_file):
    # Read JSONL and convert to a list of objects
    with open(input_file, 'r', encoding='utf-8') as f:
        # Read all lines and parse each as JSON
        data = [json.loads(line) for line in f if line.strip()]
    
    # Write as a single JSON array
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Example usage
convert_jsonl_to_json('data/processed_data/eae/rams_en_mT5/train.jsonl', 
                      'data/processed_data/eae/rams_en_mT5/train.json')
convert_jsonl_to_json('data/processed_data/eae/rams_en_mT5/dev.jsonl', 
                      'data/processed_data/eae/rams_en_mT5/dev.json')
convert_jsonl_to_json('data/processed_data/eae/rams_en_mT5/test.jsonl', 
                      'data/processed_data/eae/rams_en_mT5/test.json')
