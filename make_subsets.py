import json
import random
import os

def create_subset(input_file, output_file, percentage=0.1, seed=42):
    # Set seed for reproducibility
    random.seed(seed)
    
    # Read the original data
    with open(input_file, 'r', encoding='utf-8') as f:
        if input_file.endswith('.jsonl'):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
    
    # Calculate subset size
    subset_size = max(1, int(len(data) * percentage))
    
    # Randomly select subset
    subset = random.sample(data, subset_size)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write the subset to file
    with open(output_file, 'w', encoding='utf-8') as f:
        if input_file.endswith('.jsonl'):
            for item in subset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            json.dump(subset, f, ensure_ascii=False, indent=2)
    
    print(f"Created subset with {subset_size} items ({percentage*100}%) at {output_file}")

# Create subsets for all splits
for split in ['train', 'dev', 'test']:
    input_file = f'data/processed_data/eae/rams_en_mT5/{split}.jsonl'
    output_file = f'data/processed_data/eae/rams_en_mT5_10percent/{split}.json'
    create_subset(input_file, output_file)