import json
from datasets import load_dataset
from transformers import AutoTokenizer

def fetch_and_prepare_data(num_examples: int = 10000, train_ratio: float = 0.9):
    dataset_name = "medalpaca/medical_meadow_medical_flashcards"
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # Load tokenizer for applying chat template
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load dataset
    dataset = load_dataset(dataset_name, split="train")
    
    # Limit number of examples
    num_examples = min(num_examples, len(dataset))
    
    # Shuffle and select subset
    dataset = dataset.shuffle(seed=42).select(range(num_examples))
    
    # Split into train and validation sets
    split_dataset = dataset.train_test_split(test_size=(1 - train_ratio), seed=42)
    train_data = split_dataset["train"]
    val_data = split_dataset["test"]
    
    def save_to_jsonl(data, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                user_content = item['instruction']
                if item.get('input'):
                    user_content += f"\n{item['input']}"
                
                # Define conversation structure
                messages = [
                    {"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": item['output']}
                ]
                
                # Apply model-specific chat template (as a raw string, not tokenized yet)
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                
                record = {"text": formatted_prompt}
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Save splits to JSONL files
    save_to_jsonl(train_data, "train.jsonl")
    save_to_jsonl(val_data, "val.jsonl")
    
    return len(train_data), len(val_data)