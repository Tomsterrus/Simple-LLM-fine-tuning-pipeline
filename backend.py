import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer

def fetch_and_prepare_data(num_examples: int = 10000, train_ratio: float = 0.9):
    dataset_name = "medalpaca/medical_meadow_medical_flashcards"
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = load_dataset(dataset_name, split="train")
    num_examples = min(num_examples, len(dataset))
    dataset = dataset.shuffle(seed=42).select(range(num_examples))
    
    split_dataset = dataset.train_test_split(test_size=(1 - train_ratio), seed=42)
    train_data = split_dataset["train"]
    val_data = split_dataset["test"]
    
    def save_to_jsonl(data, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                user_content = item['instruction']
                if item.get('input'):
                    user_content += f"\n{item['input']}"
                
                messages = [
                    {"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": item['output']}
                ]
                
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                
                record = {"text": formatted_prompt}
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    save_to_jsonl(train_data, "train.jsonl")
    save_to_jsonl(val_data, "val.jsonl")
    
    return len(train_data), len(val_data)

def tokenize_data(max_length: int = 512):
    if not os.path.exists("train.jsonl") or not os.path.exists("val.jsonl"):
        raise FileNotFoundError("JSONL files not found. Please fetch training data first.")
    
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    data_files = {"train": "train.jsonl", "validation": "val.jsonl"}
    dataset = load_dataset("json", data_files=data_files)
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
        
        # Create labels for causal language modeling and ignore pad tokens (-100)
        labels = []
        for input_ids in tokenized["input_ids"]:
            label = [-100 if token == tokenizer.pad_token_id else token for token in input_ids]
            labels.append(label)
            
        tokenized["labels"] = labels
        return tokenized
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Save to disk
    tokenized_datasets["train"].save_to_disk("tokenized_train")
    tokenized_datasets["validation"].save_to_disk("tokenized_val")
    
    return len(tokenized_datasets["train"]), len(tokenized_datasets["validation"])