import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

finetune_model = None
finetune_tokenizer = None

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
        
        labels = []
        for input_ids in tokenized["input_ids"]:
            label = [-100 if token == tokenizer.pad_token_id else token for token in input_ids]
            labels.append(label)
            
        tokenized["labels"] = labels
        return tokenized
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    tokenized_datasets["train"].save_to_disk("tokenized_train")
    tokenized_datasets["validation"].save_to_disk("tokenized_val")
    
    return len(tokenized_datasets["train"]), len(tokenized_datasets["validation"])

def prepare_model(num_layers_to_unfreeze: int):
    global finetune_model, finetune_tokenizer
    
    if not os.path.exists("tokenized_train") or not os.path.exists("tokenized_val"):
        raise FileNotFoundError("Tokenized datasets not found. Please tokenize data first.")
        
    if not 1 <= num_layers_to_unfreeze <= 4:
        raise ValueError("You can only unfreeze between 1 and 4 layers.")
        
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
    finetune_tokenizer = AutoTokenizer.from_pretrained(model_id)
    if finetune_tokenizer.pad_token is None:
        finetune_tokenizer.pad_token = finetune_tokenizer.eos_token
        
    finetune_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Freeze all parameters
    for param in finetune_model.parameters():
        param.requires_grad = False
        
    # Unfreeze only the specified number of last layers
    layers_to_unfreeze = finetune_model.model.layers[-num_layers_to_unfreeze:]
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True
            
    # Keep embed_tokens and lm_head frozen to save memory
    # Unfreeze only the final layer norm (very small: 1,536 parameters)
    for param in finetune_model.model.norm.parameters():
        param.requires_grad = True
        
    # Enable gradient checkpointing to save memory
    finetune_model.gradient_checkpointing_enable()
    
    trainable_params = sum(p.numel() for p in finetune_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in finetune_model.parameters())
    
    return trainable_params, total_params