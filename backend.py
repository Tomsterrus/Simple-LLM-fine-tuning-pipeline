import os
import json
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

finetune_model = None
finetune_tokenizer = None

def get_hf_token():
    return os.getenv("HUGGINGFACE_TOKEN")

def fetch_and_prepare_data(num_examples: int = 10000, train_ratio: float = 0.9):
    dataset_name = "medalpaca/medical_meadow_medical_flashcards"
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    token = get_hf_token()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    dataset = load_dataset(dataset_name, split="train", token=token)
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
    token = get_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    
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
    token = get_hf_token()
    
    finetune_tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if finetune_tokenizer.pad_token is None:
        finetune_tokenizer.pad_token = finetune_tokenizer.eos_token
        
    finetune_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=token
    )
    
    for param in finetune_model.parameters():
        param.requires_grad = False
        
    layers_to_unfreeze = finetune_model.model.layers[-num_layers_to_unfreeze:]
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True
            
    for param in finetune_model.model.norm.parameters():
        param.requires_grad = True
        
    trainable_params = sum(p.numel() for p in finetune_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in finetune_model.parameters())
    
    return trainable_params, total_params

def run_training(epochs: int, batch_size: int, lr: float, train_progress_callback, val_progress_callback, epoch_callback):
    global finetune_model, finetune_tokenizer
    if finetune_model is None or finetune_tokenizer is None:
        raise ValueError("Model is not loaded. Please prepare the model first.")
        
    train_dataset = load_from_disk("tokenized_train")
    val_dataset = load_from_disk("tokenized_val")
    
    total_train_examples = len(train_dataset)
    total_val_examples = len(val_dataset)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    finetune_model.to(device)
    
    def collate_fn(batch):
        input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
        attention_mask = torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long)
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    total_train_batches = len(train_loader)
    
    # Calculate scheduling steps
    num_training_steps = total_train_batches * epochs
    num_warmup_steps = int(0.1 * num_training_steps) # 10% warmup is standard
    
    trainable_params = [p for p in finetune_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    
    # Set up Cosine Scheduler with Warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    for epoch in range(1, epochs + 1):
        finetune_model.train()
        total_train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = finetune_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step() # Advance learning rate decay scheduler
            
            loss_val = loss.item()
            total_train_loss += loss_val
            
            avg_loss_so_far = total_train_loss / (batch_idx + 1)
            processed_examples = min((batch_idx + 1) * batch_size, total_train_examples)
            current_lr = scheduler.get_last_lr()[0] # Retrieve current learning rate
            
            if train_progress_callback:
                train_progress_callback(
                    epoch, 
                    batch_idx + 1, 
                    total_train_batches, 
                    processed_examples, 
                    total_train_examples, 
                    loss_val, 
                    avg_loss_so_far,
                    current_lr
                )
                
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation pass
        finetune_model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = finetune_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_val_loss += outputs.loss.item()
                
                processed_examples = min((batch_idx + 1) * batch_size, total_val_examples)
                if val_progress_callback:
                    val_progress_callback(epoch, processed_examples, total_val_examples)
                    
        avg_val_loss = total_val_loss / len(val_loader)
        
        output_dir = f"finetuned_model_epoch_{epoch}"
        finetune_model.save_pretrained(output_dir)
        finetune_tokenizer.save_pretrained(output_dir)
        
        if epoch_callback:
            epoch_callback(epoch, avg_train_loss, avg_val_loss, output_dir)