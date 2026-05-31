import os
import torch
import threading
from threading import Thread
import customtkinter as ctk
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

def get_hf_token():
    # Retrieve Hugging Face token from environment variables
    return os.getenv("HUGGINGFACE_TOKEN")

class InferenceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Simple LLM Inference App (with Streaming)")
        self.geometry("750x650")
        
        self.model = None
        self.tokenizer = None
        
        # Configure grid layout
        self.grid_columnconfigure(0, weight=1)
        
        # Model Selection Frame
        self.model_frame = ctk.CTkFrame(self)
        self.model_frame.pack(pady=15, padx=20, fill="x")
        
        self.model_label = ctk.CTkLabel(self.model_frame, text="Select Model (Local or Hugging Face Base):", font=ctk.CTkFont(weight="bold"))
        self.model_label.pack(anchor="w", padx=15, pady=(10, 2))
        
        # Scan local directory and add Hugging Face Base model option
        self.model_dirs = self.scan_model_directories()
        self.model_combobox = ctk.CTkComboBox(self.model_frame, values=self.model_dirs, width=350)
        if self.model_dirs:
            self.model_combobox.set(self.model_dirs[-1])
        self.model_combobox.pack(side="left", padx=15, pady=(2, 15))
        
        self.load_button = ctk.CTkButton(self.model_frame, text="Load Selected Model", command=self.start_loading_model)
        self.load_button.pack(side="left", padx=5, pady=(2, 15))
        
        self.status_label = ctk.CTkLabel(self.model_frame, text="Status: No model loaded", text_color="orange")
        self.status_label.pack(side="left", padx=15, pady=(2, 15))
        
        # Prompt Section
        self.prompt_label = ctk.CTkLabel(self, text="Prompt / Medical Question:", font=ctk.CTkFont(weight="bold"))
        self.prompt_label.pack(anchor="w", padx=30, pady=(15, 2))
        
        self.prompt_textbox = ctk.CTkTextbox(self, height=130)
        self.prompt_textbox.insert("end", "What are the primary symptoms of chronic fatigue syndrome?")
        self.prompt_textbox.pack(padx=20, pady=(2, 10), fill="x")
        
        # Generate Button
        self.generate_button = ctk.CTkButton(self, text="Generate Response", command=self.start_generation, state="disabled", fg_color="green", hover_color="darkgreen")
        self.generate_button.pack(pady=10)
        
        # Output Display Section (For Token Streaming)
        self.output_label = ctk.CTkLabel(self, text="Generated Response (Streaming Output):", font=ctk.CTkFont(weight="bold"))
        self.output_label.pack(anchor="w", padx=30, pady=(15, 2))
        
        self.output_textbox = ctk.CTkTextbox(self, height=220, state="disabled")
        self.output_textbox.pack(padx=20, pady=(2, 20), fill="both", expand=True)

    def scan_model_directories(self):
        # Find folders in the current path that start with training outputs
        dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith("finetuned_model_epoch_")]
        dirs.sort()
        
        # Always prepend or append the raw un-fine-tuned base model option
        base_model_option = "Qwen/Qwen2.5-1.5B-Instruct (HF Base Model)"
        dirs.append(base_model_option)
        return dirs

    def start_loading_model(self):
        selected_path = self.model_combobox.get()
        token = get_hf_token()
        
        # Determine if loading from local directory or Hugging Face Hub
        is_base_model = "HF Base Model" in selected_path
        if is_base_model:
            load_path = "Qwen/Qwen2.5-1.5B-Instruct"
        else:
            load_path = selected_path
            
        if not is_base_model and not os.path.exists(load_path):
            self.status_label.configure(text="Status: Invalid path!", text_color="red")
            return
            
        self.load_button.configure(state="disabled")
        self.status_label.configure(text="Status: Loading weights...", text_color="yellow")
        
        # Load in a separate thread to prevent freezing the UI
        thread = threading.Thread(target=self._load_model_task, args=(load_path, token))
        thread.start()

    def _load_model_task(self, path, token):
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(path, token=token)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load model directly onto GPU/CUDA using modern mappings
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=token
            )
            
            # Update GUI elements on main thread
            self.after(0, lambda: self.status_label.configure(text="Status: Model Loaded!", text_color="green"))
            self.after(0, lambda: self.generate_button.configure(state="normal"))
        except Exception as e:
            self.after(0, lambda: self.status_label.configure(text="Error loading model", text_color="red"))
            self.after(0, lambda: self._update_output(f"Error loading weights: {str(e)}"))
        finally:
            self.after(0, lambda: self.load_button.configure(state="normal"))

    def start_generation(self):
        prompt_text = self.prompt_textbox.get("1.0", "end-1c").strip()
        if not prompt_text:
            return
            
        self.generate_button.configure(state="disabled")
        self._clear_output()
        
        # Run streaming inference process in a background thread
        thread = threading.Thread(target=self._generate_task, args=(prompt_text,))
        thread.start()

    def _generate_task(self, prompt_text):
        try:
            # Structure conversation into identical ChatML template
            messages = [
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": prompt_text}
            ]
            
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            device = self.model.device
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(device)
            
            # Set up the text streamer to extract generated tokens on the fly
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            # Run generation in a background sub-thread, managed by the streamer
            generation_thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            generation_thread.start()
            
            # Consume tokens from the streamer and append to GUI textbox in real-time
            for new_text in streamer:
                self.after(0, lambda text=new_text: self._append_output(text))
                
            generation_thread.join()
            
        except Exception as e:
            self.after(0, lambda: self._update_output(f"Inference error: {str(e)}"))
        finally:
            self.after(0, lambda: self.generate_button.configure(state="normal"))

    def _clear_output(self):
        self.output_textbox.configure(state="normal")
        self.output_textbox.delete("1.0", "end")
        self.output_textbox.configure(state="disabled")

    def _append_output(self, text):
        self.output_textbox.configure(state="normal")
        self.output_textbox.insert("end", text)
        self.output_textbox.see("end")
        self.output_textbox.configure(state="disabled")

    def _update_output(self, text):
        self._clear_output()
        self._append_output(text)

if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = InferenceApp()
    app.mainloop()