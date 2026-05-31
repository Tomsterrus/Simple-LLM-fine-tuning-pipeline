import os
import customtkinter as ctk
import threading
from backend import fetch_and_prepare_data, tokenize_data, prepare_model

class FineTuningApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Simple LLM fine-tuning pipeline")
        self.geometry("600x700")
        
        self.grid_columnconfigure(0, weight=1)
        
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Data Section
        self.data_label = ctk.CTkLabel(self.main_frame, text="1. Data Preparation", font=ctk.CTkFont(size=14, weight="bold"))
        self.data_label.grid(row=0, column=0, pady=(10, 5))
        
        self.examples_entry = ctk.CTkEntry(self.main_frame, justify="center")
        self.examples_entry.insert(0, "10000")
        self.examples_entry.grid(row=1, column=0, pady=5)
        
        self.fetch_button = ctk.CTkButton(self.main_frame, text="Fetch training data", command=self.on_fetch_clicked)
        self.fetch_button.grid(row=2, column=0, pady=5)
        
        self.tokenize_button = ctk.CTkButton(self.main_frame, text="Tokenize datasets", command=self.on_tokenize_clicked)
        self.tokenize_button.grid(row=3, column=0, pady=(5, 15))
        
        # Model Section
        self.model_label = ctk.CTkLabel(self.main_frame, text="2. Model Preparation", font=ctk.CTkFont(size=14, weight="bold"))
        self.model_label.grid(row=4, column=0, pady=(10, 5))
        
        self.layers_label = ctk.CTkLabel(self.main_frame, text="Last layers to unfreeze (1-4):")
        self.layers_label.grid(row=5, column=0, pady=(0, 0))
        
        self.layers_combobox = ctk.CTkComboBox(self.main_frame, values=["1", "2", "3", "4"], justify="center")
        self.layers_combobox.set("2")
        self.layers_combobox.grid(row=6, column=0, pady=5)
        
        self.prepare_model_button = ctk.CTkButton(self.main_frame, text="Load & Prepare Model", command=self.on_prepare_model_clicked)
        self.prepare_model_button.grid(row=7, column=0, pady=(5, 15))
        
        # Logs Section
        self.log_box = ctk.CTkTextbox(self.main_frame, height=200, state="disabled")
        self.log_box.grid(row=8, column=0, padx=10, pady=10, sticky="ew")

    def log_message(self, message: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def toggle_buttons(self, state: str):
        self.fetch_button.configure(state=state)
        self.tokenize_button.configure(state=state)
        self.prepare_model_button.configure(state=state)

    def on_fetch_clicked(self):
        try:
            num_examples = int(self.examples_entry.get())
        except ValueError:
            self.log_message("Error: Number of examples must be an integer.")
            return

        self.toggle_buttons("disabled")
        self.log_message(f"Fetching and preparing {num_examples} examples...")
        
        thread = threading.Thread(target=self._fetch_task, args=(num_examples,))
        thread.start()

    def _fetch_task(self, num_examples):
        try:
            train_count, val_count = fetch_and_prepare_data(num_examples=num_examples)
            self.log_message(f"Success! Saved train.jsonl ({train_count} examples).")
            self.log_message(f"Success! Saved val.jsonl ({val_count} examples).")
        except Exception as e:
            self.log_message(f"Error during data fetching: {str(e)}")
        finally:
            self.toggle_buttons("normal")

    def on_tokenize_clicked(self):
        self.toggle_buttons("disabled")
        self.log_message("Starting tokenization process...")
        
        thread = threading.Thread(target=self._tokenize_task)
        thread.start()

    def _tokenize_task(self):
        try:
            train_count, val_count = tokenize_data()
            self.log_message(f"Success! Saved tokenized train dataset ({train_count} examples).")
            self.log_message(f"Success! Saved tokenized val dataset ({val_count} examples).")
        except FileNotFoundError as e:
            self.log_message(f"Error: {str(e)}")
        except Exception as e:
            self.log_message(f"Error during tokenization: {str(e)}")
        finally:
            self.toggle_buttons("normal")

    def on_prepare_model_clicked(self):
        if not os.path.exists("tokenized_train") or not os.path.exists("tokenized_val"):
            self.log_message("Error: Tokenized data not found. Please tokenize first.")
            return
            
        try:
            num_layers = int(self.layers_combobox.get())
            if not 1 <= num_layers <= 4:
                raise ValueError
        except ValueError:
            self.log_message("Error: Select a valid number of layers (1-4).")
            return
            
        self.toggle_buttons("disabled")
        self.log_message(f"Loading model and unfreezing {num_layers} layers...")
        
        thread = threading.Thread(target=self._prepare_model_task, args=(num_layers,))
        thread.start()

    def _prepare_model_task(self, num_layers):
        try:
            trainable, total = prepare_model(num_layers)
            percentage = (trainable / total) * 100
            self.log_message(f"Success! Model prepared.")
            self.log_message(f"Trainable params: {trainable:,} / {total:,} ({percentage:.2f}%)")
        except Exception as e:
            self.log_message(f"Error during model preparation: {str(e)}")
        finally:
            self.toggle_buttons("normal")

if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    
    app = FineTuningApp()
    app.mainloop()