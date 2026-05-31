import customtkinter as ctk
import threading
from backend import fetch_and_prepare_data

class FineTuningApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Simple LLM fine-tuning pipeline")
        self.geometry("600x450")
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        
        # Main Frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Data Preparation Section
        self.data_label = ctk.CTkLabel(self.main_frame, text="Data Preparation", font=ctk.CTkFont(size=16, weight="bold"))
        self.data_label.grid(row=0, column=0, pady=(10, 5))
        
        self.examples_label = ctk.CTkLabel(self.main_frame, text="Number of examples:")
        self.examples_label.grid(row=1, column=0, pady=(10, 0))
        
        self.examples_entry = ctk.CTkEntry(self.main_frame, justify="center")
        self.examples_entry.insert(0, "10000")
        self.examples_entry.grid(row=2, column=0, pady=5)
        
        self.fetch_button = ctk.CTkButton(self.main_frame, text="Fetch training data", command=self.on_fetch_clicked)
        self.fetch_button.grid(row=3, column=0, pady=15)
        
        # Log Box
        self.log_box = ctk.CTkTextbox(self.main_frame, height=200, state="disabled")
        self.log_box.grid(row=4, column=0, padx=10, pady=10, sticky="ew")

    def log_message(self, message: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def on_fetch_clicked(self):
        try:
            num_examples = int(self.examples_entry.get())
        except ValueError:
            self.log_message("Error: Number of examples must be an integer.")
            return

        self.fetch_button.configure(state="disabled")
        self.log_message(f"Fetching and preparing {num_examples} examples...")
        
        # Run in a separate thread to prevent GUI freezing
        thread = threading.Thread(target=self._fetch_task, args=(num_examples,))
        thread.start()

    def _fetch_task(self, num_examples):
        try:
            train_count, val_count = fetch_and_prepare_data(num_examples=num_examples)
            self.log_message(f"Success! Saved train.jsonl ({train_count} examples).")
            self.log_message(f"Success! Saved val.jsonl ({val_count} examples).")
        except Exception as e:
            self.log_message(f"Error during data preparation: {str(e)}")
        finally:
            self.fetch_button.configure(state="normal")

if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    
    app = FineTuningApp()
    app.mainloop()