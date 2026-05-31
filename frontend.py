import os
import customtkinter as ctk
import threading
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from backend import fetch_and_prepare_data, tokenize_data, prepare_model, run_training

class FineTuningApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Simple LLM fine-tuning pipeline")
        self.geometry("1100x820")
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        self.left_frame.grid_columnconfigure(0, weight=1)
        
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(0, weight=1)
        
        self.data_label = ctk.CTkLabel(self.left_frame, text="1. Data Preparation", font=ctk.CTkFont(size=14, weight="bold"))
        self.data_label.grid(row=0, column=0, pady=(10, 5))
        
        self.examples_entry = ctk.CTkEntry(self.left_frame, justify="center", width=120)
        self.examples_entry.insert(0, "10000")
        self.examples_entry.grid(row=1, column=0, pady=5)
        
        self.fetch_button = ctk.CTkButton(self.left_frame, text="Fetch training data", command=self.on_fetch_clicked, width=150)
        self.fetch_button.grid(row=2, column=0, pady=5)
        
        self.tokenize_button = ctk.CTkButton(self.left_frame, text="Tokenize datasets", command=self.on_tokenize_clicked, width=150)
        self.tokenize_button.grid(row=3, column=0, pady=(5, 15))
        
        self.model_label = ctk.CTkLabel(self.left_frame, text="2. Model Preparation (no. of layers to unfreeze)", font=ctk.CTkFont(size=14, weight="bold"))
        self.model_label.grid(row=4, column=0, pady=(5, 5))
        
        self.layers_combobox = ctk.CTkComboBox(self.left_frame, values=["1", "2", "3", "4"], justify="center", width=120)
        self.layers_combobox.set("2")
        self.layers_combobox.grid(row=5, column=0, pady=5)
        
        self.prepare_model_button = ctk.CTkButton(self.left_frame, text="Load & Prepare Model", command=self.on_prepare_model_clicked, width=150)
        self.prepare_model_button.grid(row=6, column=0, pady=(5, 15))
        
        self.train_label = ctk.CTkLabel(self.left_frame, text="3. Training Settings", font=ctk.CTkFont(size=14, weight="bold"))
        self.train_label.grid(row=7, column=0, pady=(5, 5))
        
        self.batch_label = ctk.CTkLabel(self.left_frame, text="Batch size:")
        self.batch_label.grid(row=8, column=0)
        self.batch_entry = ctk.CTkEntry(self.left_frame, justify="center", width=120)
        self.batch_entry.insert(0, "4")
        self.batch_entry.grid(row=9, column=0, pady=2)
        
        self.epochs_label = ctk.CTkLabel(self.left_frame, text="Epochs:")
        self.epochs_label.grid(row=10, column=0)
        self.epochs_entry = ctk.CTkEntry(self.left_frame, justify="center", width=120)
        self.epochs_entry.insert(0, "3")
        self.epochs_entry.grid(row=11, column=0, pady=2)
        
        self.start_train_button = ctk.CTkButton(self.left_frame, text="Start Fine-Tuning", command=self.on_start_train_clicked, fg_color="green", hover_color="darkgreen", width=150)
        self.start_train_button.grid(row=12, column=0, pady=15)
        
        self.train_progress_label = ctk.CTkLabel(self.left_frame, text="Training: - / - (Loss: -)", font=ctk.CTkFont(size=12, weight="bold"))
        self.train_progress_label.grid(row=13, column=0, pady=(5, 2))
        
        self.val_progress_label = ctk.CTkLabel(self.left_frame, text="Validation: - / -", font=ctk.CTkFont(size=12, weight="bold"))
        self.val_progress_label.grid(row=14, column=0, pady=(2, 10))
        
        self.log_box = ctk.CTkTextbox(self.left_frame, height=180, state="disabled")
        self.log_box.grid(row=15, column=0, padx=10, pady=10, sticky="ew")
        
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Training & Validation Loss")
        self.ax.set_xlabel("Global Training Step (Batches)")
        self.ax.set_ylabel("Loss")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        self.train_steps = []
        self.train_losses = []
        self.val_steps = []
        self.val_losses = []
        self.steps_per_epoch = 0

    def log_message(self, message: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def toggle_buttons(self, state: str):
        self.fetch_button.configure(state=state)
        self.tokenize_button.configure(state=state)
        self.prepare_model_button.configure(state=state)
        self.start_train_button.configure(state=state)

    def draw_plot(self):
        self.ax.clear()
        
        if self.train_steps:
            self.ax.plot(self.train_steps, self.train_losses, label="Train Loss (Running Avg)", color="blue")
            
        if self.val_steps:
            self.ax.scatter(self.val_steps, self.val_losses, label="Val Loss", marker="x", color="red", s=100, zorder=5)
            if len(self.val_steps) > 1:
                self.ax.plot(self.val_steps, self.val_losses, color="red", linestyle="--", alpha=0.7)
                
        self.ax.set_title("Training & Validation Loss Evolution")
        self.ax.set_xlabel("Global Training Step (Batches)")
        self.ax.set_ylabel("Loss")
        self.ax.grid(True, linestyle="--", alpha=0.6)
        self.ax.legend()
        
        self.ax.relim()
        self.ax.autoscale_view()
        
        self.canvas.draw()

    def update_plot_data_threadsafe(self, step, loss, is_val=False):
        if is_val:
            self.val_steps.append(step)
            self.val_losses.append(loss)
        else:
            self.train_steps.append(step)
            self.train_losses.append(loss)
            
        self.after(0, self.draw_plot)

    def update_train_progress_threadsafe(self, epoch, current, total, loss, current_lr):
        self.after(0, lambda: self.train_progress_label.configure(
            text=f"Training (Epoch {epoch}): {current:,} / {total:,} (Loss: {loss:.4f} | LR: {current_lr:.2e})"
        ))

    def update_val_progress_threadsafe(self, epoch, current, total):
        self.after(0, lambda: self.val_progress_label.configure(
            text=f"Validation (Epoch {epoch}): {current:,} / {total:,}"
        ))

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

    def on_start_train_clicked(self):
        try:
            epochs = int(self.epochs_entry.get())
            batch_size = int(self.batch_entry.get())
        except ValueError:
            self.log_message("Error: Epochs and Batch size must be integers.")
            return
            
        self.toggle_buttons("disabled")
        self.log_message(f"Starting training: Epochs={epochs}, Batch Size={batch_size}...")
        
        self.train_steps = []
        self.train_losses = []
        self.val_steps = []
        self.val_losses = []
        self.steps_per_epoch = 0
        self.draw_plot()
        
        thread = threading.Thread(target=self._train_task, args=(epochs, batch_size))
        thread.start()

    def _train_task(self, epochs, batch_size):
        def train_progress_callback(epoch, step, total_steps, current, total, loss, avg_loss_so_far, current_lr):
            self.update_train_progress_threadsafe(epoch, current, total, loss, current_lr)
            
            if self.steps_per_epoch == 0:
                self.steps_per_epoch = total_steps
                
            global_step = (epoch - 1) * total_steps + step
            if global_step % 10 == 0 or step == total_steps:
                self.update_plot_data_threadsafe(global_step, avg_loss_so_far, is_val=False)
                
            if current % (batch_size * 5) == 0 or current == total:
                self.log_message(f"Epoch {epoch} | Processed: {current}/{total} | Loss: {loss:.4f} | Avg: {avg_loss_so_far:.4f} | LR: {current_lr:.2e}")
                
        def val_progress_callback(epoch, current, total):
            self.update_val_progress_threadsafe(epoch, current, total)

        def epoch_callback(epoch, avg_train_loss, avg_val_loss, output_dir):
            self.log_message(f"--- Epoch {epoch} Completed ---")
            self.log_message(f"Average Train Loss: {avg_train_loss:.4f}")
            self.log_message(f"Average Val Loss: {avg_val_loss:.4f}")
            self.log_message(f"Model saved to: {output_dir}\n")
            
            epoch_end_step = epoch * self.steps_per_epoch
            self.update_plot_data_threadsafe(epoch_end_step, avg_val_loss, is_val=True)

        try:
            run_training(
                epochs=epochs,
                batch_size=batch_size,
                lr=2e-5,
                train_progress_callback=train_progress_callback,
                val_progress_callback=val_progress_callback,
                epoch_callback=epoch_callback
            )
            self.log_message("Training completed successfully!")
        except Exception as e:
            self.log_message(f"Error during training: {str(e)}")
        finally:
            self.toggle_buttons("normal")

if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    
    app = FineTuningApp()
    app.mainloop()