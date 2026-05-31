# Simple LLM Fine-Tuning Pipeline
A lightweight desktop GUI demonstrating a local, low-VRAM Supervised Fine-Tuning (SFT) pipeline. The system allows users to fetch a medical domain dataset, tokenize it using a model-specific template, unfreeze custom transformer layers, and observe training and validation loss evolution in real-time.

## Key Features
- **Local Low-VRAM SFT:** Safely fine-tunes specific end layers (1–4 blocks) of the `Qwen/Qwen2.5-1.5B-Instruct` model on consumer GPUs (e.g., RTX 3060 6GB) by freezing the rest of the 1.5B weights, including the heavy embedding and `lm_head` layers.
- **Automated ChatML Templating:** Automatically formats raw unstructured dataset entries into Qwen's specific ChatML prompt structure using `apply_chat_template` from `transformers` to ensure formatting alignment.
- **Real-Time Loss Plotting:** Dynamically renders the running average of the training loss step-by-step and plots distinct validation loss points at the end of each epoch using an integrated Matplotlib canvas.
- **Asynchronous Live Progress:** Utilizes non-blocking background threads to execute data fetching, tokenization, model loading, and training without freezing the CustomTkinter desktop UI.
- **Dynamic Metrics Counters:** Displays live training counters (e.g., `Training (Epoch 1): 396 / 9,000`) and evaluation trackers updating in real-time with every processed batch.
- **Decoupled Architecture:** Clean structural separation between the `customtkinter` GUI layer and the `torch`/`transformers` ML execution engine.

## Tech Stack
- **Frontend:** customtkinter (Modern Python Desktop UI), Matplotlib (Data Visualization)
- **Backend:** PyTorch, Hugging Face Transformers, Hugging Face Datasets
- **Model:** Qwen/Qwen2.5-1.5B-Instruct
- **Dataset:** medalpaca/medical_meadow_medical_flashcards
- **Concurrency:** Python threading for non-blocking ML operations and reactive GUI updates.

## Prerequisites
- **Python:** 3.12.x
- **CUDA Toolkit:** Compatible version for NVIDIA GPU acceleration (RTX series with 6GB+ VRAM highly recommended).
- **Disk Space:** Approximately 3.5GB for the local base Qwen model weights and saved tokenized cache folders.
- **Hugging Face Account:** A valid token saved in your environment variables.

## Installation

1. **Clone the repository:**
    
   git clone https://github.com/YourUsername/simple-llm-fine-tuning-pipeline
   
   cd simple-llm-fine-tuning-pipeline

Create and activate a virtual environment:

python -m venv venv

# On Windows:
.\venv\Scripts\activate
set HUGGINGFACE_TOKEN=your_token_here

# On Windows (PowerShell):
$env:HUGGINGFACE_TOKEN="your_token_here"

## Usage
Run the application:

python frontend.py

Fetch Data: Enter your desired number of examples (default 10000) and click "Fetch training data" to download the medical dataset and convert it into JSONL format.
Tokenize: Click "Tokenize datasets" to pad the records to 512 tokens and save them locally. Token padding indices (pad_token_id) are automatically mapped to -100 to prevent the model from learning padding artifacts.
Prepare Model: Select how many of the final block layers you wish to unfreeze (1–4) and click "Load & Prepare Model". This dynamically configures gradient parameters.
Fine-Tune: Set your batch size (default 4) and epochs (default 3). Click "Start Fine-Tuning" to watch the real-time loss graph update and monitor metrics in the log panel.
Project Structure
backend.py: Manages Hugging Face Hub interactions, tokenization schemes, gradient target configurations, and custom PyTorch training/validation loops.
frontend.py: Handles CustomTkinter visual states, window event loops, thread safety, and Matplotlib canvas updates.
requirements.txt: Package manifest including dependencies for model parsing, data manipulation, and visualization.

Please Note
After each epoch, the whole model (full weights) is saved, so make sure you have sufficent space on your hard drive

# Hardware Compatibility Note:

Tested on:
- CPU: Intel(R) Core(TM) i5-12500H
- RAM: 16.0 GB
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU (6 GB VRAM)

IMPORTANT NOTE: while it is possible to fine-tune up to even four layers of this model through 3 epochs on such a configuration, it is not recommended - for 10000 examples, it will take about 8 to 10 hours, and portable computers, and especially their cooling systems, are not well equipped for such a long and intensive strain.

### General Note on Fine-Tuning Dynamics & Overfitting

The practical outcome of any fine-tuning process is highly sensitive to data quality, dataset size, and hyperparameter configuration (e.g., learning rate scheduling scenario). Achieving stable convergence on validation loss often requires iterative experimentation.

Furthermore, to ensure true model **generalization**, it is critical to evaluate the final model on an independent, unseen test dataset. Tuning training parameters solely to minimize validation loss can lead to **implicit hyperparameter overfitting**. In this scenario, the chosen configuration becomes highly optimized for the validation set's specific distribution, but the model may fail to perform effectively on completely novel, out-of-distribution inputs.

Use inference.py to test your fine-tuning results.