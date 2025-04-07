import os
import torch
import modal
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoProcessor, 
    AutoTokenizer, 
    LlamaForCausalLM, 
    LlamaTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from huggingface_hub import snapshot_download
import logging
from pathlib import Path
import json
import dotenv
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
import torch.nn as nn

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Define Modal App FIRST
app = modal.App("distillation-training") # Using modal.App

# --- Constants ---
MODEL_CACHE_PATH = "/cache/huggingface"
POC_DATA_FILE = "poc_distillation_data.jsonl"
PREPROCESSED_DATA_PATH = Path("/cache/data/preprocessed_data.pt")
# Ensure the directory for preprocessed data exists in the Volume
PREPROCESSED_DATA_DIR = PREPROCESSED_DATA_PATH.parent

TEACHER_MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct" # Needs H100 or similar
STUDENT_MODEL_ID = "Qwen/Qwen2.5-Omni-7B" # Should fit on smaller GPUs

BATCH_SIZE = 4
NUM_EPOCHS = 1
LEARNING_RATE = 5e-5

CUDA_VERSION = "12.4.0"  # Use the version from the example
FLAVOR = "devel"  # Includes full CUDA toolkit
OPERATING_SYS = "ubuntu22.04"
CUDA_TAG = f"{CUDA_VERSION}-{FLAVOR}-{OPERATING_SYS}"

# Define a persistent Modal Volume for caching model weights and data
# Using modal.Volume AFTER app definition
model_cache_volume = modal.Volume.from_name(
    "llama4-model-cache", create_if_missing=True
)

# Define the base image
# Using modal.Image AFTER app definition
# Use a base image with CUDA dev tools for flash-attn build
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.10")
    # Install critical build dependencies explicitly first (might be redundant with CUDA image, but safe)
    .pip_install("packaging", "torch", "numpy") # Combine installs
    # Then install the rest from the requirements file
    .pip_install_from_requirements("requirements-modal.txt")
    # Mount the raw data file (no need to mount .env as secret handles it)
    .add_local_file(POC_DATA_FILE, remote_path=f"/root/{POC_DATA_FILE}") # Mount raw data file
)

# Define GPU configuration
# Using modal.gpu AFTER app definition
h100_gpu = "H100" # Use string format
a10g_gpu = "A10G" # Use string format

# Path for the teacher model within the mounted volume (reflecting nested upload)
TEACHER_MODEL_VOLUME_PATH = "/cache/Llama-4-Scout-17B-16E-Instruct/Llama-4-Scout-17B-16E-Instruct"

# --- Helper Functions ---
def load_model_and_processor(model_id, model_cache_dir, device="cuda", is_teacher=False):
    """Loads a model and its processor/tokenizer.

    Args:
        model_id (str): Hugging Face model ID or local path.
        model_cache_dir (str): Path to the cache directory.
        device (str): Device to load the model onto ('cuda' or 'cpu').
        is_teacher (bool): Flag indicating if this is the teacher model.

    Returns:
        tuple: (model, processor)
    """
    print(f"Loading {'Teacher' if is_teacher else 'Student'} model: {model_id}")

    load_path = model_id
    model_class = AutoModelForCausalLM
    processor_class = AutoProcessor # Default to Auto classes
    tokenizer_class = AutoTokenizer

    if is_teacher:
        # Teacher model is pre-downloaded in the volume in original Llama format
        load_path = TEACHER_MODEL_VOLUME_PATH
        model_class = LlamaForCausalLM # Use specific Llama class
        # For Llama original format, we typically use LlamaTokenizer directly
        # AutoProcessor might not work well, let's use LlamaTokenizer
        # If Llama 4 Scout requires a specific Processor, we might need adjustment
        processor_class = LlamaTokenizer # Use LlamaTokenizer for processor loading
        tokenizer_class = LlamaTokenizer

        print(f"Teacher model specified. Loading from volume path: {load_path}")
        print(f"Using explicit classes: {model_class.__name__}, {processor_class.__name__}")
        # Verify the path exists in the volume
        if not os.path.exists(load_path):
            print(f"ERROR: Teacher model path not found in volume: {load_path}")
            print("Volume contents:")
            try:
                for item in os.listdir("/cache"):
                    print(f"  - /cache/{item}")
                if os.path.exists("/cache/Llama-4-Scout-17B-16E-Instruct"):
                     for item in os.listdir("/cache/Llama-4-Scout-17B-16E-Instruct"):
                         print(f"  - /cache/Llama-4-Scout-17B-16E-Instruct/{item}")
            except Exception as e:
                print(f"    Could not list volume contents: {e}")
            raise FileNotFoundError(f"Teacher model path not found: {load_path}")

    try:
        # Load processor/tokenizer
        print(f"Attempting to load processor/tokenizer using {processor_class.__name__} from: {load_path}")
        # Using AutoTokenizer/LlamaTokenizer as processor here, adjust if specific processor needed
        processor = processor_class.from_pretrained(
            load_path,
            cache_dir=model_cache_dir,
            trust_remote_code=True # May still be needed depending on tokenizer specifics
        )
        print("Processor/Tokenizer loaded successfully.")

        # Load model
        print(f"Attempting to load model using {model_class.__name__} from: {load_path}")
        model = model_class.from_pretrained(
            load_path,
            cache_dir=model_cache_dir,
            torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency on H100
            device_map=device, # Let accelerate handle device placement
            trust_remote_code=True, # May be needed for student model
            # attn_implementation="flash_attention_2" # Enable Flash Attention 2 if installed and compatible
        )
        print("Model loaded successfully.")
        return model, processor
    except OSError as e:
        # Specific handling for Hugging Face Hub errors (e.g., gated repo access for student)
        print(f"An OSError occurred while loading model {model_id} (path: {load_path}): {e}")
        if "gated repo" in str(e) and not is_teacher:
             print("This might be a gated repository access issue for the student model.")
             print("Ensure the HF_TOKEN secret is correctly configured and grants access.")
        elif "Connection error" in str(e):
             print("This looks like a network issue connecting to Hugging Face Hub.")
        # Add check for potential format issues when loading teacher model explicitly
        elif is_teacher and ("config.json" in str(e) or "Unrecognized model" in str(e)):
            print(f"Error loading teacher model from volume path {load_path}.")
            print("This might indicate the files are not in the expected format for LlamaForCausalLM/LlamaTokenizer, or essential files are missing.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading model {model_id} (path: {load_path}): {e}")
        raise

# --- Data Preprocessing ---
@app.function(
    image=image,
    volumes={"/cache": model_cache_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")], # Make .env available for HF_TOKEN
    gpu=h100_gpu
)
def preprocess_data():
    """
    Loads raw data, preprocesses (tokenizes) it using teacher and student processors,
    and saves the processed data to the volume.
    """
    print("--- Starting Data Preprocessing ---")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment. Model loading might fail if models are private.")

    # 1. Load Processors
    print("Loading processors...")
    # Use the helper function, ignore the model part for preprocessing
    # Specify distinct cache dirs to avoid conflicts if any files share names
    _, teacher_processor = load_model_and_processor(
        TEACHER_MODEL_ID, f"{MODEL_CACHE_PATH}/teacher", is_teacher=True
    )
    _, student_processor = load_model_and_processor(
        STUDENT_MODEL_ID, f"{MODEL_CACHE_PATH}/student", is_teacher=False
    )

    if teacher_processor is None or student_processor is None:
        print("ERROR: Failed to load one or both processors. Cannot preprocess data. Exiting.")
        # Optionally raise an exception here to halt execution
        return # Or simply return if you want the Modal function to finish

    # 2. Load Raw Data
    # The raw data file is mounted at /root/poc_distillation_data.jsonl
    raw_data_path_in_container = f"/root/{POC_DATA_FILE}"
    print(f"Loading raw data from {raw_data_path_in_container}...")
    raw_data = []
    try:
        with open(raw_data_path_in_container, 'r') as f:
            for line in f:
                try:
                    raw_data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line: {line.strip()}")
        print(f"Loaded {len(raw_data)} raw data points.")
    except FileNotFoundError:
        print(f"ERROR: Raw data file not found at {raw_data_path_in_container}. Make sure it's mounted correctly. Exiting.")
        return
    except Exception as e:
        print(f"ERROR reading raw data file: {e}. Exiting.")
        return

    # 3. Iterate and Tokenize
    processed_data = []
    print(f"Tokenizing data...") # Removed MAX_SEQ_LENGTH reference as it's handled by processor
    num_skipped = 0
    MAX_SEQ_LENGTH = 512 # Define or fetch from config if needed for padding/truncation
    for idx, item in enumerate(raw_data):
        input_text = item.get('input_text')
        # TODO: Handle image inputs if necessary

        if not input_text:
            print(f"Warning: No input_text found for item index {idx}. Skipping.")
            num_skipped += 1
            continue

        try:
            # Preprocess for Teacher Model
            teacher_inputs = teacher_processor(
                text=input_text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=MAX_SEQ_LENGTH
            )
            teacher_inputs = {k: v.squeeze(0) for k, v in teacher_inputs.items()}

            # Preprocess for Student Model
            student_inputs = student_processor(
                text=input_text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=MAX_SEQ_LENGTH
            )
            student_inputs = {k: v.squeeze(0) for k, v in student_inputs.items()}

            processed_data.append({
                'teacher_input_ids': teacher_inputs['input_ids'],
                'teacher_attention_mask': teacher_inputs['attention_mask'],
                'student_input_ids': student_inputs['input_ids'],
                'student_attention_mask': student_inputs['attention_mask'],
            })

        except Exception as e:
            print(f"ERROR processing item index {idx}: {e}. Text: '{input_text[:100]}...'. Skipping.")
            num_skipped += 1
            continue

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(raw_data)} items...")

    print(f"Finished tokenizing. Processed {len(processed_data)} items, skipped {num_skipped}.")

    # 4. Save Processed Data to Volume
    if not processed_data:
        print("ERROR: No data was successfully processed. Not saving output file.")
        return

    # Save to the mounted path: /cache/data/preprocessed_data.pt
    print(f"Saving {len(processed_data)} processed items to {PREPROCESSED_DATA_PATH}...")
    try:
        # Ensure the directory exists within the volume mount
        PREPROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Save the list of dictionaries containing tensors
        with open(PREPROCESSED_DATA_PATH, 'wb') as f_out:
            torch.save(processed_data, f_out)

        print("Processed data saved successfully.")
        # Commit changes to the volume to ensure visibility
        model_cache_volume.commit()
        print("Volume changes committed.")
    except Exception as e:
        print(f"ERROR saving processed data to {PREPROCESSED_DATA_PATH}: {e}")

# --- Dataset Class ---
class DistillationDataset(Dataset):
    """Dataset to load preprocessed data from a saved file."""
    def __init__(self, preprocessed_data_path):
        self.preprocessed_data_path = preprocessed_data_path
        self.data = []
        try:
            print(f"Loading preprocessed data from: {self.preprocessed_data_path}")
            # Check if file exists before attempting to load
            if not Path(self.preprocessed_data_path).is_file():
                 raise FileNotFoundError(f"Preprocessed data file not found at {self.preprocessed_data_path}")

            self.data = torch.load(self.preprocessed_data_path)
            print(f"Successfully loaded {len(self.data)} preprocessed items.")

            # Add a check for empty data
            if not self.data:
                print(f"Warning: Preprocessed data file {self.preprocessed_data_path} is empty.")

        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print("Please run the preprocessing step first: `modal run distillation_app.py --action preprocess`")
            raise # Re-raise to stop execution
        except Exception as e:
            print(f"ERROR loading preprocessed data: {e}")
            raise # Re-raise for other critical loading failures

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Data is already a list of dictionaries with tensors
        return self.data[idx]

# --- Training Function ---
@app.function(
    image=image,
    gpu=h100_gpu, # Use the updated string format
    volumes={"/cache": model_cache_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200 # 2 hours timeout
)
def train_distillation():
    """Main function to run the distillation training loop using preprocessed data."""
    print("--- Starting Distillation Training ---")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment.")

    # 1. Load Models
    print("\n--- Loading Models ---")
    # Use the helper function
    teacher_model, _ = load_model_and_processor(
        TEACHER_MODEL_ID, f"{MODEL_CACHE_PATH}/teacher", is_teacher=True
    )
    student_model, _ = load_model_and_processor(
        STUDENT_MODEL_ID, f"{MODEL_CACHE_PATH}/student", is_teacher=False
    )

    if teacher_model is None or student_model is None:
        print("ERROR: Failed to load one or both models. Exiting training.")
        return

    # Set teacher model to evaluation mode
    teacher_model.eval()

    # 2. Load Preprocessed Data
    print("\n--- Loading Preprocessed Data ---")
    try:
        # The preprocessed data path is defined globally and points to the volume mount
        train_dataset = DistillationDataset(PREPROCESSED_DATA_PATH)
        if len(train_dataset) == 0:
            print("ERROR: No data loaded. Preprocessed file might be empty or was not created properly. Exiting.")
            return

        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # Use shuffle for training
        print(f"Dataset loaded with {len(train_dataset)} samples. DataLoader created with batch size {BATCH_SIZE}.")

    except FileNotFoundError:
        # Dataset class already prints a specific message
        print("Exiting due to missing preprocessed data file.")
        return
    except Exception as e:
        print(f"ERROR setting up Dataset/DataLoader: {e}")
        return

    # 3. Setup Optimizer
    print("\n--- Setting up Optimizer ---")
    try:
        optimizer = AdamW(student_model.parameters(), lr=LEARNING_RATE)
        print(f"Optimizer AdamW configured with LR={LEARNING_RATE}")
    except Exception as e:
        print(f"Error setting up optimizer: {e}")
        return

    # 4. Define Distillation Loss Function (Example using KL Divergence)
    print("\n--- Setting up Loss Function ---")
    # Example: KL divergence loss expects log-probabilities
    # We need to apply softmax and log before passing to KLDivLoss
    # Temperature scaling is often applied here.
    temperature = 2.0 # Example temperature, hyperparameter to tune
    # KLDivLoss expects log-probabilities as input and probabilities as target
    # reduction='batchmean' averages the loss over the batch
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=False)
    print(f"Using KL Divergence Loss with temperature={temperature}")

    # --- Training Loop ---
    print("\n--- Starting Training Loop ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    student_model.train() # Set student model to training mode

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        total_loss = 0
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            # Move batch tensors to the correct device
            try:
                student_input_ids = batch['student_input_ids'].to(device)
                student_attention_mask = batch['student_attention_mask'].to(device)
                teacher_input_ids = batch['teacher_input_ids'].to(device) # Teacher inputs might differ if preprocessing was separate
                teacher_attention_mask = batch['teacher_attention_mask'].to(device)
            except KeyError as e:
                print(f"Error accessing batch data: {e}. Skipping batch {i}.")
                continue
            except Exception as e:
                 print(f"Error moving batch {i} to device: {e}. Skipping batch.")
                 continue

            # 1. Get Teacher Logits (no gradient needed)
            with torch.no_grad():
                try:
                    teacher_outputs = teacher_model(input_ids=teacher_input_ids, attention_mask=teacher_attention_mask)
                    teacher_logits = teacher_outputs.logits
                except Exception as e:
                    print(f"Error getting teacher logits for batch {i}: {e}. Skipping batch.")
                    continue

            # 2. Get Student Logits
            try:
                student_outputs = student_model(input_ids=student_input_ids, attention_mask=student_attention_mask)
                student_logits = student_outputs.logits
            except Exception as e:
                print(f"Error getting student logits for batch {i}: {e}. Skipping batch.")
                # Optionally, try to recover or just skip
                continue

            # 3. Calculate Distillation Loss (KL Divergence)
            # Ensure logits have the same shape
            if teacher_logits.shape != student_logits.shape:
                 print(f"Logit shape mismatch! Teacher: {teacher_logits.shape}, Student: {student_logits.shape}. Skipping batch {i}.")
                 # This could happen if vocab sizes differ or padding was inconsistent
                 continue

            # Apply temperature scaling and compute probabilities/log-probabilities
            teacher_probs_temp = F.softmax(teacher_logits / temperature, dim=-1)
            student_log_probs_temp = F.log_softmax(student_logits / temperature, dim=-1)

            # Calculate KL divergence loss
            # Ensure attention masks are applied correctly if sequences have padding
            # KLDivLoss needs input (student log_probs) and target (teacher probs)
            # We might need to reshape/mask the logits/probs based on the attention mask
            # to only compute loss on non-padding tokens.
            # For simplicity here, we compute over the whole sequence length.
            # A more precise implementation would mask the loss calculation.

            loss = kl_loss_fn(student_log_probs_temp, teacher_probs_temp)

            # TODO: Add CE loss if ground truth labels are available
            # TODO: Add other potential losses (e.g., cosine similarity on hidden states)

            # 4. Backpropagation and Optimizer Step
            try:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except Exception as e:
                print(f"Error during backward pass or optimizer step for batch {i}: {e}. Skipping step.")
                optimizer.zero_grad() # Ensure grads are cleared even if step failed

            if (i + 1) % 10 == 0: # Log every 10 batches
                print(f"  Batch {i+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

        avg_epoch_loss = total_loss / len(train_dataloader)
        print(f"--- Epoch {epoch+1} Finished. Average Loss: {avg_epoch_loss:.4f} ---")

    print("\n--- Training Finished ---")
    # TODO: Add code to save the trained student model checkpoint to the volume
    # student_model.save_pretrained(f"{MODEL_CACHE_PATH}/student_distilled_checkpoint")
    # model_cache_volume.commit()

# --- CLI Entrypoint ---
@app.local_entrypoint()
def main(action: str = "train"):
    if action == "preprocess":
        print("Starting data preprocessing...")
        preprocess_data.remote()
        print("Data preprocessing finished.")
    elif action == "train":
        print("Starting distillation training...")
        train_distillation.remote()
        print("Distillation training finished.")
    else:
        print(f"Unknown action: {action}. Use 'preprocess' or 'train'.")
