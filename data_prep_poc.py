import datasets
import random
import json
import os
from tqdm import tqdm

# --- Configuration ---
SAMPLE_SIZE = 1000
OUTPUT_FILE = "poc_distillation_data.jsonl"

# List of target datasets (name/path on Hub, potential subset/split args)
# Note: Some names/subsets might need refinement after testing
# Note: Added placeholder names where exact HF path wasn't immediately known
TARGET_DATASETS = {
    # General/Instruction
    "open_orca": ("Open-Orca/OpenOrca", {}),
    "dolly": ("databricks/databricks-dolly-15k", {}),
    "alpaca": ("tatsu-lab/alpaca", {}),
    # Multimodal (Image/Text)
    # LAION requires specific handling due to structure/size - STARTING WITH PLACEHOLDER
    # "laion": ("laion/laion2B-en", {"split": "train", "streaming": True}), # Example structure - needs testing
    "conceptual_captions": ("google-research-datasets/conceptual_captions", {"split": "train"}), # Updated
    "coco": ("HuggingFaceM4/COCO", {"split": "train"}),
    "vqav2": ("HuggingFaceM4/VQAv2", {"split": "train"}),
    # Code
    "code_alpaca": ("TokenBender/code_instructions_122k_alpaca_style", {}),
    "humaneval": ("openai_humaneval", {"split": "test"}), # Usually evaluated on test
    "mbpp": ("google-research-datasets/mbpp", {"split": "test"}), # Usually evaluated on test
    "the_stack": ("bigcode/the-stack-dedup", {"data_dir": "data/python", "split": "train", "streaming": True}),
    # Function/Tool
    # "tool_alpaca": ("alicia10/ToolAlpaca-scaled-dataset", {}), # Removed - Error during generation
    # Multilingual
    "oscar": ("oscar-corpus/OSCAR-2301", {"language": "en", "split": "train", "streaming": True, "trust_remote_code": True}), # Example: English
    "xnli": ("xnli", {"name": "en", "split": "train"}), # Specify English
    "paws-x": ("paws-x", {"name": "en", "split": "train"}), # Example: English
}

# --- Helper Functions ---

def get_input_fields(item, dataset_name):
    """Very basic extraction - NEEDS SIGNIFICANT REFINEMENT per dataset."""
    input_text = None
    input_image_ref = None

    # --- Add dataset-specific logic here --- 
    # This is highly dependent on the structure of each dataset
    if dataset_name == "open_orca":
        input_text = item.get('system_prompt', '') + "\n" + item.get('question', '')
    elif dataset_name == "dolly":
        input_text = item.get('instruction', '') + "\n" + item.get('context', '')
    elif dataset_name == "alpaca":
        input_text = item.get('instruction', '') + "\n" + item.get('input', '')
    elif dataset_name == "conceptual_captions":
        input_text = item.get('caption')
        input_image_ref = item.get('image_url') # Check if key is correct
    elif dataset_name == "coco":
        # COCO often has multiple captions per image_id.
        # NOTE: Skipped during POC sampling, structure not verified.
        # Placeholder assuming 'captions' is a list and 'image_url' field exists
        input_text = item.get('captions', [{}])[0].get('raw', '') if item.get('captions') else ''
        input_image_ref = item.get('image_url') # Or item.get('image') if data is present
    elif dataset_name == "vqav2":
        # NOTE: Skipped during POC sampling, structure not verified.
        # Placeholder assuming 'question' and 'image_url' fields exist.
        input_text = item.get('question')
        input_image_ref = item.get('image_url') # Or item.get('image') if data is present
    elif dataset_name == "code_alpaca":
        input_text = item.get('instruction')
        if item.get('input'):
            input_text += "\n" + item.get('input')
    elif dataset_name == "humaneval":
        input_text = item.get('prompt') # Contains signature, docstring etc.
    elif dataset_name == "mbpp":
        input_text = item.get('text') # Contains description
        # Also has 'code' field (solution) and 'test_list' (tests)
    elif dataset_name == "the_stack":
        input_text = item.get('content') # The code content
    elif dataset_name == "oscar":
        input_text = item.get('text')
    elif dataset_name == "xnli":
        # Input is premise + hypothesis
        input_text = f"Premise: {item.get('premise', '')}\nHypothesis: {item.get('hypothesis', '')}"
    elif dataset_name == "paws-x":
        # Input is two sentences
        input_text = f"Sentence1: {item.get('sentence1', '')}\nSentence2: {item.get('sentence2', '')}"
    else:
        # Default guess - try common text fields
        input_text = item.get('text') or item.get('instruction') or item.get('content')

    # --- End dataset-specific logic ---

    return {
        "input_text": str(input_text) if input_text else None,
        "input_image_ref": str(input_image_ref) if input_image_ref else None,
        "source_dataset": dataset_name
    }

# --- Main Script ---

all_samples = []

print(f"Starting dataset sampling (POC: {SAMPLE_SIZE} samples per dataset)")

for name, (path, args) in TARGET_DATASETS.items():
    print(f"\nProcessing: {name} ({path}) with args {args}")
    if "PLACEHOLDER" in path:
        print(f"Skipping placeholder dataset: {name}")
        continue
    # Skip LAION for now until specific handling is decided
    if name == "laion": # This check is redundant if LAION is commented out, but safe
        print(f"Skipping LAION dataset for POC: {name}")
        continue
    # Skip COCO and VQAv2 for POC due to size/complexity
    if name in ["coco", "vqav2"]:
        print(f"Skipping {name} dataset for POC due to size/complexity.")
        continue

    try:
        # Load dataset streamingly if specified, otherwise normally
        streaming = args.pop("streaming", False)
        split = args.pop("split", "train") # Default to train split
        trust_remote_code = args.pop("trust_remote_code", False)

        # Determine the config name (e.g., 'en' for oscar/xnli/paws-x)
        config_name = args.pop("name", None) or args.pop("language", None)

        ds = datasets.load_dataset(
            path,
            config_name, 
            data_files=args.get("data_files"),
            data_dir=args.get("data_dir"),
            split=split,
            streaming=streaming,
            trust_remote_code=trust_remote_code
        )

        samples = []
        if streaming:
            print(f"Streaming and sampling {SAMPLE_SIZE} items...")
            iterator = iter(ds)
            # Streaming doesn't easily support random sampling, take first N for POC
            for _ in tqdm(range(SAMPLE_SIZE), desc=f"Sampling {name}"):
                try:
                    item = next(iterator)
                    processed = get_input_fields(item, name)
                    if processed['input_text'] or processed['input_image_ref']:
                         samples.append(processed)
                    else:
                        print(f"Warning: Could not extract input from item in {name}: {item}")
                except StopIteration:
                    print(f"Warning: Stream ended before reaching {SAMPLE_SIZE} samples for {name}")
                    break
                except Exception as e:
                    print(f"Error processing streaming item from {name}: {e}")
        else:
            print(f"Loading full split '{split}'...")
            # Ensure dataset has enough samples
            num_rows = len(ds)
            print(f"Dataset size: {num_rows}")
            actual_sample_size = min(SAMPLE_SIZE, num_rows)
            if actual_sample_size < SAMPLE_SIZE:
                print(f"Warning: Dataset {name} has only {num_rows} items, taking all.")

            # Randomly sample indices
            print(f"Selecting {actual_sample_size} random indices...")
            indices = random.sample(range(num_rows), actual_sample_size)

            print(f"Gathering and processing samples...")
            sampled_ds = ds.select(indices)
            for item in tqdm(sampled_ds, desc=f"Processing {name}"):
                processed = get_input_fields(item, name)
                if processed['input_text'] or processed['input_image_ref']:
                     samples.append(processed)
                else:
                    print(f"Warning: Could not extract input from item in {name}: {item}")

        print(f"Collected {len(samples)} samples from {name}.")
        all_samples.extend(samples)

    except Exception as e:
        print(f"ERROR loading or processing dataset {name} ({path}): {e}")
        print("Skipping this dataset.")

print(f"\nTotal samples collected across all datasets: {len(all_samples)}")

# --- Save to JSON Lines --- 
print(f"\nSaving combined POC dataset to {OUTPUT_FILE}...")
try:
    with open(OUTPUT_FILE, 'w') as f:
        for item in tqdm(all_samples, desc="Writing JSONL"):
            f.write(json.dumps(item) + '\n')
    print("Successfully saved POC dataset.")
except Exception as e:
    print(f"Error writing output file: {e}")

print("\nData preparation POC script finished.")
print(f"Next steps: Inspect {OUTPUT_FILE}, refine preprocessing in get_input_fields, and then implement the actual data loading and processing for the distillation loop.")
