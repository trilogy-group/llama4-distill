import datasets
import pprint
import os
from dotenv import load_dotenv

# Load environment variables (specifically HF_TOKEN)
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    print("Warning: HF_TOKEN not found in environment variables or .env file.")
    print("Loading gated datasets (like the_stack, oscar) will likely fail.")

# --- Configuration (Copied from data_prep_poc.py, placeholders removed/commented) ---
TARGET_DATASETS = {
    # General/Instruction
    "open_orca": ("Open-Orca/OpenOrca", {}),
    "dolly": ("databricks/databricks-dolly-15k", {}),
    "alpaca": ("tatsu-lab/alpaca", {}),
    # Multimodal (Image/Text)
    # "laion": ("laion/laion2B-en", {"split": "train", "streaming": True}), # Skipped for now
    "conceptual_captions": ("google-research-datasets/conceptual_captions", {"split": "train"}),
    # "coco": ("HuggingFaceM4/COCO", {"split": "train"}),
    "vqav2": ("HuggingFaceM4/VQAv2", {"split": "train"}),
    # Code
    "code_alpaca": ("TokenBender/code_instructions_122k_alpaca_style", {}),
    "humaneval": ("openai_humaneval", {"split": "test"}),
    "mbpp": ("google-research-datasets/mbpp", {"split": "test"}),
    "the_stack": ("bigcode/the-stack-dedup", {"data_dir": "data/python", "split": "train", "streaming": True}),
    # Function/Tool
    # "tool_alpaca": ("alicia10/ToolAlpaca-scaled-dataset", {}), # Removed - Error during generation
    # Multilingual
    "oscar": ("oscar-corpus/OSCAR-2301", {"language": "en", "split": "train", "streaming": True, "trust_remote_code": True}),
    "xnli": ("xnli", {"name": "en", "split": "train"}),
    "paws-x": ("paws-x", {"name": "en", "split": "train"}),
}

# --- Main Inspection Script ---

print("Starting dataset inspection...")

for name, (path, args) in TARGET_DATASETS.items():
    print(f"\n--- Inspecting: {name} ({path}) with args {args} ---")

    # Skip datasets known to be problematic locally or with schema issues
    if name in ["laion", "coco", "vqav2", "gorilla"]:
        print(f"Skipping {name} due to size/local disk constraints or schema issues.")
        continue

    try:
        streaming = args.pop("streaming", False)
        split = args.pop("split", "train")
        trust_remote_code = args.pop("trust_remote_code", False)

        # Determine the config name (e.g., 'en' for oscar/xnli/paws-x)
        config_name = args.pop("name", None) or args.pop("language", None)

        print(f"Loading first item from split '{split}'...")
        ds = datasets.load_dataset(
            path,
            config_name, # Pass the config name (e.g., language code)
            data_files=args.get("data_files"),
            data_dir=args.get("data_dir"),
            split=split,
            streaming=streaming,
            token=hf_token,
            trust_remote_code=trust_remote_code
        )

        first_item = None
        if streaming:
            iterator = iter(ds)
            try:
                first_item = next(iterator)
            except StopIteration:
                print("Error: Stream is empty.")
            finally:
                 pass
        else:
            if len(ds) > 0:
                first_item = ds[0]
            else:
                print("Error: Dataset split is empty.")

        if first_item:
            print("First item structure:")
            pprint.pprint(first_item)
        else:
            print("Could not retrieve the first item.")

    except Exception as e:
        print(f"ERROR loading or accessing dataset {name} ({path}): {e}")

print("\nDataset inspection finished.")
