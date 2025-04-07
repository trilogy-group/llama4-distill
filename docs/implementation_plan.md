# Llama 4 Scout -> Qwen2.5-Omni-7B Knowledge Distillation Plan

## 1. Objective

Distill the knowledge from the larger Llama 4 Scout 17B model into the smaller Qwen2.5-Omni-7B model to create a more efficient model that retains capabilities from the teacher, leveraging Modal for GPU-accelerated training.

## 2. Models

*   **Teacher Model:** `meta-llama/Llama-4-Scout-17B-16E-Instruct` (17B active parameters, Multimodal via `AutoProcessor`)
    *   Source: Hugging Face
*   **Student Model:** `Qwen/Qwen2.5-Omni-7B` (7B parameters, Multimodal via `Qwen2_5OmniProcessor`)
    *   Source: Hugging Face

## 3. Infrastructure

*   **Compute:** Modal Labs
    *   Target GPU: `H100`
*   **Core Libraries:**
    *   `modal-client`
    *   `transformers` (>= v4.51.0 for Llama 4)
    *   `torch`
    *   `qwen_omni_utils` (for Qwen preprocessing)
    *   `datasets` (for data loading/handling)
    *   `accelerate` (potentially useful for distributed setup/memory saving)
    *   `soundfile` (dependency for Qwen utils)
    *   `wandb` or `mlflow` (Optional, for logging metrics)

## 4. Distillation Strategy

*   **Method:** Response-based distillation using Logit Matching.
*   **Loss Function:** Combination of:
    1.  **KL Divergence Loss:** Minimize the KL divergence between the softened probability distributions (logits + temperature scaling) of the teacher and student models.
    2.  --Cross-Entropy Loss (Not Applicable):-- If using a labeled dataset, include the standard cross-entropy loss between the student's predictions and the ground truth labels. (We are not doing this in this case.)
    *   Weighting between these losses will need tuning.
*   **Temperature:** Use temperature scaling (T > 1) on both teacher and student logits before calculating KL divergence to soften probabilities.

## 5. Dataset

*   **Requirement:** A diverse dataset suitable for distillation, ideally covering the modalities both models handle (text, potentially images).
*   **POC Dataset Used:** A combined dataset was created by sampling from various Hugging Face datasets using `data_prep_poc.py`, resulting in `poc_distillation_data.jsonl`. The included sources are:
    *   General/Instruction: `open_orca`, `dolly`, `alpaca`
    *   Multimodal (Text): `conceptual_captions`
    *   Code: `code_alpaca`, `humaneval`, `mbpp`, `the_stack`
    *   Multilingual (English): `oscar`, `xnli`, `paws-x`
    *   *Note: `coco`, `vqav2`, `laion` were skipped for the initial POC.* 
*   **Preprocessing:** 
    *   Basic input field extraction (`input_text`, `input_image_ref`) is done in `data_prep_poc.py`.
    *   Further preprocessing (tokenization compatible with both teacher and student) is required for the training loop.

## 6. Implementation Steps

*   [X] **Setup Modal Environment:**
    *   [X] Create a `distillation_app.py` defining the Modal app and functions.
    *   [X] Define a `modal.Image` with necessary dependencies installed (`requirements.txt`).
    *   [X] Update `requirements.txt` with libraries (needs verification for all dependencies like `qwen_omni_utils` if not already included).
    *   [X] Configure Modal secrets for Hugging Face Hub token.
    *   [X] Define and mount Modal Volume (`model_cache_volume`) for model caching.
*   [X] **Data Preparation (POC):**
    *   [X] Selected and sampled from diverse datasets (see Section 5).
    *   [X] Implemented basic input field extraction (`data_prep_poc.py`).
    *   [X] Generated POC dataset file (`poc_distillation_data.jsonl`).
*   [In Progress] **Data Preparation (Training):**
    *   [In Progress] Implement preprocessing logic (tokenization, padding) compatible with both models within `DistillationDataset` (`__getitem__`).
        *   _Note: Currently done on-the-fly. Consider pre-processing._
        *   _TODO: Handle multimodal inputs if necessary._
    *   [X] Create a PyTorch `DistillationDataset` class.
    *   [X] Instantiate `DataLoader` in the training function.
*   [In Progress] **Model Loading:**
    *   [X] Implement `load_teacher_model` function with caching.
    *   [X] Implement `load_student_model` function with caching.
    *   [ ] Verify model loading works correctly in the Modal H100 environment.
*   [ ] **Distillation Training Loop:**
    *   [ ] Decorate the training function with the required GPU resources (e.g., `@app.function(gpu="H100", timeout=..., container_idle_timeout=...)`).
    *   [ ] Load models (teacher in `eval()` mode).
    *   [ ] Load dataset/dataloader.
    *   [ ] Instantiate optimizer (acting only on student parameters).
    *   [ ] Loop through data batches:
        *   [ ] Prepare batch inputs for both models.
        *   [ ] Run inference on Teacher (`with torch.no_grad(): ...`). Get logits.
        *   [ ] Run forward pass on Student. Get logits.
        *   [ ] Apply temperature scaling to logits.
        *   [ ] Calculate KL Divergence loss.
        *   [ ] Calculate Cross-Entropy loss (if applicable).
        *   [ ] Combine losses (weighted sum).
        *   [ ] Perform `loss.backward()` on the combined loss.
        *   [ ] `optimizer.step()` (updates student weights).
        *   [ ] `optimizer.zero_grad()`.
        *   [ ] Log metrics (loss components, etc.).
*   [ ] **Checkpointing:**
    *   [ ] Implement logic to save student model checkpoints periodically (e.g., to a Modal `NetworkFileSystem` or Volume).
*   [ ] **Evaluation:**
    *   [ ] Define evaluation metrics (e.g., Perplexity on a validation set, specific task benchmarks).
    *   [ ] Create a separate Modal function for evaluating the distilled student model checkpoints.

## 7. Potential Challenges & Considerations

*   **GPU Memory:** The 17B teacher model is large. `H100` is necessary. Quantization or model parallelism (`accelerate`) might be explored if needed.
*   **Compute Cost & Time:** Teacher inference adds significant overhead. Optimize batch sizes and consider efficient implementations (e.g., `flash_attention_2` for Qwen).
*   **Preprocessing Alignment:** Ensuring multimodal inputs are handled consistently and correctly by both `AutoProcessor` (Llama 4) and `Qwen2_5OmniProcessor` is crucial.
*   **Hyperparameter Tuning:** Temperature (T), loss weighting, learning rate, batch size, optimizer choice.
*   **Modality Mismatch:** Focus distillation on common capabilities (e.g., text generation, potentially image understanding if using multimodal data) where both models have overlap. Qwen's audio capabilities might not be directly distillable from Llama 4 Scout unless the teacher also has them explicitly.

## 8. Progress Tracking

(Use checkboxes above to track completion of steps)
