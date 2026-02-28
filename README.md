# CollateralScore

Developed by the Charité Lab for AI in Medicine (CLAIM) research group at Charité University Hospital, Berlin, main developer and person to contact: Dimitrios Rallios (dimitrios.rallios@charite.de)

Goal = Automated Collateral Score Grading based on cerebrovascular radiomics.
Input = Niftis of CT Angiographies of patients with a LVO

Step 1 -> CTA preprocessing and nn-Unet-based Vessel Segmentation.
Step 2 -> Selected Radiomics Extraction and RFC prediction between sufficient (Tan Score 2 and 3) and insufficient (Tan Score 0 and 1)

# Environment
Use one unified Python environment managed by `uv`.

```bash
# Optional: TUNA mirror (mainland China)
uv lock --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
uv sync --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"

# Default index
# uv sync
```

# Configure Paths (.env)
Copy `.env.example` to `.env`. Default values are relative to project root, so local/server can share the same file structure.

```bash
cp .env.example .env
```

Important keys:
- `DATA_BASE_DIR`: patient folders root (`<patient_id>/original.nii.gz`)
- `TEMPLATE_PATH`: reference template NIfTI for registration
- `FREESURFER_HOME`: FreeSurfer install directory
- `NNUNET_BINARY_MODEL_DIR` and `NNUNET_MULTI_MODEL_DIR`: nnUNet trained model folders
- `RADS_MODEL_PATHS`: comma-separated joblib models for final ensemble

# Download nnUNet Weights
Use the helper script to download nnUNet weights into `modelsweights/` (project root):

```bash
uv run python scripts/download_nnunet_weights.py
```

The script supports resumable download (HTTP Range), auto-extracts `zip`/`tar.gz`, auto-flattens nested extracted paths (e.g. `Users/.../binary`), and prints suggested `.env` entries.

# Inference
## Step 1: Preprocess + Segmentation
Script: `inference_segms.py`

```bash
uv run python inference_segms.py
```

Dependencies outside Python:
- FreeSurfer (for `mri_synthstrip`)
- FSL (for `flirt`)

## Step 2: Radiomics + Final Prediction
Script: `inference_norm_rads.py`

```bash
uv run python inference_norm_rads.py
```

Output:
- Each patient folder gets `prediction.csv`.




 





