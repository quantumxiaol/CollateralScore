import os
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def load_dotenv(dotenv_path: Path) -> None:
    """Load simple KEY=VALUE pairs from .env without extra dependencies."""
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_path_from_env(env_name: str, default: Path | str) -> Path:
    value = os.getenv(env_name)
    path = Path(value).expanduser() if value else Path(default).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_model_paths() -> list[str]:
    raw = os.getenv("RADS_MODEL_PATHS", "")
    if raw.strip():
        resolved: list[str] = []
        for entry in raw.split(","):
            value = entry.strip()
            if not value:
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            resolved.append(str(path))
        return resolved

    return [
        str(PROJECT_ROOT / "prediction1.joblib"),
        str(PROJECT_ROOT / "prediction2.joblib"),
        str(PROJECT_ROOT / "prediction3.joblib"),
        str(PROJECT_ROOT / "prediction4.joblib"),
    ]


load_dotenv(PROJECT_ROOT / ".env")

RADIOMICS_YAML_PATH = get_path_from_env("RADIOMICS_YAML_PATH", PROJECT_ROOT / "all_shape.yaml")
RADIOMICS_MIN_MAX_CSV = get_path_from_env("RADIOMICS_MIN_MAX_CSV", PROJECT_ROOT / "min_max_values.csv")
DEFAULT_BASE_DIR = get_path_from_env("DATA_BASE_DIR", "/path/to/dir")
RADS_MODEL_PATHS = parse_model_paths()

# YAML file mappings for segmentations
yaml_mappings = {
    "right_binary_segmentation.nii.gz": str(RADIOMICS_YAML_PATH),
    "left_binary_segmentation.nii.gz": str(RADIOMICS_YAML_PATH),
    "rest_segm_reg.nii.gz": str(RADIOMICS_YAML_PATH),
    "MCA_segm_l.nii.gz": str(RADIOMICS_YAML_PATH),
    "MCA_segm_r.nii.gz": str(RADIOMICS_YAML_PATH),
    "ICA_segm_l.nii.gz": str(RADIOMICS_YAML_PATH),
    "ICA_segm_r.nii.gz": str(RADIOMICS_YAML_PATH),
}

# List of feature names to keep (currently empty — customize as needed)
required_features = [
   "original_shape_VoxelVolume_rest_segm_reg.nii.gz",
"original_shape_Flatness_right_binary_segmentation.nii.gz",
"original_shape_LeastAxisLength_left_binary_segmentation.nii.gz",
"original_shape_Flatness_left_binary_segmentation.nii.gz",
"original_shape_Maximum3DDiameter_right_binary_segmentation.nii.gz",
"original_shape_Maximum3DDiameter_ICA_segm_l.nii.gz",
"original_shape_MeshVolume_right_binary_segmentation.nii.gz",
"original_shape_SurfaceVolumeRatio_left_binary_segmentation.nii.gz",
"original_shape_SurfaceVolumeRatio_ICA_segm_l.nii.gz",
"original_shape_Elongation_ICA_segm_l.nii.gz",
"original_shape_MeshVolume_ICA_segm_l.nii.gz",
"original_shape_SurfaceVolumeRatio_rest_segm_reg.nii.gz",
"original_shape_Maximum2DDiameterSlice_rest_segm_reg.nii.gz",
"original_shape_MajorAxisLength_rest_segm_reg.nii.gz",
"original_shape_LeastAxisLength_right_binary_segmentation.nii.gz",
"original_shape_Sphericity_left_binary_segmentation.nii.gz",
"original_shape_SurfaceVolumeRatio_MCA_segm_l.nii.gz",
"original_shape_Maximum2DDiameterColumn_ICA_segm_r.nii.gz",
"original_shape_MajorAxisLength_right_binary_segmentation.nii.gz",
"original_shape_MinorAxisLength_ICA_segm_r.nii.gz",
"original_shape_SurfaceVolumeRatio_ICA_segm_r.nii.gz",
"original_shape_LeastAxisLength_rest_segm_reg.nii.gz",
"original_shape_Maximum2DDiameterSlice_ICA_segm_r.nii.gz",
"original_shape_Maximum2DDiameterColumn_MCA_segm_l.nii.gz",
"original_shape_SurfaceVolumeRatio_MCA_segm_r.nii.gz",
"original_shape_Maximum3DDiameter_MCA_segm_l.nii.gz",
"original_shape_Maximum3DDiameter_left_binary_segmentation.nii.gz",
"original_shape_Maximum3DDiameter_MCA_segm_r.nii.gz",
"original_shape_SurfaceVolumeRatio_right_binary_segmentation.nii.gz",
"original_shape_MajorAxisLength_ICA_segm_r.nii.gz",
"original_shape_Maximum2DDiameterSlice_right_binary_segmentation.nii.gz",
"original_shape_Maximum2DDiameterRow_ICA_segm_l.nii.gz"
]

#min_max_csv = "/home/dimitrios/normalisation_problem/min_max_values.csv"

#def normalize_image(input_image_path, output_image_path):
#    """
#    Normalizes an image using min-max normalization to [0, 1].
#    """
#    image = sitk.ReadImage(input_image_path)
#    array = sitk.GetArrayFromImage(image).astype(np.float32)
#
#    min_val = np.min(array)
#    max_val = np.max(array)
#
#    if max_val > min_val:
#        normalized_array = (array - min_val) / (max_val - min_val)
#    else:
#        normalized_array = np.zeros_like(array)
#
#    normalized_image = sitk.GetImageFromArray(normalized_array)
#    normalized_image.CopyInformation(image)
#    sitk.WriteImage(normalized_image, output_image_path)
#    print(f" Normalized image saved to {output_image_path}")


def extract_radiomics(image_path, segmentation_path, yaml_path):
    """
    Extracts radiomics features from the given image using the specified segmentation and YAML config.
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if not os.path.exists(segmentation_path):
            raise FileNotFoundError(f"Segmentation file not found: {segmentation_path}")

        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML configuration not found: {yaml_path}")

        print(f"📊 Extracting radiomics from {image_path} using segmentation {segmentation_path}")

        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(segmentation_path)
        extractor = featureextractor.RadiomicsFeatureExtractor(yaml_path)
        features = extractor.execute(image, mask)

        return features

    except Exception as e:
        print(f"❌ Error extracting radiomics from {image_path}: {e}")
        return None

def save_final_csv(headers, values, output_csv):
    df = pd.DataFrame([values], columns=headers)
    df.to_csv(output_csv, index=False)
    print(f"✅ Radiomics CSV updated at {output_csv}")

def normalize_radiomics(selected_radiomics_file, min_max_csv, output_file):
    selected_radiomics = pd.read_csv(selected_radiomics_file)
    print(f"✅ Loaded selected radiomics features from {selected_radiomics_file}.\n")

    min_max_values = pd.read_csv(min_max_csv)
    print(f"✅ Loaded min and max values from {min_max_csv}.\n")

    min_max_dict = dict(zip(min_max_values['Feature'], zip(min_max_values['Min'], min_max_values['Max'])))
    print(f"📐 Min-Max dictionary created with {len(min_max_dict)} entries.\n")

    normalized_radiomics = selected_radiomics.copy()

    for feature in selected_radiomics.columns:
        if feature in min_max_dict:
            min_val, max_val = min_max_dict[feature]
            raw_vals = selected_radiomics[feature]

            if pd.isna(min_val) or pd.isna(max_val):
                print(f"⚠️ Skipping {feature} due to NaN in min/max.\n")
                continue

            if max_val > min_val:
                normalized_vals = (raw_vals - min_val) / (max_val - min_val)
                normalized_vals = normalized_vals.clip(lower=0, upper=1)
                normalized_radiomics[feature] = normalized_vals

                print(f"🔍 Normalized {feature}")
            else:
                print(f"⚠️ Skipping normalization for {feature} due to zero range.\n")
        else:
            print(f"⏩ Feature {feature} not in min-max dictionary, skipping.\n")

    normalized_radiomics.to_csv(output_file, index=False)
    print(f"\n✅ Normalized radiomics saved to {output_file}\n")

def process_patient(patient_folder):
    patient_id = os.path.basename(patient_folder)
    csv_path = os.path.join(patient_folder, "selected_radiomics_features.csv")
    original_image = os.path.join(patient_folder, "original_cropped_registered.nii.gz")
    #norm_image = os.path.join(patient_folder, "normalized_original.nii.gz")

    if not os.path.exists(original_image):
        print(f"⚠️ Skipping {patient_id}: Original image not found.")
        return

    # Step 1: Normalize original image
    #normalize_image(original_image, norm_image)

    # Step 2: Define segmentation directories
    segmentation_binary = os.path.join(patient_folder, "segmentation_binary")
    segmentation_multi = os.path.join(patient_folder, "segmentation_multi")

    # Step 3: List all expected segmentation files
    segmentations = [
        os.path.join(segmentation_binary, "right_binary_segmentation.nii.gz"),
        os.path.join(segmentation_binary, "left_binary_segmentation.nii.gz"),
        os.path.join(segmentation_multi, "rest_segm_reg.nii.gz"),
        os.path.join(segmentation_multi, "MCA_segm_l.nii.gz"),
        os.path.join(segmentation_multi, "MCA_segm_r.nii.gz"),
        os.path.join(segmentation_multi, "ICA_segm_l.nii.gz"),
        os.path.join(segmentation_multi, "ICA_segm_r.nii.gz")
    ]

    feature_dict = {}

    for seg_path in segmentations:
        seg_file = os.path.basename(seg_path)

        if not os.path.exists(seg_path):
            print(f"⚠️ Segmentation not found: {seg_path}")
            continue

        if seg_file not in yaml_mappings:
            print(f"⚠️ No YAML mapping for {seg_file}")
            continue

        yaml_path = yaml_mappings[seg_file]
        features = extract_radiomics(original_image, seg_path, yaml_path)

        if features:
            for feat, value in features.items():
                feature_name = f"{feat}_{seg_file}"
                feature_dict[feature_name] = value

    # Step 4: Filter features if any filtering is needed
    filtered_features = {
        k: v for k, v in feature_dict.items()
        if not required_features or any(req in k for req in required_features)
    }

    # Step 5: Merge with existing CSV or create new
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        existing_features = existing_df.columns.tolist()
        existing_values = existing_df.iloc[0].tolist()
    else:
        existing_df = None
        existing_features = []
        existing_values = []

    for feature, value in filtered_features.items():
        if feature not in existing_features:
            existing_features.append(feature)
            existing_values.append(value)

    existing_features.insert(0, 'patient_ID')
    existing_values.insert(0, patient_id)

    min_max_csv = str(RADIOMICS_MIN_MAX_CSV)
    norm_csv_path = os.path.join(patient_folder, "normalized_selected_radiomics.csv")
    # Step 6: Save to CSV and normalize features
    save_final_csv(existing_features, existing_values, csv_path)
    normalize_radiomics(csv_path, min_max_csv, norm_csv_path)

    # Step 7: Keep only required features (in specified order) + patient_ID
    if required_features:
        norm_df = pd.read_csv(norm_csv_path)

        final_columns = ['patient_ID'] + [f for f in required_features if f in norm_df.columns]
        filtered_df = norm_df[final_columns]

        filtered_df.to_csv(norm_csv_path, index=False)
    predict_patient_outcome(norm_csv_path)


def predict_patient_outcome(patient_csv_path):
    # Step 1: Load patient data
    print(f"🔍 Loading data from: {patient_csv_path}")
    test_df = pd.read_csv(patient_csv_path)
    patient_folder = os.path.dirname(patient_csv_path)

    if 'patient_ID' not in test_df.columns:
        print("❌ 'patient_ID' column is missing.")
        return
    
    # Optional: provide dummy label for compatibility
    label_column = 'cgsc_cta_abl_c'
    test_df[label_column] = 0  # dummy label for shape matching
    exclude_columns = ['patient_ID', 'mrs', 'sympt_side', label_column]
    X_test = test_df.drop(columns=exclude_columns, errors='ignore').values
    patient_id = test_df['patient_ID'].iloc[0]

    # Step 2: Load models
    model_paths = RADS_MODEL_PATHS

    print("🤖 Running model ensemble predictions...")
    predictions = []
    for path in model_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        model = joblib.load(path)
        preds = model.predict_proba(X_test)[:, 1]
        predictions.append(preds)

    # Step 3: Soft voting
    y_pred_proba = np.mean(predictions, axis=0)[0]  # Only one patient

    # Step 4: Save prediction
    output_path = os.path.join(patient_folder, "prediction.csv")
    prediction_df = pd.DataFrame({
        'patient_ID': [patient_id],
        'Predicted Probability': [y_pred_proba]
    })
    prediction_df.to_csv(output_path, index=False)

    # Step 5: Print result
    print(f"✅ Prediction for patient {patient_id}: {y_pred_proba:.4f}")
    print(f"📁 Saved to: {output_path}")

def process_all_patients(base_dir):
    for patient_id in sorted(os.listdir(base_dir)):
        patient_folder = os.path.join(base_dir, patient_id)
        if os.path.isdir(patient_folder):  # Ensure it's a folder, not a file
            process_patient(patient_folder)

def main():
    base_dir = str(DEFAULT_BASE_DIR)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(
            f"DATA_BASE_DIR not found: {base_dir}. Set DATA_BASE_DIR in .env or your shell."
        )
    process_all_patients(base_dir)


if __name__ == "__main__":
    main()
