import os
import subprocess
import nibabel as nib
import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import shutil
import SimpleITK as sitk
import pandas as pd
import glob
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


def resolve_torch_device() -> torch.device:
    configured = os.getenv("TORCH_DEVICE")
    if configured:
        return torch.device(configured)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


load_dotenv(PROJECT_ROOT / ".env")

MODELS_ROOT = get_path_from_env("COLLATERALSCORE_MODELS_DIR", "modelsweights")
FREESURFER_HOME = str(get_path_from_env("FREESURFER_HOME", "/path/to/freesurfer"))
NNUNET_BINARY_MODEL_DIR = get_path_from_env("NNUNET_BINARY_MODEL_DIR", MODELS_ROOT / "nnunet" / "binary")
NNUNET_MULTI_MODEL_DIR = get_path_from_env("NNUNET_MULTI_MODEL_DIR", MODELS_ROOT / "nnunet" / "multi")
DEFAULT_BASE_DIR = get_path_from_env("DATA_BASE_DIR", "/path/to/dir")
DEFAULT_TEMPLATE_PATH = get_path_from_env("TEMPLATE_PATH", "/path/to/template")
TORCH_DEVICE = resolve_torch_device()

if FREESURFER_HOME != "/path/to/freesurfer":
    os.environ.setdefault("FREESURFER_HOME", FREESURFER_HOME)
    freesurfer_bin = os.path.join(FREESURFER_HOME, "bin")
    if freesurfer_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + freesurfer_bin

def create_brain_mask(input_path, output_dir):
    """Creates a brain mask using SynthStrip from FreeSurfer."""
    try:
        if FREESURFER_HOME == "/path/to/freesurfer":
            raise ValueError("FREESURFER_HOME is not configured. Set it in .env or your shell.")

        print(f"Creating brain mask for {input_path}")
        output_path = os.path.join(output_dir, os.path.basename(input_path).replace(".nii.gz", "_BET.nii.gz"))
        mask_output_path = os.path.join(output_dir, os.path.basename(input_path).replace(".nii.gz", "_BET_mask.nii.gz"))

        if not os.path.exists(mask_output_path):
            command = (
                f"bash -c 'source {os.path.join(FREESURFER_HOME, 'SetUpFreeSurfer.sh')} "
                f"&& mri_synthstrip -i {input_path} -o {output_path} -m {mask_output_path}'"
            )
            print(f"Running command: {command}")
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            print(result.stdout, result.stderr)
            
            if result.returncode == 0:
                print(f"Skullstripping with Synthstrip completed successfully for {input_path}")
            else:
                raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout, stderr=result.stderr)

        return mask_output_path
    except Exception as e:
        print(f"Error creating brain mask: {e}")
        return None

def crop_image(original_nifti_path, brain_mask_path, output_dir, voxel_count_threshold=5000):
    """Crops the image using the brain mask."""
    try:
        print(f"Cropping image {original_nifti_path}")
        original_nifti_img = nib.load(original_nifti_path)
        brain_binary_img = nib.load(brain_mask_path)


        slice_count = brain_binary_img.shape[2]
        print(f"Total slices in z-axis: {slice_count}")

        first_slice = next((z for z in range(slice_count) if np.sum(brain_binary_img.get_fdata()[:, :, z]) >= voxel_count_threshold), None)
        last_slice = next((z for z in reversed(range(slice_count)) if np.sum(brain_binary_img.get_fdata()[:, :, z]) > 0), None)

        if first_slice is None or last_slice is None:
            raise ValueError("No significant brain tissue found for cropping.")

        print(f"Cropping from slice {first_slice} to {last_slice}")

        cropped_img = original_nifti_img.get_fdata()[:, :, first_slice:last_slice + 1]
        img_cropped = nib.Nifti1Image(cropped_img, original_nifti_img.affine)
        cropped_image_path = os.path.join(output_dir, os.path.basename(original_nifti_path).replace(".nii.gz", "_cropped.nii.gz"))
        nib.save(img_cropped, cropped_image_path)
        print(f"Saved cropped image to {cropped_image_path}")

        return cropped_image_path
    except Exception as e:
        print(f"Error cropping image: {e}")
        return None

def register_image_to_template(cta_cropped, template_path, output_dir, cost_function="mutualinfo"):
    """Registers only the cropped CTA image to the template."""
    try:
        registered_cta_path = os.path.join(output_dir, os.path.basename(cta_cropped).replace(".nii.gz", "_registered.nii.gz"))
        output_matrix = os.path.join(output_dir, "registration_matrix.mat")

        print(f"Registering {cta_cropped} to {template_path}")
        command = (f"flirt -in {cta_cropped} -ref {template_path} "
                   f"-out {registered_cta_path} -omat {output_matrix} "
                   f"-cost {cost_function} -dof 6")
        subprocess.run(command, shell=True, check=True)
        print(f"Registration completed: {registered_cta_path}")

        return registered_cta_path
    except subprocess.CalledProcessError as e:
        print(f"Error in registration: {e}")
        return None


def run_inference_binary(input_cta):
    """Runs binary segmentation on the registered cropped CTA."""
    try:
        # Ensure the CTA file exists
        print(f"CTA path: {input_cta}")
        if not os.path.exists(input_cta):
            raise FileNotFoundError(f"CTA file not found at {input_cta}")

        # Check the file loading
        try:
            image = sitk.ReadImage(input_cta)
            print("Image loaded successfully.")
        except Exception as e:
            print(f"Error loading image: {e}")
            raise

        # Extract patient ID from input path
        patient_folder = os.path.dirname(input_cta)  # e.g., "/home/dimitrios/example_publication/001/"
        patient_id = os.path.basename(patient_folder)  # e.g., "001"

        # Define segmentation model folder
        segmentation_folder = os.path.join(patient_folder, "segmentation_binary")
        os.makedirs(segmentation_folder, exist_ok=True)

        # Define new file path inside segmentation folder
        renamed_cta = os.path.join(segmentation_folder, f"BRAIN_{patient_id}_0000.nii.gz")

        # Copy the original file to the segmentation_models folder
        shutil.copy(input_cta, renamed_cta)

        print(f"Copied and renamed {input_cta} to {renamed_cta}")

        # Initialize the predictor
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=TORCH_DEVICE,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )

        # Load the trained model
        model_path = str(NNUNET_BINARY_MODEL_DIR)
        
        predictor.initialize_from_trained_model_folder(
            model_path,
            use_folds=(0, 1, 2, 3, 4),  #  Assuming 5-fold cross-validation
            checkpoint_name='checkpoint_best.pth',
        )
        print(f"CTA path: {renamed_cta}")
        # Run segmentation in `segmentation_models`

        predictor.predict_from_files(
            segmentation_folder,  # Provide a list with the file
            segmentation_folder,  # Save outputs in the same folder
            overwrite=False,
            folder_with_segs_from_prev_stage=None
        )

        print(f"Inference completed. Output saved in {segmentation_folder}")


        # Split the segmentation into left and right parts
        segmentation_file = os.path.join(segmentation_folder, f"BRAIN_{patient_id}.nii.gz")
        
        # Load the segmentation NIfTI file
        nii_seg = nib.load(segmentation_file)
        seg_data = nii_seg.get_fdata()

        # Determine the midline (assuming the midline is the midpoint of the x-dimension)
        midline = seg_data.shape[0] // 2

        # Create masks for the left and right sides (patient's perspective)
        right_mask = np.zeros_like(seg_data, dtype=bool)  # Display right -> Patient left
        left_mask = np.zeros_like(seg_data, dtype=bool)   # Display left -> Patient right

        # Apply masks
        right_mask[:midline, :, :] = True
        left_mask[midline:, :, :] = True

        # Create new segmentations by applying the masks
        right_seg = np.where(right_mask, seg_data, 0)  # Patient left
        left_seg = np.where(left_mask, seg_data, 0)    # Patient right

        # Create new NIfTI images
        right_nii = nib.Nifti1Image(right_seg, affine=nii_seg.affine, header=nii_seg.header)
        left_nii = nib.Nifti1Image(left_seg, affine=nii_seg.affine, header=nii_seg.header)

        # Define output paths
        output_right_name = f"right_binary_segmentation.nii.gz"
        output_left_name = f"left_binary_segmentation.nii.gz"
        
        output_right_path = os.path.join(segmentation_folder, output_right_name)  # Patient left
        output_left_path = os.path.join(segmentation_folder, output_left_name)    # Patient right

        # Save the new segmentations
        nib.save(right_nii, output_right_path)
        nib.save(left_nii, output_left_path)
        print(f"Left and right segmentations saved for patient {patient_id}")

       
        #extract_radiomics_and_save(output_right_path, "/home/dimitrios/yaml_configs/right_params.yaml", patient_folder) #
        #extract_radiomics_and_save(output_left_path, "/home/dimitrios/yaml_configs/left_params.yaml", patient_folder) #

       
    except Exception as e:
        print(f"Error in segmentation: {e}")


def run_inference_mutli(input_cta):
    """Runs binary segmentation on the registered cropped CTA and ensures specific segmentations have label 1."""
    try:
        # Ensure the CTA file exists
        print(f"CTA path: {input_cta}")
        if not os.path.exists(input_cta):
            raise FileNotFoundError(f"CTA file not found at {input_cta}")

        # Check the file loading
        try:
            image = sitk.ReadImage(input_cta)
            print("Image loaded successfully.")
        except Exception as e:
            print(f"Error loading image: {e}")
            raise

        # Extract patient ID from input path
        patient_folder = os.path.dirname(input_cta)  # e.g., "/home/dimitrios/example_publication/001/"
        patient_id = os.path.basename(patient_folder)  # e.g., "001"

        # Define segmentation model folder
        segmentation_folder = os.path.join(patient_folder, "segmentation_multi")
        os.makedirs(segmentation_folder, exist_ok=True)

        # Define new file path inside segmentation folder
        renamed_cta = os.path.join(segmentation_folder, f"BRAIN_{patient_id}_0000.nii.gz")

        # Copy the original file to the segmentation_models folder
        shutil.copy(input_cta, renamed_cta)

        print(f"Copied and renamed {input_cta} to {renamed_cta}")

        # Initialize the predictor
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=True,
            device=TORCH_DEVICE,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )

        # Load the trained model
        model_path = str(NNUNET_MULTI_MODEL_DIR)
        
        predictor.initialize_from_trained_model_folder(
            model_path,
            use_folds=(0, 1, 2, 3, 4),  # Assuming 5-fold cross-validation
            checkpoint_name='checkpoint_best.pth',
        )
        print(f"CTA path: {renamed_cta}")

        # Run segmentation in `segmentation_models`
        predictor.predict_from_files(
            segmentation_folder,  # Provide a list with the file
            segmentation_folder,  # Save outputs in the same folder
            overwrite=False,
            folder_with_segs_from_prev_stage=None
        )

        print(f"Inference completed. Output saved in {segmentation_folder}")

        # After inference, split the multi-label segmentation into specific label groups
        segmentation_file = os.path.join(segmentation_folder, f"BRAIN_{patient_id}.nii.gz")

        # Load the segmentation NIfTI file
        segmentation_nii = nib.load(segmentation_file)
        segmentation = segmentation_nii.get_fdata()

        # Get the directory of the original file
        dir_name = os.path.dirname(segmentation_file)

        # Define label groups and their corresponding output filenames
        label_groups = {
            'rest_segm_reg.nii.gz': [1],
            'ICA_segm_r.nii.gz': [2],
            'MCA_segm_r.nii.gz': [3],
            'MCA_segm_l.nii.gz': [5],
            'ICA_segm_l.nii.gz': [4],
        }

        # Create and save segmentation for each label group
        for output_filename, labels in label_groups.items():
            # Create a mask for the current label group
            label_mask = np.isin(segmentation, labels).astype(np.uint8)
            
            # Preserve original label values
            for label in labels:
                label_mask[segmentation == label] = label

            # Ensure the specified segmentations have only label 1
            if output_filename in [
                "right_binary_segmentation.nii.gz",
                "left_binary_segmentation.nii.gz",
                "rest_segm_reg.nii.gz",
                "MCA_segm_l.nii.gz",
                "MCA_segm_r.nii.gz",
                "ICA_segm_l.nii.gz",
                "ICA_segm_r.nii.gz"
            ]:
                label_mask[label_mask > 0] = 1  # Convert all nonzero values to 1

            # Create a new Nifti image for the label mask
            label_mask_nii = nib.Nifti1Image(label_mask, segmentation_nii.affine, segmentation_nii.header)

            # Save the label mask segmentation with the specified filename
            label_segmentation_path = os.path.join(dir_name, output_filename)
            nib.save(label_mask_nii, label_segmentation_path)

            # Print feedback
            print(f"Saved {output_filename} for {segmentation_file} with labels converted to 1 where needed.")

    except Exception as e:
        print(f"Error in segmentation: {e}")

  

def process_cta(cta_path, output_dir, template_path):
    """Runs the full pipeline only if necessary. If the registered image exists, go straight to segmentation."""
    print(f"Processing CTA for {cta_path}")
    os.makedirs(output_dir, exist_ok=True)

    registered_cta = os.path.join(output_dir, "original_cropped_registered.nii.gz")

    # **✅ If the final registered image exists, skip to segmentation**
    if os.path.exists(registered_cta):
        print(f"✅ Registered image already exists: {registered_cta}")
        print("Skipping preprocessing, going straight to segmentation.")
    else:
        # **🔄 Run full pipeline**
        brain_mask_path = create_brain_mask(cta_path, output_dir)
        if not brain_mask_path:
            return None  # Exit if mask creation fails

        cropped_cta = crop_image(cta_path, brain_mask_path, output_dir)
        if not cropped_cta:
            return None  # Exit if cropping fails

        registered_cta = register_image_to_template(cropped_cta, template_path, output_dir)
        if not registered_cta:
            return None  # Exit if registration fails

        print(f"✅ Final registered cropped image saved: {registered_cta}")

    # **🚀 Run segmentation on the final registered image**
    run_inference_binary(registered_cta)
    run_inference_mutli(registered_cta)

    return registered_cta

def main():
    base_dir = str(DEFAULT_BASE_DIR)
    template_path = str(DEFAULT_TEMPLATE_PATH)

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(
            f"DATA_BASE_DIR not found: {base_dir}. Set DATA_BASE_DIR in .env or your shell."
        )
    if not os.path.exists(template_path):
        raise FileNotFoundError(
            f"TEMPLATE_PATH not found: {template_path}. Set TEMPLATE_PATH in .env or your shell."
        )

    # Loop through all folders in base_dir
    for folder in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder)
        
        if os.path.isdir(folder_path):  # Check if it's a directory
            cta_path = os.path.join(folder_path, "original.nii.gz")  
            output_dir = folder_path  # Output in the same folder
            
            if os.path.exists(cta_path):  # Ensure the CTA file exists
                print(f"Processing: {cta_path}")  # Debugging output
                
                # Call your processing function here
                process_cta(cta_path, output_dir, template_path)


if __name__ == "__main__":
    main()
