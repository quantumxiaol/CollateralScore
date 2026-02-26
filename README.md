# CollateralScore

Developed by the Charité Lab for AI in Medicine (CLAIM) research group at Charité University Hospital, Berlin, main developer and person to contact: Dimitrios Rallios (dimitrios.rallios@charite.de)

Goal = Automated Collateral Score Grading based on cerebrovascular radiomics.
Input = Niftis of CT Angiographies of patients with a LVO

Step 1 -> CTA preprocessing and nn-Unet-based Vessel Segmentation.
Step 2 -> Selected Radiomics Extraction and RFC prediction between sufficient (Tan Score 2 and 3) and insufficient (Tan Score 0 and 1)

# Env
```bash
# Clone
git clone https://github.com/YourUsername/BrainIAC_V2.git
cd BrainIAC

# (Optional) Use TUNA mirror in mainland China
uv lock --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
uv sync --extra test --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"

# Or without mirror
# uv sync --extra test

source .venv/bin/activate
```

# Inference
As Pyradiomics words on CPU and nn-UNet on GPU we provide different yamls and requirment.txt for each step.

## Step 1 
Python script for the first step : inference_segms.py

### Environment Creation - First Option
Environment YAML : segmentation__environment.yml

```bash
conda env create -f path/to/segmentation__environment.yml -n this_will_be_the_new_name_of_your_new_env_for_preprocess_and_segmentation
```

### Environment Creation - Second Option
Requierments text : nnunet_requirements.txt

```bash
conda create -n this_will_be_the_new_name_of_your_new_env_for_preprocess_and_segmentation

conda activate this_will_be_the_new_name_of_your_new_env_for_preprocess_and_segmentation

pip install -r pip install -r path/to/nnunet_requirements.txt
```

### Things you need to change in the provided .py file.

Line 14 : freesurfer_home = '/path/to/freesurfer' . In order to create the brain mask for the cropping we need the installation of the freesurfer.

Line 135 : model_path = "/path/to/binary". Add here the location of the binary model. 
---> https://drive.google.com/file/d/1p4TYVPz_QA0CvuX5HueYVUUYj-wgs5cj/view?usp=share_link


Line 247 : model_path = "/path/to/multi". Add here the location of the multilabel model.
---> https://drive.google.com/file/d/1r14hRPjsc_443_TZbsRSK3xm-ZbfFlK6/view?usp=share_link

Line 355 : base_dir = "/path/to/dir". Update with the main directory. The main directory should be have subdirectories with the CTA NIfTI

Line 356 : template_path = "/path/to/template". You can find the template used for this study here : https://github.com/muschellij2/high_res_ct_template/tree/master/template


## Step 2
Python script for the second step : inference_norm_rads.py

### Environment Creation - First Option.
Environment YAML : radiomics_environment.yml

```bash
conda env create -f path/to/radiomics_environment.yml -n this_will_be_the_new_name_of_your_new_env_for_radiomics_and_prediction
```

### Environment Creation - Second Option.
Requierments text :rads_requirements.txt

```bash
conda create -n this_will_be_the_new_name_of_your_new_env_for_radiomics_and_prediction
conda activate
 this_will_be_the_new_name_of_your_new_env_for_radiomics_and_prediction
pip install -r pip install -r path/to/rads_requirements.txt
```

### Things you need to change in the provided .py file.
Lines 11 to 17 : provide the yaml file the extraction of the radiomics /path/to/all_shape.yaml

Lines 259 to 262 : Provide the pathways to the models."/path/to/models"

Lines 296 : base_dir = "/path/to/dir". Update with the main directory. The main directory should be have subdirectories with the CTA NIfTI

The prediction of each patient will be saved as a csv in the folder of the patient.




 









