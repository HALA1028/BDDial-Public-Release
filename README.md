# BDdial Public Release

This repository contains the public source code and synthetic resources for the paper:

**A Personalized Dialogue-Based System for Bipolar Disorder Assessment Using Multi-Stage Data Augmentation**

The system has three main components:

1. `data_augmentation/`: multi-stage construction of the BDdial dialogue dataset.
2. `HY_scoring/`: training and inference code for the HAMD/YMRS (HY) scoring model.
3. `response_generation/`: the personalized bipolar-disorder dialogue system with MemoryBank, context-aware HY item selection, response generation, and HY scoring integration.

This repository is intended as a minimal reproducible research release. It does not include private clinical data, real patient profiles, or experiment logs.

## Public Release Scope

Included:

- Source code for BDdial construction.
- Generated BDdial dialogue files in `data_augmentation/stage3_fomal_distinct_filter/`.
- Synthetic HY scoring data in `HY_scoring/HY_score_without_real_data.csv`.
- Data augmentation prompts in `data_augmentation/prompt/`.
- HAMD/YMRS scoring rules in `data_augmentation/HY_rules/`.
- HY item descriptions and keywords in `response_generation/memories/`.
- Response-generation prompts in `response_generation/system_prompt/`.
- The relevance matrix used by context-aware HY item selection in `response_generation/evaluation/Relevance_Matrix.json`.

Not included:

- The original NCKU BD real clinical dataset.
- Real patient dialogues, real patient profiles, memory indexes, evaluation conversations, human-evaluation files, cache, logs, or debug notebooks.
- The original full PsyQA dataset.
- HY scoring model weights or checkpoints. Users should train the model themselves.

## Repository Structure

```text
.
|-- data_augmentation/
|   |-- prompt/
|   |-- HY_rules/
|   |-- stage3_fomal_distinct_filter/
|   |-- requirements.txt
|   |-- stage1_Problem_localization.py
|   |-- stage2_dialogization.py
|   |-- stage3_scoring.py
|   |-- Generate_Data_Filtering.py
|   `-- data_augmentation_readme.md
|-- HY_scoring/
|   |-- HY_score_without_real_data.csv
|   |-- requirements.txt
|   |-- 5_fold_valid_train_regression_divideHY.py
|   |-- inf_regression.py
|   |-- HY_Scoring/
|   |   |-- HY_Scoring.py
|   |   `-- label_mappings.json
|   `-- HY_scoring_readme.md
|-- response_generation/
|   |-- MEMORY_BANK/
|   |-- system_prompt/
|   |-- memories/
|   |-- evaluation/Relevance_Matrix.json
|   |-- requirements.txt
|   |-- HY_my_algorithm_case_study.py
|   `-- response_generation_readme.md
|-- CITATION.cff
|-- DATASET_CARD.md
|-- DATA_LICENSE.md
|-- LICENSE
`-- THIRD_PARTY_NOTICES.md
```

## Environment Setup

### Data Augmentation

The data augmentation environment should be installed from:

```bash
pip install -r data_augmentation/requirements.txt
```

The current data augmentation scripts use the OpenAI API for LLM-based generation and scoring. Set:

```bash
export OPENAI_API_KEY="your_api_key"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

### HY Scoring

The HY scoring training and inference environment should be installed from:

```bash
pip install -r HY_scoring/requirements.txt
```

For GPU training, install a CUDA-compatible PyTorch build according to your local CUDA version.

### Response Generation

The response-generation environment should be installed from:

```bash
pip install -r response_generation/requirements.txt
```

For GPU inference, install a CUDA-compatible PyTorch build according to your local CUDA version. The dialogue system also imports the HY scoring inference engine, so install `HY_scoring/requirements.txt` first if using separate virtual environments.

The dialogue system can use these optional environment variables:

```bash
export BASE_LLM_MODEL_PATH="path_or_huggingface_id_for_base_dialogue_model"
export HY_SCORING_MODEL_PATH="path_to_trained_hy_scoring_model"
```

If `HY_SCORING_MODEL_PATH` is not set, the code expects the HY scoring model under:

```text
HY_scoring/HY_Scoring/fold5/
```

Model weights are not included in this repository. Users should train the HY scoring model first.

## Obtaining PsyQA and Preparing Input

BDdial is constructed from PsyQA-style mental-health QA data. The full original PsyQA dataset is not redistributed in this repository.

Users should obtain PsyQA from the original public sources and follow their usage terms:

- Official GitHub: [thu-coai/PsyQA](https://github.com/thu-coai/PsyQA)
- Paper page: [PsyQA: A Chinese Dataset for Generating Long Counseling Text for Mental Health Support](https://aclanthology.org/2021.findings-acl.130/)
- Some Hugging Face mirrors may also provide processed PsyQA variants. Users must verify the license and redistribution terms before use.

After obtaining authorized PsyQA data, prepare the stage-1 input as text files under:

```text
data_augmentation/stage1_formal_v2/
```

Each file should contain one PsyQA-derived single-turn example in this format:

```text
Single-turn dialogue: {
topic: <keywords>

P: <question text> <detailed description>

}
```

The data augmentation scripts will write intermediate results to:

```text
data_augmentation/stage1_formal_output_v2/
data_augmentation/stage2_formal_v2_selected/
data_augmentation/stage2_formal_output_v2_selected/
data_augmentation/stage3_formal_output_v2_selected/
```

The public generated BDdial files currently included in this repository are stored in:

```text
data_augmentation/stage3_fomal_distinct_filter/
```

## Running Data Augmentation

After preparing authorized PsyQA inputs:

```bash
python data_augmentation/stage1_Problem_localization.py
python data_augmentation/stage2_dialogization.py
python data_augmentation/stage3_scoring.py
```

Optional filtering and balancing scripts:

```bash
python data_augmentation/Generate_Data_Filtering.py
python data_augmentation/single_turn_completed.py
python data_augmentation/singlr_turn_completed_stage2.py
```

`Generate_Data_Filtering.py` can compare generated turns against private reference turns. Real clinical reference turns are not included. If authorized, provide them locally through:

```bash
export REF_TURNS_DIR="path_to_private_reference_turns"
```

## Training HY Scoring

The public synthetic HY-turn file is:

```text
HY_scoring/HY_score_without_real_data.csv
```

This public file contains 12,193 synthetic HY assessment-turn records. In the full experimental setting described in the paper, these synthetic records were combined with 507 non-public real clinical records to form a 12,700-record balanced HY scoring resource. The 507 real clinical records are not redistributed in this repository.

Train the HY scoring model with:

```bash
cd HY_scoring
python 5_fold_valid_train_regression_divideHY.py \
  --primary_data_file HY_score_without_real_data.csv \
  --item_label_map_file HY_Scoring/label_mappings.json \
  --model_name_or_path hfl/chinese-macbert-large \
  --output_dir output_regression_macbert
```

The original paper uses private real clinical data for 5-fold evaluation. Those private fold files are not included. Users with authorized private data may place them in `HY_scoring/private_real_data/` and pass:

```bash
--crossval_dir private_real_data
```

After training, point the dialogue system to the trained model:

```bash
export HY_SCORING_MODEL_PATH="HY_scoring/output_regression_macbert/fold_5"
```

## Running Response Generation

Install the response-generation environment, prepare or train an HY scoring model, and set model paths:

```bash
export BASE_LLM_MODEL_PATH="THUDM/glm-4-9b-chat"
export HY_SCORING_MODEL_PATH="path_to_trained_hy_scoring_model"
```

Then run:

```bash
python response_generation/HY_my_algorithm_case_study.py
```

The demo memory file is:

```text
response_generation/memories/update_memory.json
```

Real patient memory files are not included and should not be committed.

## HAMD / YMRS Keywords and Rules

The HY item descriptions and keywords used by the response-generation system are stored in:

```text
response_generation/memories/HY_inf.json
response_generation/memories/HY_origin.json
response_generation/memories/updated_file2.json
```

The scoring rules used during data augmentation are stored in:

```text
data_augmentation/HY_rules/
```

The valid scoring labels used by the HY scoring model are stored in:

```text
HY_scoring/HY_Scoring/label_mappings.json
```

Some HAMD/YMRS items may be present in the keyword files for item-selection context but excluded from text-only scoring when they require visual assessment.

## License

Source code is released under the MIT License. See `LICENSE`.

Released BDdial dataset resources are released for non-commercial research use under CC BY-NC 4.0. See `DATA_LICENSE.md`.

Third-party components and source-data notices are documented in `THIRD_PARTY_NOTICES.md`.

## Third-Party Components

The memory retrieval utilities in `response_generation/MEMORY_BANK/` are adapted from the MIT-licensed MemoryBank-SiliconFriend project. See `THIRD_PARTY_NOTICES.md` for attribution and license information.

## Privacy and Ethical Use

This repository is for research reproducibility only. It is not a clinical diagnostic tool.

Do not commit or redistribute:

- Real patient data.
- Clinical transcripts.
- Patient profiles or memory files.
- Hospital records.
- Private API keys or tokens.
- Model checkpoints trained on non-public clinical data without proper authorization.
