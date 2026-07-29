# Data Augmentation

This folder contains the source code and prompts for constructing BDDial from authorized PsyQA inputs.

Public release status:
- The full original PsyQA dataset is not included. Users should obtain PsyQA from its original provider and prepare input files locally.
- The generated BDDial dialogue files are kept in `stage3_fomal_distinct_filter/`.
- Real clinical reference turns are not included. If using `Generate_Data_Filtering.py`, provide private reference turns locally through `REF_TURNS_DIR`.

Environment:
- Install the data augmentation dependencies with `pip install -r requirements.txt` from this folder, or `pip install -r data_augmentation/requirements.txt` from the repository root.

Main pipeline:
1. `stage1_Problem_localization.py`: identify potentially related HY items and generate HY-related questions.
2. `stage2_dialogization.py`: reconstruct single-turn PsyQA examples into multi-turn nurse-patient dialogues.
3. `stage3_scoring.py`: score patient answers according to HY scoring rules.
4. `Generate_Data_Filtering.py`: optional assessment-turn consistency filtering using private reference turns.
5. `single_turn_completed.py` and `singlr_turn_completed_stage2.py`: optional label-balancing augmentation.

Important prompts are stored under `prompt/`, and HY scoring rules are stored under `HY_rules/`.
