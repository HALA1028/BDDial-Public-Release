# BDDial Dataset Card

## Dataset Summary

BDDial is a synthetic Chinese dialogue dataset for research on dialogue-based bipolar-disorder assessment. It is designed to support reproduction of the data construction pipeline and the HY scoring module described in the associated paper, "A Personalized Dialogue-Based System for Bipolar Disorder Assessment Using Multi-Stage Data Augmentation".

The released resources include two related but structurally different data products:

1. Full generated BDDial dialogue files after HY item identification, dialogue reconstruction, HY scoring, and filtering.
2. HY assessment-turn records used for HY scoring model training.

The original real clinical dataset and real patient profiles are not included.

## Released Files

### Generated Dialogue Files

Path:

```text
data_augmentation/stage3_fomal_distinct_filter/
```

Current size:

```text
1,233 generated dialogue text files
```

These files are the public generated BDDial dialogue files after:

1. HY item identification
2. Dialogue reconstruction
3. HY scoring
4. Filtering

At this stage, the files still retain the generated multi-turn dialogue structure.

### HY Assessment-Turn Data

Path:

```text
HY_scoring/HY_score_without_real_data.csv
```

Current size:

```text
12,193 rows, excluding the header row
```

Fields:

```text
item_id, question, answer, label
```

This file contains HY assessment-turn records, not full multi-turn dialogues. It corresponds to the later public training data used by the HY scoring model. After assessment-turn filtering and optional label balancing, the original dialogue structure is no longer preserved; the data is represented as item-level nurse question, patient answer, and HY score label pairs.

In the full experimental setting described in the paper, the 12,193 public synthetic HY assessment turns were combined with 507 non-public real clinical records to form a 12,700-record balanced HY scoring resource. The 507 real clinical records are private and are not redistributed in this repository. Users can train on the public synthetic records alone or add their own authorized private data locally.

## Data Construction Pipeline

The public data augmentation source code is provided in:

```text
data_augmentation/
```

The construction process is:

1. HY item identification: identify HAMD/YMRS-related content and candidate assessment targets from PsyQA-style inputs.
2. Dialogue reconstruction: convert single-turn mental-health QA examples into multi-turn nurse-patient dialogue format.
3. HY scoring: assign HY item scores according to HAMD/YMRS scoring rules.
4. Filtering: remove unsuitable generated dialogues and keep the public generated dialogue files in `stage3_fomal_distinct_filter/`.
5. Assessment-turn filtering: extract and filter HY-related assessment turns from the generated dialogues.
6. Optional label balancing: augment or balance item-level assessment turns for HY scoring model training.

Important distinction: steps 1-4 produce generated dialogue files that preserve dialogue structure. Steps 5-6 operate at the HY assessment-turn level and produce item-level training examples, not complete dialogues.

## Source Data

BDDial is constructed from authorized PsyQA-style mental-health QA inputs. The full PsyQA dataset is not redistributed in this repository.

Users who want to regenerate BDDial from PsyQA should obtain PsyQA from the original source and follow the original PsyQA usage agreement:

- PsyQA GitHub: https://github.com/thu-coai/PsyQA
- PsyQA paper: https://aclanthology.org/2021.findings-acl.130/

## HAMD / YMRS Resources

HAMD/YMRS scoring rules used during data augmentation are stored in:

```text
data_augmentation/HY_rules/
```

HY item descriptions and keywords used by the response-generation system are stored in:

```text
response_generation/memories/HY_inf.json
response_generation/memories/HY_origin.json
response_generation/memories/updated_file2.json
```

Valid score labels for HY scoring model training are stored in:

```text
HY_scoring/HY_Scoring/label_mappings.json
```

## Intended Uses

BDDial is intended for:

- Research reproducibility for the associated paper.
- Research on bipolar-disorder dialogue systems.
- Training and evaluating HY scoring models on synthetic assessment turns.
- Studying multi-stage data augmentation for mental-health dialogue systems.

## Out-of-Scope Uses

BDDial should not be used as:

- A clinical diagnostic tool.
- A replacement for professional psychiatric assessment.
- A medical decision system without proper clinical validation and regulatory approval.
- A source of real patient behavior or real patient profiles.

## Privacy

The released BDDial resources are synthetic or derived/generated resources for research. The repository does not include:

- Real patient data.
- Real clinical transcripts.
- Real patient memory files or profiles.
- Hospital records.
- Private clinical evaluation files.

Users should not commit private clinical data, private reference turns, memory indexes, or model checkpoints trained on non-public clinical data without proper authorization.

## Limitations

- The generated dialogues may contain artifacts from LLM-based augmentation.
- HY labels may contain noise because they are produced through rule-based and LLM-assisted scoring.
- The assessment-turn CSV does not preserve complete dialogue context.
- Some HAMD/YMRS items that require visual or clinician-observed assessment may be limited or excluded from text-only scoring.
- The dataset is intended for research reproducibility, not clinical deployment.

## License

The released BDDial dataset resources are licensed under CC BY-NC 4.0 for non-commercial research use. See `DATA_LICENSE.md`.

The full PsyQA dataset is not redistributed here. Users must obtain PsyQA from the original provider and comply with its usage agreement.

## Citation

Please cite this repository and the associated paper when available. A `CITATION.cff` file should be added before final public release.
