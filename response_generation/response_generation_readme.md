# Response Generation

This folder contains the dialogue system source code: MemoryBank-based profile retrieval, context-aware HY item selection, response generation prompts, and HY scoring integration.

Public release status:
- Real patient profiles, real dialogue test files, generated evaluation conversations, and human-evaluation materials have been removed.
- `memories/update_memory.json` is a small synthetic demo memory file.
- `memories/HY_inf.json`, `memories/HY_origin.json`, and `memories/updated_file2.json` contain public HY descriptions/keywords used by item selection.
- `evaluation/Relevance_Matrix.json` is retained because it is needed by the CA-HYIS flow.
- HY scoring model weights are not committed. Set `HY_SCORING_MODEL_PATH` or place a separately released model under `../HY_scoring/HY_Scoring/fold5`.
- Set `BASE_LLM_MODEL_PATH` to a local base dialogue model path, or allow Hugging Face to resolve the default model identifier.

Environment:
- Install dependencies with `pip install -r requirements.txt` from this folder, or `pip install -r response_generation/requirements.txt` from the repository root.
- Install `../HY_scoring/requirements.txt` as well if using a separate virtual environment, because the dialogue system imports the HY scoring inference engine.
- For GPU inference, install a CUDA-compatible PyTorch build according to your local CUDA version.

Main entry:
- `HY_my_algorithm_case_study.py`: command-line dialogue system demo.
