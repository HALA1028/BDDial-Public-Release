from huggingface_hub import snapshot_download

model_name = "hfl/chinese-macbert-large"  

local_dir = "models/chinese-macbert-large" 
snapshot_download(
    repo_id=model_name,  
    local_dir=local_dir,  
    local_dir_use_symlinks=False,  
    resume_download=True,  
    token=None, 
)

print(f"Model downloaded to: {local_dir}")
