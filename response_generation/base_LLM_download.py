from huggingface_hub import snapshot_download

# 模型的 Hugging Face 名称或路径
model_name = "THUDM/glm-4-9b-chat"  # 替换为你要下载的模型名称

# 下载模型到本地目录
local_dir = "models/glm-4-9b-chat" # 本地保存路径
snapshot_download(
    repo_id=model_name,  # 模型名称
    local_dir=local_dir,  # 本地保存目录
    local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
    resume_download=True,  # 支持断点续传
    token=None,  # 如果需要访问私有模型，可以传入 Hugging Face 的 token
)

print(f"模型已下载到: {local_dir}")
