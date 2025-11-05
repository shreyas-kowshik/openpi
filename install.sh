conda create -n openpi python=3.11.9
GIT_LFS_SKIP_SMUDGE=1 uv sync # --frozen --no-install-project --no-dev --cache-dir "/data/user_data/skowshik/uv_cache"
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . --cache-dir "/data/user_data/skowshik/uv_cache"
