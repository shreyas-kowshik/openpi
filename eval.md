# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero

# Run the simulation
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
python examples/libero/main.py

# In another terminal
# Run the server
# This seems to run a server for querying
uv run scripts/serve_policy.py --env LIBERO

Evaluation will dump videos at `data/libero/videos`

To evaluate particular model on particular split:
Serve custom model
```
python scripts/serve_policy.py policy:checkpoint --policy.config pi05_libero --policy.dir gs://openpi-assets/checkpoints/pi05_libero

# Or make libero above
python scripts/serve_policy.py policy:checkpoint --policy.config pi0_libero_low_mem_finetune --policy.dir /data/user_data/skowshik/openpi_cache/libero_custom_lora_ft/checkpoints/pi0_libero_low_mem_finetune/debug-v1/29999/

# Eval LoRA v3
python scripts/serve_policy.py policy:checkpoint --policy.config pi0_libero_low_mem_finetune_v3 --policy.dir /data/user_data/skowshik/openpi_cache/libero_custom_lora_ft_lowmem_v3/checkpoints/pi0_libero_low_mem_finetune_v3/loramem_ft_v3/29999/

# Eval LoRA v1
python scripts/serve_policy.py policy:checkpoint --policy.config pi0_libero_low_mem_finetune_v1 --policy.dir /data/user_data/skowshik/openpi_cache/libero_custom_lora_ft_lowmem_v1/checkpoints/pi0_libero_low_mem_finetune_v1/loramem_ft_v1/18000/

# Eval LoRA v2
python scripts/serve_policy.py policy:checkpoint --policy.config pi0_libero_low_mem_finetune_v2 --policy.dir /data/user_data/skowshik/openpi_cache/libero_custom_lora_ft_lowmem_v2/checkpoints/pi0_libero_low_mem_finetune_v2/loramem_ft_v2/18000/

# Eval LoRA v4
python scripts/serve_policy.py policy:checkpoint --policy.config pi0_libero_low_mem_finetune_v4 --policy.dir /data/user_data/skowshik/openpi_cache/libero_custom_lora_ft_lowmem_v4/checkpoints/pi0_libero_low_mem_finetune_v4/loramem_ft_v4/
```

Evaluate on libero10
```
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
python examples/libero/main.py --args.task_suite_name libero_spatial | tee logs/libero_spatial_full_ft.log

python examples/libero/main.py --args.task_suite_name libero_10 --args.video_out_path data/libero/videos_v3/ 2>&1 | tee logs/libero_10_lora_v3.log

python examples/libero/main.py --args.task_suite_name libero_10 --args.video_out_path data/libero/videos_v1/ 2>&1 | tee logs/libero_10_lora_v1.log

python examples/libero/main.py --args.task_suite_name libero_10 --args.video_out_path data/libero/videos_v2/ 2>&1 | tee logs/libero_10_lora_v2.log

python examples/libero/main.py --args.task_suite_name libero_10 --args.video_out_path data/libero/videos_v4/ 2>&1 | tee logs/libero_10_lora_v4.log
```

# Kill all processes in case server is still running from a previous launch

List all server processes manually
`htop`

Find PGID of any process PID above
```
ps -o pid,ppid,pgid,cmd -p 2996517
```

Get PGID and kill all processes with given PGID
```
kill -- -2996402
```

