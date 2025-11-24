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
python scripts/serve_policy.py policy:checkpoint --policy.config pi0_libero_low_mem_finetune_v4 --policy.dir /data/user_data/skowshik/openpi_cache/libero_custom_lora_ft_lowmem_v4/checkpoints/pi0_libero_low_mem_finetune_v4/loramem_ft_v4/29999/
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

python examples/libero/main.py --args.task_suite_name libero_10 --args.video_out_path data/libero/videos_v4_30k/ 2>&1 | tee logs/libero_10_lora_v4_30k.log
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

```
python scripts/serve_policy.py policy:checkpoint --policy.config pi05_libero_lora_vision_full_ft_action_full_ft_siglip --policy.dir /data/user_data/skowshik/openpi_cache/pi05_libero_lora_vision_full_ft_action_full_ft_siglip/checkpoints/pi05_libero_lora_vision_full_ft_action_full_ft_siglip/pi05_libero_lora_vision_full_ft_action_full_ft_siglip-v1/50000/

python scripts/serve_policy.py policy:checkpoint --policy.config pi05_libero_lora_vision_full_ft_action_full_ft_siglip --policy.dir /data/user_data/skowshik/openpi_cache/pi05_libero_lora_vision_full_ft_action_full_ft_siglip/checkpoints/pi05_libero_lora_vision_full_ft_action_full_ft_siglip/pi05_libero_lora_vision_full_ft_action_full_ft_siglip-v1/80000/

python examples/libero/main.py --args.task_suite_name libero_10 --args.video_out_path data/libero/pi05_libero_lora_vision_full_ft_action_full_ft_siglip/ 2>&1 | tee logs/pi05_libero_lora_vision_full_ft_action_full_ft_siglip_50k.log

python examples/libero/main.py --args.task_suite_name libero_10 --args.video_out_path data/libero/pi05_libero_lora_vision_full_ft_action_full_ft_siglip_80k/ 2>&1 | tee logs/pi05_libero_lora_vision_full_ft_action_full_ft_siglip_80k.log
```

```
python scripts/serve_policy.py policy:checkpoint --policy.config pi0_libero_lora_vision_full_ft_action_full_ft_siglip --policy.dir /data/user_data/skowshik/openpi_cache/pi0_libero_lora_vision_full_ft_action_full_ft_siglip/checkpoints/pi0_libero_lora_vision_full_ft_action_full_ft_siglip/pi0_libero_lora_vision_full_ft_action_full_ft_siglip-v1/50000/

python examples/libero/main.py --args.task_suite_name libero_10 --args.video_out_path data/libero/pi0_libero_lora_vision_full_ft_action_full_ft_siglip/ 2>&1 | tee logs/pi0_libero_lora_vision_full_ft_action_full_ft_siglip_50k.log
```

```
python scripts/serve_policy.py policy:checkpoint --policy.config pi05_libero_fullft_vision_lora_action_full_ft_siglip --policy.dir /data/user_data/skowshik/openpi_cache/pi05_libero_fullft_vision_lora_action_full_ft_siglip/checkpoints/pi05_libero_fullft_vision_lora_action_full_ft_siglip/pi05_libero_fullft_vision_lora_action_full_ft_siglip-v1_fresh/20000/

python examples/libero/main.py --args.task_suite_name libero_10 --args.video_out_path data/libero/pi05_libero_fullft_vision_lora_action_full_ft_siglip_20k/ 2>&1 | tee logs/pi05_libero_fullft_vision_lora_action_full_ft_siglip_20k.log
```


# Libero one task evals
```
mkdir -p /data/user_data/skowshik/tmp_jax
export TMPDIR=/data/user_data/skowshik/tmp_jax
export TEMP=$TMPDIR
export TMP=$TMPDIR

python scripts/serve_policy.py policy:checkpoint --policy.config pi0_libero_lora_moka_pots_task_ep29 --policy.dir /data/user_data/skowshik/openpi_cache/pi0_libero_lora_moka_pots_task_ep29/checkpoints/pi0_libero_lora_moka_pots_task_ep29/pi0_libero_lora_moka_pots_task_ep29-v1/10000/

python examples/libero/main.py --args.task_suite_name libero_10 --args.task_name "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove" --args.video_out_path data/libero/pi0_libero_lora_moka_pots_task_ep29_10k/ 2>&1 | tee logs/pi0_libero_lora_moka_pots_task_ep29_10k.log

python examples/libero/main.py --args.task_suite_name libero_10 --args.task_name "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove" --args.video_out_path data/libero/pi0_libero_lora_moka_pots_task_ep29_20k/ 2>&1 | tee logs/pi0_libero_lora_moka_pots_task_ep29_20k.log
```

Full FT
```
python scripts/serve_policy.py policy:checkpoint --policy.config pi0_libero_fullft_moka_pots_task_ep29 --policy.dir /data/user_data/skowshik/openpi_cache/pi0_libero_fullft_moka_pots_task_ep29/checkpoints/pi0_libero_fullft_moka_pots_task_ep29/pi0_libero_fullft_moka_pots_task_ep29-v1/10000/

python examples/libero/main.py --args.task_suite_name libero_10 --args.task_name "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove" --args.video_out_path data/libero/pi0_libero_fullft_moka_pots_task_ep29_10k/ 2>&1 | tee logs/pi0_libero_fullft_moka_pots_task_ep29_10k.log
```

Debug
```
python scripts/serve_policy.py policy:checkpoint --policy.config pi05_libero_vision_init_libero_lora_vision_fullft_action_onetask_ep29 --policy.dir /data/user_data/skowshik/openpi_cache/merged_checkpoint/

python examples/libero/main.py --args.task_suite_name libero_10 --args.task_name "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove" --args.video_out_path data/libero/debug/ 2>&1 | tee logs/debug.log
```

```
python scripts/serve_policy.py policy:checkpoint --policy.config pi05_libero_vision_init_libero_lora_vision_fullft_action_onetask_ep29 --policy.dir /data/user_data/skowshik/openpi_cache/pi05_libero_vision_init_libero_lora_vision_fullft_action_onetask_ep29/pi05_libero_vision_init_libero_lora_vision_fullft_action_onetask_ep29-v1/2500/

python examples/libero/main.py --args.task_suite_name libero_10 --args.task_name "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove" --args.video_out_path data/libero/pi05_libero_vision_init_libero_lora_vision_fullft_action_onetask_ep29_2.5k/ 2>&1 | tee logs/pi05_libero_vision_init_libero_lora_vision_fullft_action_onetask_ep29_2.5k.log
```

```
python scripts/serve_policy.py policy:checkpoint --policy.config pi0_libero_lora_book_compartment_task_ep41 --policy.dir /data/user_data/skowshik/openpi_cache/pi0_libero_lora_book_compartment_task_ep41/pi0_libero_lora_book_compartment_task_ep41-v1/20000/

python examples/libero/main.py --args.task_suite_name libero_10 --args.task_name "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy" --args.video_out_path data/libero/pi0_libero_lora_book_compartment_task_ep41.20k/ 2>&1 | tee logs/pi0_libero_lora_book_compartment_task_ep41.20k.log
```

```
python scripts/serve_policy.py policy:checkpoint --policy.config pi05_libero_vision_lora_fullft_action_book_compartment_task_ep41_init_vision_pi05_libero --policy.dir /data/user_data/skowshik/openpi_cache/pi05_libero_vision_lora_fullft_action_book_compartment_task_ep41_init_vision_pi05_libero/pi05_libero_vision_lora_fullft_action_book_compartment_task_ep41_init_vision_pi05_libero-v1/20000/

python examples/libero/main.py --args.task_suite_name libero_10 --args.task_name "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy" --args.video_out_path data/libero/pi05_libero_vision_lora_fullft_action_book_compartment_task_ep41_init_vision_pi05_libero.20k/ 2>&1 | tee logs/pi05_libero_vision_lora_fullft_action_book_compartment_task_ep41_init_vision_pi05_libero.20k.log
```

```
python scripts/serve_policy.py policy:checkpoint --policy.config pi05_libero_lora_vision_fullft_action_init_vision_v1 --policy.dir /data/user_data/skowshik/openpi_cache/pi05_libero_lora_vision_fullft_action_init_vision_v1/pi05_libero_lora_vision_fullft_action_init_vision_v1-v1/1000

python examples/libero/main.py --args.task_suite_name libero_10 --args.video_out_path data/libero/pi05_libero_lora_vision_fullft_action_init_vision_v1.1k/ 2>&1 | tee logs/pi05_libero_lora_vision_fullft_action_init_vision_v1.1k.log
```

```
python scripts/serve_policy.py policy:checkpoint --policy.config pi05_libero_lora_vision_fullft_action_v1 --policy.dir /data/user_data/skowshik/openpi_cache/pi05_libero_lora_vision_fullft_action_v1/pi05_libero_lora_vision_fullft_action_v1-v1/3000

python examples/libero/main.py --args.task_suite_name libero_10 --args.video_out_path data/libero/pi05_libero_lora_vision_fullft_action_v1.3k/ 2>&1 | tee logs/pi05_libero_lora_vision_fullft_action_v1.3k.log
```