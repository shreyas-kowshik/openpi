# Evaluations

## BC Checkpoints
Task: "put both moka pots on the stove"

1. 10 episodes
```
conda activate openpi
python scripts/serve_policy.py policy:checkpoint --policy.config pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep10_bs32_v1_icml --policy.dir /data/user_data/skowshik/openpi_cache/pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep10_bs32_v1_icml/pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep10_bs32_v1_icml-v1/3000

source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
python examples/libero/main.py --args.task_suite_name libero_10 --args.task_name "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove" --args.video_out_path data/libero/pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep10_bs32_v1_icml.3000/ 2>&1 | tee logs/pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep10_bs32_v1_icml.3000.log
```

2. 5 episodes
```
python scripts/serve_policy.py policy:checkpoint --policy.config pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep5_bs32_v1_icml --policy.dir /data/user_data/skowshik/openpi_cache/pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep5_bs32_v1_icml/pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep5_bs32_v1_icml-v1/1500

python examples/libero/main.py --args.task_suite_name libero_10 --args.task_name "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove" --args.video_out_path data/libero/pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep5_bs32_v1_icml.1500/ 2>&1 | tee logs/pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep5_bs32_v1_icml.1500.log
```

3. 3 episodes
```
python scripts/serve_policy.py policy:checkpoint --policy.config pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep3_bs32_v1_icml --policy.dir /data/user_data/skowshik/openpi_cache/pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep3_bs32_v1_icml/pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep3_bs32_v1_icml-v1/4000

python examples/libero/main.py --args.task_suite_name libero_10 --args.task_name "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove" --args.video_out_path data/libero/pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep3_bs32_v1_icml.4000/ 2>&1 | tee logs/pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep3_bs32_v1_icml.4000.log

```

4. 1 episodes
```
python scripts/serve_policy.py policy:checkpoint --policy.config pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep1_bs32_v1_icml --policy.dir /data/user_data/skowshik/openpi_cache/pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep1_bs32_v1_icml/pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep1_bs32_v1_icml-v1/500

python examples/libero/main.py --args.task_suite_name libero_10 --args.task_name "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove" --args.video_out_path data/libero/pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep1_bs32_v1_icml.500/ 2>&1 | tee logs/pi05_libero_lora_vision_lora_action_putbothmokapots_task_ep1_bs32_v1_icml.500.log

```


### 5 episodes vision only init on all libero data, no action ft
```
python scripts/serve_policy.py policy:checkpoint --policy.config pi05_libero_lora_vision_fullft_action_putbothmokapots_task_ep5_bs32_v2_icml_train_only_vision --policy.dir /data/hf_cache/models/pi05_libero_lora_vision_fullft_action_putbothmokapots_task_ep5_bs32_v2_icml_train_only_vision/pi05_libero_lora_vision_fullft_action_putbothmokapots_task_ep5_bs32_v2_icml_train_only_vision-v1/5000/

python examples/libero/main.py --args.task_suite_name libero_10 --args.task_name "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove" --args.video_out_path data/libero/pi05_libero_lora_vision_fullft_action_putbothmokapots_task_ep5_bs32_v2_icml_train_only_vision.5000/ 2>&1 | tee logs/pi05_libero_lora_vision_fullft_action_putbothmokapots_task_ep5_bs32_v2_icml_train_only_vision.5000.log

```