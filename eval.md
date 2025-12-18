source ../../openpi/examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

python scripts/serve_policy.py policy:checkpoint --policy.config pi05_libero_custom_low_mem --policy.dir gs://openpi-assets/checkpoints/pi05_libero