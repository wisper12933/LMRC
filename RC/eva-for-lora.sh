source activate your_env
python eva-for-lora.py \
    --base_model './models/your model' \
    --lora_weight './checkpoints/your checkpoint' \
    --test_path 'your path to the test file' \
    --out_path 'your output path' \
    --load_8bit 