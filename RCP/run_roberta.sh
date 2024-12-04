source activate your_env
python train.py --data_dir ./dataset/docred \
--transformer_type roberta \
--model_name_or_path ./models/roberta-large \
--load_path ./checkpoints/your checkpoint\
--train_file train_annotated.json \
--dev_file dev.json \
--test_file test.json \
--train_batch_size 4 \
--test_batch_size 8 \
--gradient_accumulation_steps 1 \
--evaluation_steps 200 \
--learning_rate 3e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 30.0 \
--seed 66 \
--num_class 2
