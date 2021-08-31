nohup python mT5-finetune/finetune.py \
    --model_name_or_path mt5-base \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang si \
    --source_prefix "en-si: " \
    --train_file XX \
    --validation_file XX \
    --output_dir output/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --save_steps=1000 \
    --report_to="wandb" \
    --predict_with_generate &

# Prev task name 'translate English to Sinhala'