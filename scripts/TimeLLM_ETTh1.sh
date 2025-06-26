model_name=TimeLLM
train_epochs=100
learning_rate=0.001
llama_layers=6
num_process=2 
batch_size=64
d_model=32
d_ff=128
comment='TimeLLM-ETTh1'

export CUDA_VISIBLE_DEVICES=0,1

accelerate launch \
  --num_processes $num_process \
  --mixed_precision fp16 \
  run_main.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_512_96 \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len 512 \
    --label_len 48 \
    --pred_len 96 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --d_model $d_model \
    --d_ff $d_ff \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --llm_layers $llama_layers \
    --train_epochs $train_epochs \
    --model_comment $comment

