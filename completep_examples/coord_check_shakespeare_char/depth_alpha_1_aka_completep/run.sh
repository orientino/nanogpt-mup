for depth in 2 8 32 128
do
    # for lr in 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625 0.001953125 0.0009765625 0.00048828125 0.000244140625 0.0001220703125 0.00006103515625
    for lr in 0.125 0.03125 0.0078125 0.001953125 0.00048828125 0.0001220703125 
    do
        for seed in 1 2 3
        do
            width=256
            head_size=64
            depth_alpha_exp=1.0
            n_heads=$((width / head_size))
            mup_base_depth=2
            mup_depth_multiplier=$(echo "scale=8; $depth/$mup_base_depth" | bc -l)
            out_dir="/project/home/p200535/project/nanogpt-mup/shakespeare_char/depth_alpha1/width${width}_depth${depth}_seed${seed}_lr${lr}"
            python train.py \
                --out_dir=$out_dir \
                --eval_interval=1 \
                --log_interval=1 \
                --eval_iters=1 \
                --eval_only=False \
                --always_save_checkpoint=False \
                --never_save_checkpoint=False \
                --init_from='scratch' \
                --wandb_log=False \
                --csv_log=True \
                --dataset='shakespeare_char' \
                --gradient_accumulation_steps=8 \
                --batch_size=1 \
                --block_size=1024 \
                --n_layer=$depth \
                --n_head=$n_heads \
                --n_embd=$width \
                --dropout=0.0 \
                --bias=True \
                --init_std=0.02 \
                --learning_rate=$lr \
                --max_iters=128 \
                --weight_decay=1e-1 \
                --beta1=0.95 \
                --beta2=0.95 \
                --grad_clip=1.0 \
                --decay_lr=False \
                --mup_enabled=True \
                --mup_width_multiplier=1.0 \
                --mup_input_alpha=1.0 \
                --mup_output_alpha=1.0 \
                --mup_enable_coord_check_logging=True \
                --depth_alpha_enabled=True  \
                --depth_alpha_exp=1.0 \
                --depth_multiplier=$mup_depth_multiplier \
                --seed=$seed \
                --backend='nccl' \
                --device='cuda' \
                --dtype='float32' \
                --compile=False
        done
    done
done
