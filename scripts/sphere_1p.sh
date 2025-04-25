#!/bin/bash
# FB15k-237
CUDA_VISIBLE_DEVICES=0 python clqa/src/main.py --cuda \
	--data_path data/FB15k-237-betae \
	--do_train \
	--do_test \
	--model sphere \
	-lr 0.0001 -n 128 -b 512 -d 256 -g 24 -cenr "(none, 0.02)" \
	--save_checkpoint_steps 30000 --valid_steps 30000 \
	--max_steps 450001 --cpu_num 4 --seed 0 --print_on_screen \
	--tasks "1p.2p.3p" \
	--select_train_query "1p" \
	--select_test_query "1p.2p.3p"
# # FB15K
# CUDA_VISIBLE_DEVICES=0 python clqa/src/main.py --cuda \
# --data_path data/FB15k-betae \
# --do_train \
# --do_test \
# --model sphere \
# -lr 0.0001 -n 128 -b 512 -d 256 -g 36 -cenr "(none, 0.02)" \
# --save_checkpoint_steps 30000 --valid_steps 30000 \
# --max_steps 750001 --cpu_num 4 --seed 0 --print_on_screen \
# --tasks "1p.2p.3p" \
# --select_train_query "1p" \
# --select_test_query "1p.2p.3p"
# # NELL995
# CUDA_VISIBLE_DEVICES=0 python clqa/src/main.py --cuda \
# --data_path data/NELL-betae \
# --do_train \
# --do_test \
# --model sphere \
# -lr 0.0003 -n 128 -b 512 -d 256 -g 24 -cenr "(none, 0.02)" \
# --save_checkpoint_steps 30000 --valid_steps 30000 \
# --max_steps 450001 --cpu_num 4 --seed 0 --print_on_screen \
# --tasks "1p.2p.3p" \
# --select_train_query "1p" \
# --select_test_query "1p.2p.3p"
