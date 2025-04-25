#!/bin/bash
# FB15K-237
CUDA_VISIBLE_DEVICES=0 python main.py --cuda \
	--data_path data/FB15k-237-betae \
	--clqa_path clqa/pre_trained_clqa/FB15k-237_box_400_1p \
	--neural_adj_path neural_adj/ \
	--pre_model "clqa" \
	--model_name "box" -d 400 -g 24 --center_mode "(none,0.02)" \
	--fraction 100 --ent_sample 10 \
	--thrshd 0.000001 --neg_scale 1 \
	--cpu_num 8
# # FB15K
# CUDA_VISIBLE_DEVICES=0 python main.py --cuda \
# --data_path data/FB15k-betae \
# --clqa_path clqa/pre_trained_clqa/FB15k_box_400_1p \
# --neural_adj_path neural_adj/ \
# --pre_model "clqa" \
# --model_name "box" -d 400 -g 24 --center_mode "(none,0.02)" \
# --fraction 100 --ent_sample 10 \
# --thrshd 0.00001 --neg_scale 3 \
# --cpu_num 8 \
# # NELL995
# # PART 1
# CUDA_VISIBLE_DEVICES=0 python main.py --cuda \
# --data_path data/NELL-betae \
# --clqa_path clqa/pre_trained_clqa/NELL995_box_400_1p \
# --neural_adj_path neural_adj/ --single_rel \
# --pre_model "clqa" \
# --model_name "box" -d 400 -g 24 --center_mode "(none,0.02)" \
# --fraction 100 --ent_sample 10 \
# --thrshd 0.000001 --neg_scale 3 \
# --cpu_num 8 \
# --tasks "1p.2p.3p.2i.3i.ip.pi"
# # PART 2 (this script can be executed when atomic query matrix is successfully generated in part 1)
# # replace `yyyy.mm.dd-hh.mm.ss` with the true path of atomic query matrix created in PART 1 in the following
# CUDA_VISIBLE_DEVICES=0 python main.py --cuda \
# --data_path data/NELL-betae \
# --clqa_path clqa/pre_trained_clqa/NELL995_box_400_1p \
# --neural_adj_path neural_adj/yyyy.mm.dd-hh.mm.ss/ --single_rel \
# --pre_model "clqa" \
# --model_name "box" -d 400 -g 24 --center_mode "(none,0.02)" \
# --fraction 100 --ent_sample 10 \
# --thrshd 0.000001 --neg_scale 3 \
# --cpu_num 8 \
# --tasks "2u.up.2in.3in.inp.pin.pni"
