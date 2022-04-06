#!/bin/sh

# run copy1
#python -u run_copy1.py --gpu 3 --n-hidden 200  --evaluate-every 250 --pretrained_model_state_file 'pretrained_model_state_run_nhidden200_0509.pth' --model_state_file 'model_state_copy1.pth' > nohup_copy1.log 2>&1 &

# run aux
#python -u run_aux.py --gpu 1 --n-epochs 5000 --n-hidden 200  --evaluate-every 250 --pretrained_model_state_file 'pretrained_model_state_run_nhidden200_0509.pth' --model_state_file 'model_state_aux.pth' > nohup_aux_nhidden200.log 2>&1 &
#python -u run_aux.py --gpu 1 --n-epochs 5000 --n-hidden 200 --evaluate-every 250 --pretrained_model_state_file 'model_state_run_aux_nhidden200.pth' --model_state_file 'model_state_aux.pth' > nohup_aux_nhidden200_2.log 2>&1 &
#python -u run_aux.py --gpu 3 --n-epochs 5000 --n-hidden 200  --evaluate-every 250 --pretrained_model_state_file 'pretrained_model_state_run_nhidden200_0509.pth' --model_state_file 'model_state_aux_batch.pth' > nohup_aux_nhidden200_batch.log 2>&1 &


# run no total graph
#python -u run_not_total_graph.py --gpu 0 --n-epochs 4000 --C 0.001 --n-hidden 200 --evaluate-every 250 --pretrained_model_state_file 'pretrained_model_state_run_nhidden200_0509.pth' --model_state_file 'model_state_not_total_graph.pth' > nohup_not_total_graph.log 2>&1 &

# run not update rules (aux and virtual data are trained respectively)
#python -u run_not_update_rules.py --gpu 2 --C 0.00001 --n-epochs 4000 --n-hidden 200 --evaluate-every 50 --pretrained_model_state_file pretrained_model_state_run_nhidden200_0509.pth --model_state_file model_state_not_update_rules_C00001_sigmoid_pool3.pth --isSigmoid True --n-epochs-aux 100 > nohup_not_update_rules_C00001_sigmoid_pool3s.log 2>&1 &


# run aux and virtual together 
#python -u run_aux_and_virtual.py --gpu 1 --C 0.0001 --n-epochs 4000 --n-hidden 200 --evaluate-every 50 --pretrained_model_state_file model_state_aux_and_virtual__C0001_sigmoid0511.pth --model_state_file model_state_aux_and_virtual__C0001_sigmoid.pth --isSigmoid True > nohup_aux_and_virtual_C0001_sigmoid.log 2>&1 &

#python -u run_aux_and_virtual.py --gpu 1 --C 0.0001 --n-epochs 4000 --n-hidden 200 --evaluate-every 50 --pretrained_model_state_file pretrained_model_state_run_nhidden200_0509.pth  --model_state_file model_state_aux_and_virtual_C0001_sigmoid_pool3.pth --isSigmoid True > nohup_aux_and_virtual_C0001_sigmoid_pool3.log 2>&1 &


# calculate sigma before batch, save time but the computation branch of softlabel is discarded
#python -u run_aux_and_virtual_sigma_before_batch.py --gpu 3 --C 0.000001 --n-epochs 4000 --n-hidden 200 --evaluate-every 50 --pretrained_model_state_file pretrained_model_state_run_nhidden200_0509.pth --model_state_file model0915.pth --isSigmoid True --n-epochs-aux 100 > nohup_aux_and_virtual_C000001_sigmoid_pool3_sigma_before_batch.log 2>&1 &

# correct command
#python -u run_aux_and_virtual_sigma_before_batch.py --gpu 0 --C 0.00001 --n-epochs 4000 --n-hidden 200 --evaluate-every 50 --pretrained_model_state_file pretrained_model_state_run_nhidden200_0509.pth --model_state_file model0916.pth --isSigmoid True  --axiom_pool_dir axiom_pool_sys 

# run by run_vn_v1.py and  dataset created by hyq
# yago37
#python -u run_vn.py --gpu 3 --penalty 1 --epochs 4000 --embedding-dim 200 --evaluate-every 50  --data yago37  --isSigmoid True  --n-bases 20 --train-graph-size 200000 --batch-size 3000 --n_epochs_aux 100 

# wn18
#python -u run_vn.py --gpu 2 --penalty 1 --epochs 4000 --embedding-dim 200 --evaluate-every 50  --data wn18  --sub-data object-10 --isSigmoid True  --n-bases 20  --batch-size 10000 --n_epochs_aux 100  

# wn18rr  
#python -u run_vn.py --gpu 3 --penalty 1 --epochs 2500 --embedding-dim 200 --evaluate-every 50  --data wn18rr --sub-data subject-10 --isSigmoid True  --n-bases 20  --batch-size 10000   --n_epochs_aux 200

#python -u run_vn.py --gpu 2 --penalty 0.001 --epochs 4000 --embedding-dim 200 --evaluate-every 50  --data wn18  --isSigmoid True  --n-bases 20  --batch-size 100

# fb15k

#python -u run_vn.py --gpu 2 --penalty 0.1 --epochs 4000 --embedding-dim 200 --evaluate-every 50  --data fb15k  --sub-data object-10 --isSigmoid True  --n-bases 100  --batch-size 30000  --n_epochs_aux 100 

# fb15k237

decoder

python -u main_run.py --gpu 0 --penalty 0.5 --epochs 4000 --model "conve" --embedding-dim 200 --evaluate-every 50  --data fb15k  --sub-data subject-10 --isSigmoid True  --n-bases 100  --batch-size 4000  --n_epochs_aux 200

python -u main_run.py --gpu 1 --penalty 0.5 --epochs 4000 --model "analogy" --embedding-dim 200 --evaluate-every 50  --data fb15k  --sub-data subject-10 --isSigmoid True  --n-bases 100  --batch-size 4000  --n_epochs_aux 200

python -u main_run.py --gpu 2 --penalty 0.5 --epochs 4000 --model "transe" --embedding-dim 200 --evaluate-every 50  --data fb15k  --sub-data subject-10 --isSigmoid True  --n-bases 100  --batch-size 4000  --n_epochs_aux 200

hyper
0
python -u main_run.py --gpu 0 --penalty 0.5 --epochs 4000 --model "distmult" --embedding-dim 100 --evaluate-every 50  --data fb15k  --sub-data subject-10 --isSigmoid True  --n-bases 100  --batch-size 30000  --n_epochs_aux 200
python -u main_run.py --gpu 0 --penalty 0.5 --epochs 4000 --model "distmult" --embedding-dim 200 --evaluate-every 50  --data fb15k  --sub-data subject-10 --isSigmoid True  --n-bases 100  --batch-size 30000  --n_epochs_aux 200
python -u main_run.py --gpu 3 --penalty 0.5 --epochs 4000 --model "distmult" --embedding-dim 300 --evaluate-every 50  --data fb15k  --sub-data subject-10 --isSigmoid True  --n-bases 100  --batch-size 30000  --n_epochs_aux 200
python -u main_run.py --gpu 3 --penalty 0.5 --epochs 4000 --model "distmult" --embedding-dim 400 --evaluate-every 50  --data fb15k  --sub-data subject-10 --isSigmoid True  --n-bases 100  --batch-size 30000  --n_epochs_aux 200
python -u main_run.py --gpu 0 --penalty 0.5 --epochs 4000 --model "distmult" --embedding-dim 500 --evaluate-every 50  --data fb15k  --sub-data subject-10 --isSigmoid True  --n-bases 100  --batch-size 30000  --n_epochs_aux 200
# no pretrained model to avoid process of validating pretrained model
#python -u run_aux_and_virtual_sigma_before_batch.py --gpu 0 --C 0.000001 --n-epochs 4000 --n-hidden 200 --evaluate-every 1 --model_state_file model0916.pth --isSigmoid True  --axiom_pool_dir axiom_pool_sys 


# consider training process is interrupted
#python -u run_aux_and_virtual_sigma_before_batch.py --gpu 2 --negative-sample 64 --C 0.000001 --n-epochs 6000 --n-hidden 200 --evaluate-every 100 --pretrained_model_state_file model_state_aux_and_virtual_C000001_sigmoid_pool3_sigma_before_batch_epoch4100.pth --model_state_file model_state_aux_and_virtual_C000001_sigmoid_pool3_sigma_before_batch.pth --isSigmoid True  >> ./log/nohup_aux_and_virtual_C000001_sigmoid_pool3_sigma_before_batch.log 2>&1 &


# run aux and hard rule
#test graph consists of total data including the virtual data`;
#python -u run_aux_and_hard_rule.py --gpu 0 --n-epochs 5000 --n-hidden 200 --max_entailment 500 --evaluate-every 100 --pretrained_model_state_file pretrained_model_state_run_nhidden200_0509.pth --model_state_file model_state_aux_and_hard_rule_and_batch_pool3_max500.pth > nohup_hard_rule_pool3_max500.log 2>&1 &

#pretrin for the yago data
#python -u run_pretrain_200.py  --gpu 3 --n-bases 20 --valid_num 2000 --evaluate-every 300 --graph-batch-size 50000 --datadir ../data/yago37/head-10/ --pretrained_model_state_file model_yago_pretrained_base5.pth --axiom_pool_dir axiom_pool > nohup_yago_pretrain_base5.log 2>&1 &
##basis
#python -u run_pretrain_200.py  --gpu 1 --regularizer basis --n-bases 5  --valid_num 2000 --n-hidden 200 --evaluate-every 100 --graph-batch-size 15000  --datadir ../data/yago37/head-10/ --pretrained_model_state_file model_yago_pretrained_basis5.pth --axiom_pool_dir axiom_pool > nohup_yago_pretrain_basis5.log --negative-sample 64  2>&1 &

#bdd
#python -u run_pretrain_200.py  --gpu 0 --n-bases 40  --valid_num 2000 --evaluate-every 100  --negative-sample 50  --graph-batch-size 60000 --datadir ../data/yago37/head-10/ --pretrained_model_state_file model_yago_pretrained_bdd40_neg50.pth --axiom_pool_dir axiom_pool > nohup_yago_pretrain_bdd40_neg50.log 2>&1 &

# pretrain for the fb15k, with bases number 50
#python -u run_pretrain_200.py  --gpu 1 --n-bases 50  --evaluate-every 300 --graph-batch-size 20000 --datadir ../fb15K/head-10/ --pretrained_model_state_file model_fb15K_pretrained.pth --axiom_pool_dir axiom_pool > nohup_fb15K_pretrain.log 2>&1 &

# train for yago with soft labels
#python -u run_aux_and_virtual_sigma_before_batch.py --datadir ../data/yago37/head-10 --axiom_pool_dir axiom_pool_p07_times1 --gpu 1 --C 0.00001 --n-epochs 4000 --n-hidden 200 --evaluate-every 150 --pretrained_model_state_file model_yago_virtual_C00001_sigmoid_sigma_before_batch_epoch1350.pth --n-base 40 --model_state_file model_yago_virtual_C00001_sigmoid_sigma_before_batch.pth --isSigmoid True --n-epochs-aux 100 --max_entailment 4000 --is_save 1 --batch-virtual-size 7000 > nohup_yago_virtual_C00001.log 2>&1 &

