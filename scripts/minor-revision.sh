python3.8 train.py --OOD_Score K+1 --lm 0.8 --lambda_G 0.6 --update_term 5 --lambda_ent_Ctk 0.25 --lambda_cls_Ctu 0.1 --net 'resnet50' --dataset "cifar10vsplace365" --warmup_iter 60000 --training_iter 20  --scheduler 'cos' --e_lr 0.002 --lr 0.001 --g_lr 0.1 --seed 0 --set_gpu 0
