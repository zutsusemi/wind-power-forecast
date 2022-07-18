python main.py --device 0 \
--seed 0 --load ../kdd/model_weight_6000.pth \
--save ../output/ --s_iter 1000 --dataset kddcup --data_path ../kdd/kdd_train.csv --data_path_test ../kdd/kdd_test.csv \
--model attention_mlp \