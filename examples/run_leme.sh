CUDA_VISIBLE_DEVICES=0

python run_leme.py \
 --editing_method=FT \
 --hparams_dir=../hparams/FT/qwen2.5-7b.yaml \
 --data_dir=../data/LEME \
 --ds_size=2 \
 --data_type=ZsRE \

python run_leme.py \
 --editing_method=FT \
 --hparams_dir=../hparams/FT/qwen2.5-7b.yaml \
 --data_dir=../data/LEME \
 --ds_size=2 \
 --data_type=CounterFact \
