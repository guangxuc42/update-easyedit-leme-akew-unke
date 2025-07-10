CUDA_VISIBLE_DEVICES=0
# python run_akew.py \
#  --editing_method=FT \
#  --hparams_dir=../hparams/FT/qwen2.5-7b.yaml \
#  --data_dir=../data/AKEW \
#  --edit_type=extract \
#  --ds_size=2 \
#  --data_type=WikiUpdate \

python run_akew.py \
 --editing_method=AnyEdit \
 --hparams_dir=../hparams/AnyEdit/qwen2.5-7b.yaml \
 --data_dir=../data/AKEW \
 --ds_size=2 \
 --edit_type=unstruct \
 --data_type=WikiUpdate \
