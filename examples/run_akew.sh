export HF_HOME=/root/autodl-tmp/cache/
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0
# python run_akew.py \
#  --editing_method=FT \
#  --hparams_dir=../hparams/FT/qwen2.5-7b.yaml \
#  --data_dir=../data/AKEW \
#  --edit_type=extract \
#  --ds_size=2 \
#  --data_type=WikiUpdate \

# python run_akew.py \
#  --editing_method=UnKE \
#  --hparams_dir=../hparams/UnKE/qwen2.5-7b.yaml \
#  --data_dir=../data/AKEW \
#  --ds_size=2 \
#  --data_type=WikiUpdate \

# python run_leme.py \
#  --editing_method=FT \
#  --hparams_dir=../hparams/FT/qwen2.5-7b.yaml \
#  --data_dir=../data/LEME \
#  --ds_size=50 \
#  --data_type=ZsRE \
#  >./logs/qa--ft--qwen2.5--batch=1--ds=50.log 2>&1

python run_leme.py \
 --editing_method=FT \
 --hparams_dir=../hparams/FT/qwen2.5-7b.yaml \
 --data_dir=../data/LEME \
 --ds_size=50 \
 --data_type=CounterFact \
 >./logs/cf--ft--qwen2.5--batch=1--ds=50.log 2>&1
