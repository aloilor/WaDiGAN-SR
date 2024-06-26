#!/bin/sh
export MASTER_PORT=6036
echo MASTER_PORT=${MASTER_PORT}

export PYTHONPATH=$(pwd):$PYTHONPATH

CURDIR=$(cd $(dirname $0); pwd)
echo 'The work dir is: ' $CURDIR

DATASET=$1
MODE=$2
GPUS=$3

if [ -z "$1" ]; then
   GPUS=1
fi

echo $DATASET $MODE $GPUS

# ----------------- Wavelet -----------
if [[ $MODE == train ]]; then
	echo "==> Training SRWaveDiff"

	if [[ $DATASET == celebahq_16_128 ]]; then 
		python train_srwddgan.py --dataset celebahq_16_128 --image_size 128 --exp celebahq16_128_wolatent_batch64_ts2_atn16_wg12224_d6_recloss_150ep --num_channels 24 \
			--num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 64 --num_epoch 100 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--cond_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content \
			--datadir /content/gdrive/MyDrive/srwavediff/datasets/celebahq_16_128/ \
			--master_port $MASTER_PORT \
			--current_resolution 64 --attn_resolution 16 --num_disc_layers 6 --rec_loss \
			--l_resolution 16 --h_resolution 128 \
			--use_pytorch_wavelet \		
	fi
else
	echo "==> Testing WaveDiff"

	if [[ $DATASET == celebahq_16_128 ]]; then
		python test_srwddgan.py --dataset celebahq_16_128 --image_size 128 --exp celebahq16_128_wolatent_batch64_ts2_atn16_wg12224_d6_recloss_150ep --num_channels 24 --num_channels_dae 64 \
			--ch_mult 1 2 2 2 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 70 #--num_iters 25000 \
			--current_resolution 64 --attn_resolutions 16 \
			--net_type wavelet \
			--l_resolution 16 --h_resolution 128 \
			--datadir /content/gdrive/MyDrive/srwavediff/datasets/celebahq_16_128/ \
			--batch_size 64 \
			#--compute_fid --real_img_dir ./pytorch_fid/celebahq128_stats.npz \
			#--measure_time \

	fi
fi
