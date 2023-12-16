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

	if [[ $DATASET == celebahq_16_128 ]]; then #same as celebahq_256 - might need to revisit later
		python train_wddgan.py --dataset celebahq_16_128 --image_size 128 --exp celebahq16_128_expxxx_atn16_wg12224_d5_recloss_300ep --num_channels 24 \
			--num_channels_dae 64 --ch_mult 1 2 4 8 8 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 32 --num_epoch 150 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content \
			--datadir /content/gdrive/MyDrive/srwavediff/datasets/celebahq_16_128/ \
			--master_port $MASTER_PORT \
			--current_resolution 64 --attn_resolution 16 --num_disc_layers 6 --rec_loss \
			--net_type wavelet \
			--l_resolution 16 --h_resolution 128 \

	elif [[ $DATASET == celebahq_16_64 ]]; then #same as celebahq_256 - might need to revisit later
		python train_wddgan.py --dataset celebahq_16_64 --image_size 64 --exp srwavediff_celebahq_exp3_atn16_wg12224_d5_recloss_500ep --num_channels 24 \
			--num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 128 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content \
			--datadir /content/gdrive/MyDrive/srwavediff/datasets/celebahq_16_64/ \
			--master_port $MASTER_PORT \
			--current_resolution 32 --attn_resolution 16 --num_disc_layers 5 --rec_loss \
			--net_type wavelet \
			--l_resolution 16 --h_resolution 64 \



	elif [[ $DATASET == stl10 ]]; then
		python train_wddgan.py --dataset stl10 --image_size 64 --exp wddgan_stl10_exp1_atn16_wg1222_d4_recloss_900ep/ --num_channels 12 --num_channels_dae 128 --num_timesteps 4 \
			--num_res_blocks 2 --batch_size 256 --num_epoch 900 --ngf 64 --nz 100 --z_emb_dim 256 --n_mlp 4 --embedding_type positional \
			--use_ema --ema_decay 0.9999 --r1_gamma 0.02 --lr_d 1.25e-4 --lr_g 1.6e-4 --lazy_reg 15 \
			--ch_mult 1 2 2 2 --save_content --datadir ./data/STL-10 \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 32 --attn_resolutions 16 --num_disc_layers 4 --rec_loss \
			--net_type wavelet \
			--use_pytorch_wavelet \

	elif [[ $DATASET == celeba_256 ]]; then
		python train_wddgan.py --dataset celeba_256 --image_size 256 --exp wddgan_celebahq_exp1_atn16_wg12224_d5_recloss_500ep --num_channels 12 --num_channels_dae 64 --ch_mult 1 2 2 2 4 --num_timesteps 2 \
			--num_res_blocks 2 --batch_size 32 --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --r1_gamma 2. \
			--z_emb_dim 256 --lr_d 1e-4 --lr_g 2e-4 --lazy_reg 10 --save_content --datadir data/celeba/celeba-lmdb/ \
			--master_port $MASTER_PORT --num_process_per_node $GPUS \
			--current_resolution 128 --attn_resolution 16 --num_disc_layers 5 --rec_loss \
			--save_content_every 10 \
			--net_type wavelet \
			# --use_pytorch_wavelet \
	fi
else
	echo "==> Testing WaveDiff"

	if [[ $DATASET == celebahq_16_128 ]]; then
		python test_wddgan.py --dataset celebahq_16_128 --image_size 128 --exp celebahq16_128_expxxx_atn16_wg12224_d5_recloss_300ep --num_channels 24 --num_channels_dae 64 \
			--ch_mult 1 2 4 8 8 --num_timesteps 2 --num_res_blocks 2  --epoch_id 20 \
			--current_resolution 64 --attn_resolutions 16 \
			--net_type wavelet \
			--l_resolution 16 --h_resolution 128 \
			--datadir /content/gdrive/MyDrive/srwavediff/datasets/celebahq_16_128/ \
			--batch_size 32 \
			# --compute_fid --real_img_dir ./pytorch_fid/celebahq128_stats.npz \
			# --measure_time \

	elif [[ $DATASET == celebahq_16_64 ]]; then
		python test_wddgan.py --dataset celebahq_16_64 --image_size 64 --exp srwavediff_celebahq16-64_exp3_atn16_wg12224_d5_recloss_500ep --num_channels 24 --num_channels_dae 64 \
			--ch_mult 1 2 2 2 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 100 \
			--current_resolution 32 --attn_resolutions 16 \
			--net_type wavelet \
			--l_resolution 16 --h_resolution 64 \
			--datadir /content/gdrive/MyDrive/srwavediff/datasets/celebahq_16_64/ \
			--batch_size 64 \
			--compute_fid --real_img_dir ./pytorch_fid/celebahq64_stats.npz \
			# --measure_time \

	elif [[ $DATASET == celeba_256 ]]; then
		python test_wddgan.py --dataset celeba_256 --image_size 256 --exp wddgan_celebahq_exp1_atn16_wg12224_d5_recloss_500ep --num_channels 12 --num_channels_dae 64 \
			--ch_mult 1 2 2 2 4 --num_timesteps 2 --num_res_blocks 2  --epoch_id 475 \
			--current_resolution 128 --attn_resolutions 16 \
			--net_type wavelet \
			--datadir /content/gdrive/MyDrive/srwavediff/datasets/celebahq_16_128/ \
			# --use_pytorch_wavelet \
			# --compute_fid --real_img_dir ./pytorch_fid/celebahq_stat.npy \
			# --batch_size 100 --measure_time \

	fi
fi
