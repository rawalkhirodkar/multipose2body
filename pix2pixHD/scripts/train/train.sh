cd ../..

CUDA_VISIBLE_DEVICES=2,1,0 python -m torch.distributed.launch train.py --dataroot datasets/person_left \
				--label_nc 0 \
				--resize_or_crop none \
				--no_instance \
				--name multipose2body \
				--batchSize 24 \
				--gpu_ids 0,1,2 \
