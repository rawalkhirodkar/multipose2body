cd ../..

# CUDA_VISIBLE_DEVICES=2,1,0 python -m torch.distributed.launch train.py --dataroot datasets/person_left \
# 				--label_nc 0 \
# 				--resize_or_crop none \
# 				--no_instance \
# 				--name multipose2body \
# 				--batchSize 24 \
# 				--gpu_ids 0,1,2 \


# CUDA_VISIBLE_DEVICES=2,1,0 python train_global.py --dataroot datasets/video1_global \
# 				--label_nc 0 \
# 				--resize_or_crop none \
# 				--no_instance \
# 				--name video1_global \
# 				--debug \
# 				--nThreads 0 \
# 				--model globalpix2pixHD \
# 				--input_nc 9


CUDA_VISIBLE_DEVICES=2,1,0 python -m torch.distributed.launch train_global.py --dataroot datasets/video1_global \
				--label_nc 0 \
				--resize_or_crop none \
				--no_instance \
				--name video1_global \
				--model globalpix2pixHD \
				--input_nc 9 \
				--checkpoints_dir './checkpoints/global'
