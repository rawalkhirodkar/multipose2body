cd ../..


CUDA_VISIBLE_DEVICES=3,2,1 python -m torch.distributed.launch train_global.py --dataroot 'datasets/video2_global_train' \
				--label_nc 0 \
				--resize_or_crop none \
				--no_instance \
				--name video2_global \
				--model globalpix2pixHD \
				--input_nc 9 \
				--checkpoints_dir './checkpoints/global_video2' \
				--batchSize 9 \
				--gpu_ids 0,1,2 \

# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch train_global.py --dataroot 'datasets/video2_global_train' \
# 				--label_nc 0 \
# 				--resize_or_crop none \
# 				--no_instance \
# 				--name video2_global \
# 				--model globalpix2pixHD \
# 				--input_nc 9 \
# 				--checkpoints_dir './checkpoints/global_video2' \
# 				--batchSize 3 \
# 				--nThreads 0 \
