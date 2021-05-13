#!/bin/sh

cd ../..


python test_global.py --name video1_global --netG global \
				--resize_or_crop none \
				--checkpoints_dir 'checkpoints/global' \
				--dataroot datasets/video1_global \
				--no_instance \
				--label_nc 0 \
				--results_dir './results/global' \
				--how_many 6869 \
				--which_epoch 50 \
				--model globalpix2pixHD \
				--input_nc 9 \


