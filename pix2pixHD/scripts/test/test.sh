#!/bin/sh

cd ../..


# ### total frames 6869
# python test.py --name multipose2body_person_left --netG global \
# 				--resize_or_crop none \
# 				--checkpoints_dir './checkpoints' \
# 				--dataroot 'datasets/person_left' \
# 				--no_instance \
# 				--label_nc 0 \
# 				--results_dir './results/person_left' \
# 				--how_many 6869 \



python test.py --name multipose2body_person_right --netG global \
				--resize_or_crop none \
				--checkpoints_dir './checkpoints' \
				--dataroot 'datasets/person_right' \
				--no_instance \
				--label_nc 0 \
				--results_dir './results/person_right' \
				--how_many 6869 \
				--which_epoch 50 \
