#!/bin/sh

cd ../..

python test.py --name multipose2body --netG global \
				--resize_or_crop none \
				--checkpoints_dir './checkpoints' \
				--dataroot 'datasets/person_left' \
				--no_instance \
				--label_nc 0 \
