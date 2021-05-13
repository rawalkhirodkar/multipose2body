cd /openpose

rm -rf ./pose_jsons
rm -rf ./pose_images
rm -rf /home/rawal/Desktop/multipose2body/openpose/output


### for video
# ./build/examples/openpose/openpose.bin --video examples/media/video.avi\
# 					--face --hand \
# 					--write_json ./output_json \
# 					--display 0 \
# 					--write_images ./output_images \
# 					--disable_blending


### for images
./build/examples/openpose/openpose.bin --image_dir /home/rawal/Desktop/datasets/multipose2body/raw_frames/00000\
					--write_json ./pose_jsons \
					--display 0 \
					--write_images ./pose_images \
					--disable_blending

mkdir /home/rawal/Desktop/multipose2body/openpose/output

cp -r ./pose_jsons /home/rawal/Desktop/multipose2body/openpose/output
cp -r ./pose_images /home/rawal/Desktop/multipose2body/openpose/output


# # -----------------------------------------------------
# cd /openpose

# rm -rf ./pose_jsons
# rm -rf ./pose_images
# rm -rf /home/rawal/Desktop/multipose2body/openpose/output


# ### for images
# ./build/examples/openpose/openpose.bin --image_dir /home/rawal/Desktop/datasets/multipose2body/raw_frames/00004\
# 					--write_json ./pose_jsons \
# 					--display 0 \
# 					--write_images ./pose_images \
# 					# --disable_blending

# mkdir /home/rawal/Desktop/multipose2body/openpose/output

# cp -r ./pose_jsons /home/rawal/Desktop/multipose2body/openpose/output
# cp -r ./pose_images /home/rawal/Desktop/multipose2body/openpose/output

