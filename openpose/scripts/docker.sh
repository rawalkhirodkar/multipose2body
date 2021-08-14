docker run -it --rm --gpus all --net=host --name=openpose \
		-p 8888:8888 \
		--env="DISPLAY" \
    	--env="QT_X11_NO_MITSHM=1" \
		--ipc=host \
		--privileged \
		-e XAUTHORITY -e NVIDIA_DRIVER_CAPABILITIES=all \
		-e DISPLAY  \
		-v "/home/rawal:/home/rawal:Z" \
		cwaffles/openpose


docker run -it --rm --gpus all --net=host --name=openpose \
		-p 8888:8888 \
		--env="DISPLAY" \
    	--env="QT_X11_NO_MITSHM=1" \
		--ipc=host \
		--privileged \
		-e XAUTHORITY -e NVIDIA_DRIVER_CAPABILITIES=all \
		-e DISPLAY  \
		-v "/home/ANT.AMAZON.COM/khirawal:/home/ANT.AMAZON.COM/khirawal:Z" \
		cwaffles/openpose

cd /home/ANT.AMAZON.COM/khirawal/Desktop/multipose2body/openpose/scripts
