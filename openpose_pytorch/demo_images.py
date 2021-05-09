import copy
import numpy as np
import cv2
from glob import glob
import os
import argparse
import json

# video file processing setup
# from: https://stackoverflow.com/a/61927951
import argparse
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

# openpose setup
from src import model
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

def process_frame(frame,):
    canvas = copy.deepcopy(frame)
    candidate, subset = body_estimation(frame)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    
    return canvas, candidate, subset

# open specified video
parser = argparse.ArgumentParser(
        description="Process a images annotating poses detected.")
parser.add_argument('--dir', type=str, help='Dir file location to process.')
parser.add_argument('--output_dir', type=str, help='Dir file location to process.')
parser.add_argument('--output_json_dir', type=str, help='Dir fprocess.')

args = parser.parse_args()

image_names = sorted(os.listdir(args.dir))
output_dir = args.output_dir
output_json_dir = args.output_json_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(output_json_dir):
    os.makedirs(output_json_dir)

# ------------------------------------
keypoint_names = [
                    'nose', 'neck', \
                    'right_shoulder', 'right_elbow', 'right_wrist', \
                    'left_shoulder', 'left_elbow', 'left_wrist', \
                    'right_hip', 'right_knee', 'right_ankle', \
                    'left_hip', 'left_knee', 'left_ankle', \
                    'right_eye', 'left_eye', \
                    'right_ear', 'left_ear', \
                ]

# ------------------------------------
for i, image_name in enumerate(image_names):
    print(image_name)
    image_path = os.path.join(args.dir, image_name)
    frame = cv2.imread(image_path)

    posed_frame, candidate, subset = process_frame(frame)

    image_output = {'image': image_name, 'image_path': image_path}

    person_a = None
    person_b = None

    min_kps_x = 100000000000000000000
    min_person_id = None

    persons = []

    if len(subset) != 2:
        print('skipping')
        continue

    for n in range(len(subset)):
        keypoints = {}

        for i, keypoint_name in enumerate(keypoint_names):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y, score = candidate[index][0:3]

            keypoints[keypoint_name] = {'x': x, 'y': y, 'score': score}

            if x < min_kps_x:
                min_kps_x = x
                min_person_id = n

        persons.append(keypoints)

    if min_person_id == 0:
        person_left = persons[0]
        person_right = persons[1]
    elif min_person_id == 1:
        person_left = persons[1]
        person_right = persons[0]

    image_output['person_left'] = person_left
    image_output['person_right'] = person_right

    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, posed_frame)

    output_json_path = os.path.join(output_json_dir, image_name.replace('.png', '').replace('.jpg', '') + '.json')    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(image_output, f, ensure_ascii=False, indent=4)


