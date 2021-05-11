import argparse
import os
import cv2
import numpy as np
import json

# ---------------------------------------------------------
parser = argparse.ArgumentParser(description='')
parser.add_argument('--image_dir', help='image_dir')
parser.add_argument('--pose_image_dir', help='openpose output without blending')
parser.add_argument('--pose_json_dir', help='openpose json')
parser.add_argument('--output_dir', help='openpose output')

args = parser.parse_args()
output_dir = args.output_dir

person_left_rgb_dir = os.path.join(output_dir, 'person_left', 'rgb')
person_left_pose_dir = os.path.join(output_dir, 'person_left', 'pose')
person_left_json_dir = os.path.join(output_dir, 'person_left', 'bb_json')

person_right_rgb_dir = os.path.join(output_dir, 'person_right', 'rgb')
person_right_pose_dir = os.path.join(output_dir, 'person_right', 'pose')
person_right_json_dir = os.path.join(output_dir, 'person_right', 'bb_json')

make_dir_list = [person_left_rgb_dir, person_left_pose_dir, person_left_json_dir, \
				person_right_rgb_dir, person_right_pose_dir, person_right_json_dir]

for make_dir in make_dir_list:
  if not os.path.exists(make_dir):
      os.makedirs(make_dir)

# ---------------------------------------------------------
def get_bb(person, shape, padding=60):
  x = [kp for kp in person[0::3] if kp > 0]
  y = [kp for kp in person[1::3] if kp > 0]

  height, width = shape[0], shape[1]

  x1 = round(min(x)); x2 = round(max(x))
  y1 = round(min(y)); y2 = round(max(y))

  ##--- add padding of 10 pixels
  x1 = max(0, x1 - padding)
  x2 = min(width, x2 + padding)

  y1 = max(0, y1 - padding)
  y2 = min(height, y2 + padding)

  return [x1, y1, x2, y2]


def crop(bb, image):
  x1, y1, x2, y2 = bb[0], bb[1], bb[2], bb[3]
  cropped_image = image[y1:y2, x1:x2, :]
  return cropped_image
# ---------------------------------------------------------
image_names = sorted(os.listdir(args.image_dir))

# ---------------------------------------------------------
for image_name in image_names:
  print(image_name)

  image_id = image_name.replace('.png', '').replace('.jpg', '')

  rgb_image_path = os.path.join(args.image_dir, image_name) 
  pose_image_path = os.path.join(args.pose_image_dir, '{}_rendered.png'.format(image_id)) 
  pose_json_path = os.path.join(args.pose_json_dir, '{}_keypoints.json'.format(image_id)) 

  with open(pose_json_path) as f:
    output = json.load(f)

  if len(output['people']) != 2:
    continue

  person_a = output['people'][0]['pose_keypoints_2d']
  person_b = output['people'][1]['pose_keypoints_2d']

  x_a = [kp for kp in person_a[0::3] if kp > 0]
  x_b = [kp for kp in person_b[0::3] if kp > 0]

  person_left = person_a
  person_right = person_b

  if min(x_b) < min(x_a):
    person_left = person_b
    person_right = person_a

  rgb_image = cv2.imread(rgb_image_path)
  pose_image = cv2.imread(pose_image_path)

  # --------------save left------------------
  bb_left = get_bb(person_left, shape=rgb_image.shape)
  pose_image_left = crop(bb_left, np.copy(pose_image))
  cv2.imwrite(os.path.join(person_left_pose_dir, image_name), pose_image_left)

  rgb_image_left = crop(bb_left, np.copy(rgb_image))
  cv2.imwrite(os.path.join(person_left_rgb_dir, image_name), rgb_image_left)

  bb = bb_left
  bb_dict = {'x1': bb[0], 'y1': bb[1], 'x2': bb[2], 'y2': bb[3]}
  with open(os.path.join(person_left_json_dir, '{}.json'.format(image_id)), 'w') as f:
    json.dump(bb_dict, f)

  # --------------save right------------------
  bb_right = get_bb(person_right, shape=rgb_image.shape)
  pose_image_right = crop(bb_right, np.copy(pose_image))
  cv2.imwrite(os.path.join(person_right_pose_dir, image_name), pose_image_right)

  rgb_image_right = crop(bb_right, np.copy(rgb_image))
  cv2.imwrite(os.path.join(person_right_rgb_dir, image_name), rgb_image_right)

  bb = bb_right
  bb_dict = {'x1': bb[0], 'y1': bb[1], 'x2': bb[2], 'y2': bb[3]}
  with open(os.path.join(person_right_json_dir, '{}.json'.format(image_id)), 'w') as f:
    json.dump(bb_dict, f)

