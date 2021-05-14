import os
from copy import deepcopy
import json
import cv2
import numpy as np
# from util import draw_bodypose
import pdb

OPENPOSE_ORDER = ['head', 'neck', 'right_shoulder', 'right_elbow', 'right_hand', 'left_shoulder', 'left_elbow', 'left_hand', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle']

with open('00004_formatted_new.json', 'r') as fp:
    json_data = json.load(fp)


# draw the body keypoint and lims
### candidates = num keypoints
### subset = index into candidates, N peoplese 

def draw_bodypose(canvas, candidate, subset):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(18):
        for n in range(len(subset)): ## loop over person, n
            index = int(subset[n][i]) ### index for person x keypoints
            if index == -1:
                continue
            x, y = candidate[index][0:2] #### hint!!
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1) #### hint!!
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    # plt.imsave("preview.jpg", canvas[:, :, [2, 1, 0]])
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    return canvas

def create_skeleton(keypoints):
    H = 820
    W = 1380
    frame = np.zeros((H, W, 3))

    points = list()

    import pdb; pdb.set_trace()

    for i in range(len(keypoints)//3):
        cv2.circle(frame, (int(keypoints[i*3]), int(keypoints[i*3+1])), 15, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        points.append((int(keypoints[i*3]), int(keypoints[i*3+1])))

    POSE_PAIRS = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                   [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                   [1, 16], [16, 18], [3, 17], [6, 18]]

    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 0), 3)

    cv2.imwrite('skeleton.png', frame)
    return


def load_openpose_json(path, image_name):
    with open(os.path.join(path, image_name+'_keypoints.json'), 'r') as fp:
        openpose_json = json.load(fp)
    return openpose_json


def check_if_keypoints_in_bb(manual_bbox, keypoints):
    npoints_inside = 0
    zero_points = 0 # invalid points
    top_width, top_height, width, height = manual_bbox
    nkeypoints = len(keypoints)//3
    
    # pdb.set_trace()
    for idx in range(nkeypoints):
        if keypoints[idx*3] == 0 and keypoints[idx*3] == 0:
            zero_points += 1
        elif keypoints[idx*3] >= top_width and keypoints[idx*3] <= top_width + width and keypoints[idx*3+1]  >= top_height and keypoints[idx*3+1] <= top_height + height:
            npoints_inside += 1
    
    # pdb.set_trace()
    if npoints_inside == nkeypoints-zero_points:
        return True
    else:
        return False


def merge_jsons(openpose_json, annotated_json, image_id):
    for annotation in annotated_json:
        if os.path.basename(annotation['image_name']).split('_')[0] == image_id:
            print('found')
            manual_annotation = annotation
            break

    for annotation in manual_annotation['keypoints']:
        # pdb.set_trace()
        occluded_keypoints = list()
        for keypoint in manual_annotation['keypoints'][annotation]:
            if not np.array_equal(manual_annotation['keypoints'][annotation][keypoint], np.array([0, 0, 0])):
                occluded_keypoints.append(keypoint)

        # pdb.set_trace()
        for pid, person in enumerate(openpose_json['people']):
            _person = deepcopy(person)    
            for occ_keypoint in occluded_keypoints:
                if occ_keypoint in OPENPOSE_ORDER:
                    idx = OPENPOSE_ORDER.index(occ_keypoint)
                    _person['pose_keypoints_2d'][idx*3] = manual_annotation['keypoints'][annotation][occ_keypoint][0]
                    _person['pose_keypoints_2d'][idx*3+1] = manual_annotation['keypoints'][annotation][occ_keypoint][1]
                    # set confidence high for manual annotations
                    _person['pose_keypoints_2d'][idx*3+2] = 0.99
        
        # pdb.set_trace()
        if check_if_keypoints_in_bb(manual_annotation[annotation+'_bbox'], _person['pose_keypoints_2d']):
            openpose_json['people'][pid] = _person
            # break
    
    # pdb.set_trace()
    return openpose_json


if __name__ == '__main__':
    images = [os.path.basename(annotation['image_name']) for annotation in json_data]
    
    for image in images:
        image = '00506'
        # openpose_json = load_openpose_json('/home/mab623/pose_jsons', image.split('_')[0])
        openpose_json = load_openpose_json('/home/rawal/Desktop/datasets/multipose2body/openpose_output/00000/pose_jsons', image.split('_')[0])

        openpose_output = merge_jsons(openpose_json, json_data, image_id=image)

        # TODO: use openpose plot functionality here
        create_skeleton(openpose_output['people'][1]['pose_keypoints_2d'])
        break
    print('here')
    pdb.set_trace()
