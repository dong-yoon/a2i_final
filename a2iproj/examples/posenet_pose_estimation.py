"""
python3 examples/movenet_pose_estimation.py \
  --model test_data/movenet_single_pose_lightning_ptq_edgetpu.tflite  \
  --input test_data/squat.bmp
```
"""

import argparse
import os
import cv2
import numpy as np

from pycoral.adapters import common
#from pycoral.utils.edgetpu import make_interpreter
from tflite_runtime.interpreter import load_delegate
from tflite_runtime.interpreter import Interpreter

_NUM_KEYPOINTS = 17
EDGETPU_SHARED_LIB = 'libedgetpu.so.1'
POSENET_SHARED_LIB = os.path.join('posenet_lib', os.uname().machine,
        'posenet_decoder.so')

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default = 'model/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite', help='File path of .tflite file.')
    parser.add_argument('-i', '--image', default = None, help='Image to be classified.')
    parser.add_argument('-a', '--movie', default = 'stretch.mp4', help = 'Movie to be classified.')
    parser.add_argument('-f', '--fps', default = 30,type = int, help = 'FPS of output movie')
    parser.add_argument('-s', '--second', default = 3,help = 'Length of output movie')
    parser.add_argument('-c','--camera_idx',default = 0,help = 'Index of which video source to use')  
    parser.add_argument('--output',default='movenet_result.jpg',help='File path of the output image.')
    args = parser.parse_args()
  
    edgetpu_delegate = load_delegate(EDGETPU_SHARED_LIB)
    posenet_decoder_delegate = load_delegate(POSENET_SHARED_LIB)
    interpreter = Interpreter(args.model, experimental_delegates=[edgetpu_delegate,posenet_decoder_delegate])
    interpreter.allocate_tensors()
  
    if args.movie:
        cap = cv2.VideoCapture(args.movie)
    else:
        cap = cv2.VideoCapture(args.camera_idx)
    fps = args.fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    frames = fps*args.second
      
    while frames>0 and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img = frame
        inference_image = inference(interpreter, img)

          
        out.write(inference_image)
        frames-=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def inference(interpreter, img, threshold = 0.2):
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    model_w, model_h = common.input_size(interpreter)
    scale_x, scale_y = width/model_w, height/model_h

    resized_img = cv2.resize(img_rgb, (model_w, model_h))
    common.set_input(interpreter, resized_img)
    
    interpreter.invoke()

    keypoints = common.output_tensor(interpreter, 0).copy()
    keypoint_scores = common.output_tensor(interpreter, 1).copy()
    pose_scores = common.output_tensor(interpreter, 2).copy()
    num_poses = common.output_tensor(interpreter, 3).copy()
    
    
    return draw_skeleton(img, keypoints, pose_scores, num_poses, scale_x, scale_y)


def draw_skeleton(img, keypoints, pose_scores, num_poses, scale_x, scale_y, threshold = 0.2):
  
  height,width, _ = img.shape
  edge_pair = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),(6,8),
          (8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
  coords = {}
  num_poses = np.squeeze(num_poses)
  pose_scores = np.squeeze(pose_scores, axis = 0)
  keypoints = np.squeeze(keypoints, axis = 0)
  
  for i in range(int(num_poses)):
    if pose_scores[i]>=threshold:

        for j in range(0, _NUM_KEYPOINTS):

            x, y = int(round(keypoints[i][j][1]*scale_x)), int(round(keypoints[i][j][0]*scale_y))
            cv2.circle(img, (x, y),
            6,(0,0,255),-1)
            coords[j] = (x,y)

        for start, end in edge_pair:
            cv2.line(img, coords[start], coords[end], (255,0,0),5)

  return img  


if __name__ == '__main__':
  main()
