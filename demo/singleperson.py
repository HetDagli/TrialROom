import os
import sys
import scipy.io
import numpy
sys.path.append(os.path.dirname(__file__) + "/../")

import imageio

from config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input
import cv2
import matplotlib.pyplot as plt
import pickle
cfg = load_config("demo/pose_cfg.yaml")

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

# Read image from file
file_name = "demo/polo_tshirt.jpg"
image = imageio.imread(file_name, mode='RGB')

image_batch = data_to_input(image)

# Compute prediction with the CNN
outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

# Extract maximum scoring location from the heatmap, assume 1 person
pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
print(pose)
#0 -> left ankle
#1 -> left knee
#2 -> left hip
#3 -> right hip
#4 -> right knee
#5 -> right ankle
#6 -> left wrist
#7 -> left elbow
#8 -> left shoulder
#9 -> right shoulder
#10 -> right elbow
#11 -> right wrist
#12 -> neck
#13 -> head
arr_formatted = [(pose[i][0],pose[i][1]) for i in range(0,len(pose))]
arr_con = [pose[i][2] for i in range(0,len(pose))]
arr_x = [arr_formatted[i][0] for i in range(len(arr_formatted))]
arr_y = [arr_formatted[i][1] for i in range(len(arr_formatted))]
print(cfg['all_joints_names'])
ret_arr = [arr_x,arr_y,arr_con]
scipy.io.savemat('arrdata.mat', mdict={'arr': ret_arr})
im = plt.imageio.imread(file_name)
implot = plt.imshow(im)
plt.scatter([pose[i][0] for i in range(len(pose))],[pose[i][1] for i in range(len(pose))])
for i in range(len(pose)):
    plt.annotate(i,(pose[i][0],pose[i][1]))
plt.show()
# Visualise
#visualize.show_heatmaps(cfg, image, scmap, pose)
#visualize.waitforbuttonpress()
