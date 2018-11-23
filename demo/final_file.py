import os
import sys
import scipy.io
import numpy
sys.path.append(os.path.dirname(__file__) + "/../")

from scipy.misc import imread

from config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input
import cv2
import matplotlib.pyplot as plt
import pickle
import math
##Functions
def dist(par1,par2):
    return math.sqrt((par1[0]-par2[0])**2 + (par1[1]-par2[1])**2)
#print(dist([5,4],[2,7]))
cfg = load_config("demo/pose_cfg.yaml")
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

#file_name = "demo/img.jpg"
file_name1 = "demo/images_check/"+sys.argv[1]
file_name2 = "demo/images_check/"+sys.argv[2]
image1 = imread(file_name1, mode='RGB')
image2 = imread(file_name2, mode="RGB")
image_batch = data_to_input(image1)
image_batch2 = data_to_input(image2)
outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)
pose1 = predict.argmax_pose_predict(scmap, locref, cfg.stride)
outputs_np = sess.run(outputs, feed_dict={inputs: image_batch2})
scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)
pose2 = predict.argmax_pose_predict(scmap, locref, cfg.stride)
##print(pose1)
##print("-----------------------------------------------------------------------")
##print(pose2)
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
h1 = int(sys.argv[3])
h2 = int(sys.argv[4])
## For T-shirt
sc1 = h1/(((pose1[0][1] + pose1[5][1])//2) - pose1[13][1])
sc2 = h2/(((pose2[0][1] + pose2[5][1])//2) - pose2[13][1])
print("Scaling")
print(sc1,sc2)
dis_shoulders1 = dist(pose1[8],pose1[9])*sc1
dis_shoulders2 = dist(pose2[8],pose2[9])*sc2
##dist b/w shoulders and hips
# dis_vert_sh_person_left(1) or right(2)
dis_vert_sh_1_1 = dist(pose1[8],pose1[2])*sc1
dis_vert_sh_1_2 = dist(pose1[9],pose1[3])*sc1
dis_vert_sh_2_1 = dist(pose2[8],pose2[2])*sc2
dis_vert_sh_2_2 = dist(pose2[9],pose2[3])*sc2
print(dis_vert_sh_1_1)
print(dis_vert_sh_1_2)
print(dis_vert_sh_2_1)
print(dis_vert_sh_2_2)
print(dis_shoulders1,dis_shoulders2)
## Calculating fitting parameter
## Fitting parameter is calculated for 2nd image
ft_1 = 1.5*abs(dis_shoulders2 - dis_shoulders1) + 0.5*abs(dis_vert_sh_2_1-dis_vert_sh_1_1) + 0.5*abs(dis_vert_sh_2_2 - dis_vert_sh_1_2)
print(ft_1)
if(ft_1>0):
    print("Try one size smaller")
##arr_formatted = [(pose[i][0],pose[i][1]) for i in range(0,len(pose))]
##arr_con = [pose[i][2] for i in range(0,len(pose))]
##arr_x = [arr_formatted[i][0] for i in range(len(arr_formatted))]
##arr_y = [arr_formatted[i][1] for i in range(len(arr_formatted))]
##print(cfg['all_joints_names'])
##ret_arr = [arr_x,arr_y,arr_con]
##scipy.io.savemat('arrdata.mat', mdict={'arr': ret_arr})
im = plt.imread(file_name2)
implot = plt.imshow(im)
plt.scatter([pose2[i][0] for i in range(len(pose2))],[pose2[i][1] for i in range(len(pose2))])
for i in range(len(pose2)):
    plt.annotate(i,(pose2[i][0],pose2[i][1]))
plt.show()
# Visualise
#visualize.show_heatmaps(cfg, image, scmap, pose)
#visualize.waitforbuttonpress()
