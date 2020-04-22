import scipy.io as io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random


mat = io.loadmat('./data/lsp_dataset/joints.mat')
joints = mat['joints']
x_cords = joints[0].T 
y_cords = joints[1].T
visible = joints[2].T
joints_information = np.array([x_cords,y_cords,visible])
np.save('joints_information.npy',joints_information)

joints_info = np.load('joints_information.npy',allow_pickle=True)
x_cords_read = joints_info[0]
y_cords_read = joints_info[1]
visible_read = joints_info[2]

#-------------------------------------------------------------------------------------------

#This piece of code is just for verifying correctness of preprocessing the data

#for first image
Xs = x_cords_read[103] #x cords of all 14 joints for 1st image from dataset
Ys = y_cords_read[103] #y cords of all 14 joints for 1st image from dataset

#do not remove the scatter line below
Xs = [i+np.random.randint(-7,7) for i in Xs]
Ys = [i+np.random.randint(-7,7) for i in Ys]
plt.scatter(Xs,Ys)


im = cv2.imread('./data/lsp_dataset/images/im0104.jpg')
im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
plt.axis("off")
plt.imshow(im)
plt.show()

#--------------------------------------------------------------------------------------------

