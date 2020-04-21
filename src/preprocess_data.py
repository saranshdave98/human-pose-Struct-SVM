import scipy.io as io
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random


# mat = io.loadmat('./data/lsp_dataset/joints.mat')
# joints = mat['joints']
# x_cords = joints[0].T 
# y_cords = joints[1].T
# visible = joints[2].T
# joints_information = np.array([x_cords,y_cords,visible])
# np.save('joints_information.npy',joints_information)

joints_info = np.load('joints_information.npy',allow_pickle=True)
x_cords_read = joints_info[0]
y_cords_read = joints_info[1]
visible_read = joints_info[2]

#-------------------------------------------------------------------------------------------

#This piece of code is just for verifying correctness of preprocessing the data

#for first image
Xs = x_cords_read[0] #x cords of all 14 joints for 1st image from dataset
Ys = y_cords_read[0] #y cords of all 14 joints for 1st image from dataset

# Xs = [i+np.random.randint(-7,7) for i in Xs]
# Ys = [i+np.random.randint(-7,7) for i in Ys]



#skeleton
plt.scatter(Xs,Ys,color='black',marker='o')
plt.plot(Xs[:2],Ys[:2])
plt.plot(Xs[1:3],Ys[1:3])
plt.plot(Xs[2:4],Ys[2:4])
plt.plot(Xs[3:5],Ys[3:5])
plt.plot(Xs[4:6],Ys[4:6])
plt.plot(Xs[6:8],Ys[6:8])
plt.plot(Xs[7:9],Ys[7:9])
plt.plot(Xs[9:11],Ys[9:11])
plt.plot(Xs[10:12],Ys[10:12])
plt.plot(Xs[12:14],Ys[12:14])
plt.plot([Xs[8],Xs[12]],[Ys[8],Ys[12]])
plt.plot([Xs[9],Xs[12]],[Ys[9],Ys[12]])
plt.plot([Xs[2],Xs[12]],[Ys[2],Ys[12]])
plt.plot([Xs[3],Xs[12]],[Ys[3],Ys[12]])


im = cv2.imread('./data/lsp_dataset/images/im0001.jpg')
im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
plt.axis("off")
plt.imshow(im)
plt.show()

#--------------------------------------------------------------------------------------------

