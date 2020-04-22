from sklearn.svm import SVR
from pystruct.learners import OneSlackSSVM
from pystruct.learners import LatentSSVM
import numpy as np

train_X1 = np.load('training_input.npy',allow_pickle=True)
train_X2 = np.load('training_input_augmented.npy',allow_pickle=True)[1300:1800]
train_X = np.concatenate([train_X1,train_X2])
print(train_X.shape)

joints_info = np.load('joints_information.npy',allow_pickle=True)
x_cords_train = joints_info[0][:1800]
y_cords_train = joints_info[1][:1800]
joints_info_aug = np.load('joints_info_aug.npy',allow_pickle=True)
x_cords_train_aug = joints_info_aug[0][1300:1800]
y_cords_train_aug = joints_info_aug[1][1300:1800]

x_cords_train = np.concatenate([x_cords_train,x_cords_train_aug])
y_cords_train = np.concatenate([y_cords_train,y_cords_train_aug])
print(x_cords_train.shape)
print(y_cords_train.shape)

test_X = np.load('testing_input.npy',allow_pickle=True)
joints_info = np.load('joints_information.npy',allow_pickle=True)
x_cords_test = joints_info[0][1800:]
y_cords_test = joints_info[1][1800:]

clf = SVR(C=1.0, epsilon=0.1,kernel='rbf')
# clf = OneSlackSSVM(C=1.0)
# clf = LatentSSVM(C=1.0)

#x-coordinates
predicted_loc_x = []
gt_loc_x = []
for i in range(14):
    #training
    train_y = x_cords_train[:,i]
    clf.fit(train_X, train_y)
    #testing
    gt = x_cords_test[:,i]
    prediction = clf.predict(test_X)    
    predicted_loc_x.append(np.copy(prediction))
    gt_loc_x.append(gt)

#y-coordinates
predicted_loc_y = []
gt_loc_y = []
for i in range(14):
    #training
    train_y = y_cords_train[:,i]
    clf.fit(train_X, train_y)
    #testing
    gt = y_cords_test[:,i]
    prediction = clf.predict(test_X)    
    predicted_loc_y.append(np.copy(prediction))
    gt_loc_y.append(gt)

predicted_loc_x = np.array(predicted_loc_x)
predicted_loc_y = np.array(predicted_loc_y)

print(predicted_loc_x.shape,predicted_loc_y.shape)

np.save('predicted_loc_x_aug_inc.npy',predicted_loc_x)
np.save('predicted_loc_y_aug_inc.npy',predicted_loc_y)