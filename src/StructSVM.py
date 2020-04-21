from sklearn.svm import SVR
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import numpy as np
import time
import sys


def get_data(concatenate = False):
    train_batch_size = 150
    train_X = np.load('training_input.npy',allow_pickle=True)[:train_batch_size]
    print(train_X.shape)

    joints_info = np.load('joints_information.npy',allow_pickle=True)
    x_cords_train = joints_info[0][:train_batch_size]
    print(x_cords_train.shape)
    y_cords_train = joints_info[1][:train_batch_size]
    print(y_cords_train.shape)
    y = np.concatenate([x_cords_train,y_cords_train],axis=1)
    print(y.shape)

    #concatenating input vector and coordinates
    X_cat = np.concatenate([train_X,y],axis=1)
    print(X_cat.shape)

    if concatenate==False:
        X_cat = train_X

    return X_cat,y



def joint_rbf_kernel(x1,y1,x2,y2):
    inp1 = np.concatenate([x1,y1])
    inp2 = np.concatenate([x2,y2])
    inp = np.linalg.norm((inp1-inp2),2)   
    k = np.exp(-1*inp) 
    return k


def compute_H_ij(X,y,i,j):
    sum_ = 0
    for p in range(X.shape[0]):
        for q in range(X.shape[0]):
            t1 = joint_rbf_kernel(X[p],y[p],X[q],y[q])
            t2 = joint_rbf_kernel(X[p],y[p],X[q],y[j])
            t3 = joint_rbf_kernel(X[p],y[i],X[q],y[q])
            t4 = joint_rbf_kernel(X[p],y[i],X[q],y[j])
            sum_ = sum_ + (t1 - t2 - t3 + t4)
    
    return sum_

def loss(y,y_bar):
    inp = np.linalg.norm((y-y_bar),2)   
    k = np.exp(-1*inp)
    return 2*(1-k)

def delta_per_sample(y,i):
    sum_ = 0
    for j in range(y.shape[0]):
        sum_ = sum_ + loss(y[i],y[j])
    
    return sum_




X_concatenated,y = get_data(concatenate=True)
X,y = get_data(concatenate=False)

m,n = X.shape

H = np.zeros((m,m))

# for i in range(m):
#     print(i)
#     for j in range(m):

#         # ts = time.time()       

#         H[i][j] = compute_H_ij(X,y,i,j)

#         # tt = time.time()
#         # print(tt-ts)


# print(H.shape)
# np.save('H_150.npy',H)

H = np.load('H_150.npy',allow_pickle=True)
H = H/(m*m)

C = 10


del_y = np.zeros((m,1))
for i in range(m):
    del_y[i] = delta_per_sample(y,i)/m
print(del_y.shape)

#Converting into cvxopt format
P = cvxopt_matrix(H)
q = cvxopt_matrix(del_y)
G = cvxopt_matrix(np.zeros((m,1)).T)
h = cvxopt_matrix(np.zeros(1))
A = cvxopt_matrix(np.ones((m,1)).T)
b = cvxopt_matrix(np.ones(1)*C)

#Setting solver parameters (change default to decrease tolerance) 
cvxopt_solvers.options['show_progress'] = True
cvxopt_solvers.options['abstol'] = 1e-10
cvxopt_solvers.options['reltol'] = 1e-10
cvxopt_solvers.options['feastol'] = 1e-10

#Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])

print(alphas.shape)


#testing
test_batch_size = 10

testing_X = np.load('training_input.npy',allow_pickle=True)[:test_batch_size]
print(testing_X.shape)

joints_info = np.load('joints_information.npy',allow_pickle=True)
x_cords_train = joints_info[0][:test_batch_size]
print(x_cords_train.shape)
y_cords_train = joints_info[1][:test_batch_size]
print(y_cords_train.shape)
y_gt = np.concatenate([x_cords_train,y_cords_train],axis=1)
print(y_gt.shape)



solutions = []
for x,y_g in zip(testing_X,y_gt):
    max_ = -1*(sys.maxsize-100000)
    opt_y = None
    loop = 0
    for y_k in y:
        print(loop)
        loop+=1
        sum_out = 0
        for j in range(m):
            sum_ = 0
            for i in range(m):
                t1 = joint_rbf_kernel(X[i],y[i],x,y_k)
                t2 = joint_rbf_kernel(X[i],y[j],x,y_k)
                sum_ = sum_ + (t1-t2)
            sum_out = sum_out + alphas[j]*sum_       
        if sum_out>max_ :
            max_ = sum_out
            opt_y = y_k
    solutions.append(opt_y)
    # print(opt_y)
    # print()
    # print(y_g)

solutions = np.array(solutions)
print(solutions.shape)
np.save('StructSVM_predictions_150_10.npy',solutions)