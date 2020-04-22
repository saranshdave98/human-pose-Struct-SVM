import matplotlib.pyplot as plt
import numpy as np
from utils.plotting import plot_iter_codebook,plot_data_vs_testing


# it,loss = plot_iter_codebook()

# plt.title("Kmeans Iteration vs Testing Error")
# plt.plot(it,loss)
# plt.xticks([])
# plt.yticks([])
# plt.xlabel("Iterations")
# plt.ylabel("Testing Error")
# plt.show()

size,svm, ssvm,lssvm = plot_data_vs_testing()

plt.title("Comparison")
plt.plot(size,svm,label='SVR')
plt.plot(size,ssvm,label='SSVR')
# plt.plot(size,lssvm,label='LSVR')
plt.xticks([])
plt.yticks([])
plt.xlabel("Dataset Size")
plt.ylabel("Testing Error")
plt.legend()
plt.show()