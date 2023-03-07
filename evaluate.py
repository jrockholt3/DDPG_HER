import matplotlib.pyplot as plt
import numpy as np
import pickle

file = open('tmp/score_hist_0302.pkl','rb')
score_hist = pickle.load(file)
# file = open('tmp/loss_hist.pkl','rb')
# loss_hist = pickle.load(file)

fig1 = plt.figure()
plt.plot(np.arange(len(score_hist)), score_hist)
plt.title('score history')
# fig2 = plt.figure()
# plt.plot(np.arange(len(loss_hist)), loss_hist)
# plt.title('critic loss history')
plt.show()