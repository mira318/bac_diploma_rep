import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

checkpoint_path = sys.argv[1]
res_df = pd.read_csv(checkpoint_path + '/summary.csv')

plt.figure(figsize = (20, 20))

plt.subplot(2, 2, 1)
plt.plot(res_df.epoch, res_df.train_loss, label = 'train')
plt.title('Cross entropy loss', fontsize = 25)
plt.xlabel('Epoch', fontsize = 20)
plt.xticks(fontsize = 20)
plt.ylabel('Loss', fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend(fontsize = 20)

plt.subplot(2, 2, 2)
plt.plot(res_df.epoch, res_df.eval_loss, label = 'val')
plt.title('Cross entropy loss', fontsize = 25)
plt.xlabel('Epoch', fontsize = 20)
plt.xticks(fontsize = 20)
plt.ylabel('Loss', fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend(fontsize = 20)

plt.show()