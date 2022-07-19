import pandas as pd
import numpy as np
path = 'D:\\2021_Summer\\VE450\\models\\wind-power-forecast\\kdd\\sdwpf_baidukddcup2022_turb_location.CSV'

def get_adj_mat(path, thres=100):
    location = pd.read_csv(path)
    x = location['x'].to_numpy()
    y = location['y'].to_numpy()
    l2 = (x.reshape(-1, 1) - x.reshape(1, -1))**2 + (y.reshape(-1, 1) - y.reshape(1, -1))**2
    l2 = l2 ** 0.5
    adj = (l2 < thres).astype(np.float32)
    adj -= np.eye(adj.shape[0], adj.shape[1])
    
    
    # import matplotlib.pyplot as plt
    
    # img = plt.figure()
    # ax = img.subplots()
    
    # ax.scatter(x, y)
    # for j in range(adj.shape[0]):
    #     for k in range(adj.shape[1]):
    #         if adj[j, k] == 1:
    #             ax.plot([x[j], x[k]],[y[j], y[k]],color='g')
    # img.show()


    return adj


A = get_adj_mat(path, 2000)


