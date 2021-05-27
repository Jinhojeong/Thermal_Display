import os
import numpy as np
from scipy.signal import savgol_filter
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from config import config


for npy_name in os.listdir(config.npy_dest_dir):
    attributes = npy_name.split("_")
    del attributes[0]
    del attributes[-1]
    data = np.load(config.npy_dest_dir+npy_name)

    # clf = KNeighborsRegressor(n_neighbors=100, weights='uniform')
    timeline = data[:,0]
    hf = data[:,1]
    temp = data[:,2]
    temp = savgol_filter(temp, 99, 2, mode='nearest')
    
    force = data[:,3]
    force = savgol_filter(force, 15, 2, mode='nearest')
    material = data[:,4]
    Ra = data[:,5]

    plt.figure()
    
    plt.subplot(311)
    plt.plot(timeline, force, '-k', label='Contact force')
    plt.legend()
    plt.title('Material: {0}, R_a value: {1}'.format(
        attributes[0], 
        config.Ra_dict[attributes[0]][attributes[1]]
        )
    )
    plt.subplot(312)
    plt.plot(timeline, temp, '-r', label='Temperature')
    plt.legend()
    plt.subplot(313)
    plt.plot(timeline, hf, 'b-', label='Heat flux')
    plt.legend()
    
    print(max(material),min(material))
    print(max(Ra),min(Ra))
    print('='*60)
    plt.show()

