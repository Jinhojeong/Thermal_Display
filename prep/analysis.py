import os
import numpy as np
from config import config
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

frontal = True
material_info = True
cut_length = 500


if material_info == True:
    f_n = 6
else:
    f_n = 5


dict_ = {}
for material_name in config.material_str_list:
    dict_[material_name] = np.array([]).reshape(cut_length, f_n, -1)

for npy_name in os.listdir(config.npy_dest_dir):
    raw_data = np.load(config.npy_dest_dir+npy_name)
    attributes = npy_name.split("_")
    attributes[-1] = attributes[-1].split(".")[0]
    del attributes[0]

    if material_info == True:
        pass
    else:
        raw_data = np.delete(raw_data, 4, 1) #material number deleted

    if frontal == True:
        raw_data = raw_data[:cut_length, :].reshape(cut_length, f_n,1)
    elif frontal == False:
        raw_data = raw_data[-cut_length:, :].reshape(cut_length, f_n,1)

    dict_[attributes[0]] = np.concatenate(
        (dict_[attributes[0]], raw_data),
        axis=2
    )

plt.figure()
for idx, feature in enumerate(config.features):
    temp_ = np.array([]).reshape(-1, cut_length)
    label_ = np.array([]).reshape(-1, 1)
    for material_name in config.material_str_list:
        data_ = dict_[material_name][:, idx+1,:] # ignoring timestamp
        
        num_serial = np.shape(data_)[1]
        data_ = data_.reshape(num_serial, -1)
        temp_ = np.concatenate(
            (temp_, data_),
            axis=0
        )
        label_ = np.concatenate(
            (label_, np.ones((num_serial,1))*config.materials[material_name]),
            axis=0
        )
    # pca = PCA(n_components=2)
    pca = KernelPCA(n_components=2, kernel='linear')
    X_r = pca.fit(temp_).transform(temp_)

    
    plt.subplot(3,2,idx+1)
    for material_name in config.material_str_list:
        indices = label_==config.materials[material_name]
        plt.scatter(
            X_r[indices[:,0], 0],
            X_r[indices[:,0], 1],
            label=material_name
        )
    plt.title(feature)
plt.legend(loc=(1.5,-0))
plt.show()
        
