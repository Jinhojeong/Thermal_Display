import os
import numpy as np
from config import config
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from matplotlib import rc_params

cm = plt.get_cmap('gist_rainbow')

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

# print(dict_['Balsa'][:, 5, :])
# print(np.shape(dict_['Balsa'][:, 5, :]))

# fig = plt.figure()

# for idx, feature in enumerate(config.features):
#     temp_ = np.array([]).reshape(-1, cut_length)
#     label_ = np.array([]).reshape(-1, 1)
#     for material_name in config.material_str_list:
#         data_ = dict_[material_name][:, idx+1,:] # ignoring timestamp
        
#         num_serial = np.shape(data_)[1]
#         data_ = data_.reshape(num_serial, -1)
#         temp_ = np.concatenate(
#             (temp_, data_),
#             axis=0
#         )
#         label_ = np.concatenate(
#             (label_, np.ones((num_serial,1))*config.materials[material_name]),
#             axis=0
#         )
#     # pca = PCA(n_components=2)
#     pca = KernelPCA(n_components=3, kernel='linear')
#     X_r = pca.fit(temp_).transform(temp_)

    
#     ax = fig.add_subplot(3,2,idx+1, projection='3d')
#     for material_name in config.material_str_list:
#         indices = label_==config.materials[material_name]
        
#         ax.scatter(
#             X_r[indices[:,0], 0],
#             X_r[indices[:,0], 1],
#             X_r[indices[:,0], 2],
#             label=material_name
#         )

        
#     plt.title(feature)


# plt.legend(loc=(1.5,-0))
# plt.show()

feature_list = ['heat_flux', 'temperature']


fig = plt.figure()

for idx, feature in enumerate(feature_list):
    temp_ = np.array([]).reshape(-1, cut_length, 3)
    label_ = np.array([]).reshape(-1, 1)
    for material_name in config.material_str_list:
        data_ = dict_[material_name][:, idx+1, :]
        
        num_serial = np.shape(data_)[1]
        data_ = data_.T.reshape(num_serial, -1, 1)
        data_r = dict_[material_name][:, 5, :]
        data_r = data_r.T.reshape(num_serial, -1, 1)
        data_f = dict_[material_name][:, 3, :]
        data_f = data_r.T.reshape(num_serial, -1, 1)
        data_ = np.concatenate(
            (data_, data_r, data_f),
            axis=2
        )
        temp_ = np.concatenate(
            (temp_, data_),
            axis=0
        )
        label_ = np.concatenate(
            (label_, np.ones((num_serial,1))*config.materials[material_name]),
            axis=0
        )
    pca = KernelPCA(n_components=3, kernel='linear')
    X_r = pca.fit(temp_[:,:,0]).transform(temp_[:,:,0])

    ax = fig.add_subplot(1,2,idx+1, projection='3d')
    ax.set_prop_cycle(
        color=[cm(1.*i/len(config.material_str_list)) for i in range(len(config.material_str_list))],
        marker=['o', '^', 's','o', '^', 's','o', '^', 's','o', '^', 's','o', '^', 's','o', '^',]
    )
    for material_name in config.material_str_list:
        indices = label_==config.materials[material_name]
        r = temp_[:,:,1][indices[:,0]][:,0]
        f = np.amax(
            temp_[:,:,2][indices[:,0]], 
            axis=1
        )

        # ax.plot(
        #     X_r[indices[:,0], 0],
        #     X_r[indices[:,0], 1],
        #     X_r[indices[:,0], 2],
        #     label=material_name,
        #     linestyle='None'
        # )
        ax.scatter(
            X_r[indices[:,0], 0],
            X_r[indices[:,0], 1],
            X_r[indices[:,0], 2],
            s=f**1.0,
            label=material_name,
        )

        
    plt.title(feature)

plt.legend(loc=(1.0,-0.2))
plt.show()