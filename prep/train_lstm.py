import tensorflow as tf
import numpy as np
from scipy.signal import savgol_filter
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import datetime
import os
from tqdm import tqdm
from config import config

timesteps = 500

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def input_normalizer(data_wo_t):
    for i in range(config.num_features):
        if data_wo_t.ndim == 3:
            data_wo_t[:,:,i] = data_wo_t[:,:,i]/config.scaler[i]
        elif data_wo_t.ndim == 2:
            data_wo_t[:,i] = data_wo_t[:,i]/config.scaler[i]
        else:
            print('error')
    return data_wo_t

def output_normalizer(label_wo_t):
    scale_list = config.scaler[:2]
    for i in range(2):
        label_wo_t[:,i] = label_wo_t[:,i]/scale_list[i]
    return label_wo_t

n_steps = config.n_steps
ep = config.ep





# data_ = tf.data.Dataset.from_tensor_slices((data_))
# label_ = tf.data.Dataset.from_tensor_slices((label_))

# kf = KFold(n_splits=5)
# for train_index, test_index in kf.split(data_):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     x_train, x_test = data_[train_index], data_[test_index]
#     y_train, y_test = label_[train_index], label_[test_index]


###
model1 = tf.keras.Sequential([
    tf.keras.layers.LSTM(40, return_sequences=True, input_shape=(config.n_steps, config.num_features)),
    tf.keras.layers.LSTM(20),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(32, activation='tanh'),
    tf.keras.layers.Dense(2, activation='linear')
])

model1.compile(optimizer='adam',
              loss= 'mean_squared_error', #'categorical_crossentropy'
              metrics=['accuracy'])

log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_path = "../models/training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
 save_weights_only=True,
 verbose=1,
 period=5)

model1.save_weights(checkpoint_path.format(epoch=1))


for idx, npy_name in enumerate(os.listdir(config.npy_dest_dir)):
    print('>>>>> Data {0} of {1}'.format(
        idx, len(os.listdir(config.npy_dest_dir))
        )
    )
    print(str(npy_name))
    data_ = np.load('../data/splitted/{0}/data/data_split{1}_{2}'.format(n_steps, n_steps, npy_name[8:]))
    label_ = np.load('../data/splitted/{0}/label/label_split{1}_{2}'.format(n_steps, n_steps, npy_name[8:]))
    # print(np.shape(data_))
    # print(np.shape(label_))
    if config.data_smoothing:
        data_ = savgol_filter(data_, 99, 2, mode='nearest')
    else:
        pass
    # for idx, item in enumerate(config.scaler):
    #     data_[:,:,idx] = data_[:,:,idx]/item
    # for idx, item in enumerate(config.scaler[0:2]):
    #     label_[:,idx] = label_[:,idx]/item
    model1.fit(data_, label_, epochs=ep, batch_size=config.batch_size)
model1.save('./models/{0}_2LSTM4020_4hidden_tanhs_641286432_sgfon_201116'.format(config.n_steps))

# model.evaluate(data_, label_, verbose=2)

# def input_fn(features, batch_size=256):
#     return tf.data.Dataset.from_tensor_slices((features)).batch(batch_size)
# bechmark_name = 'data_200925_Al_100_100g_0'
# benchmark = np.load('./segmented/'+bechmark_name+'.npy')
# benchmark_wot = input_normalizer(benchmark[:,1:])
# benchmark_traj = benchmark_wot[0,:].reshape(1,1,config.num_features)
# benchmark_traj = np.repeat(benchmark_traj, n_steps, axis=1)


# outputs_ = np.array([]).reshape(-1,2)

# timesteps = np.shape(benchmark)[0]
# for t in tqdm(range(timesteps)):
    

#     predictions = model.predict(
#         benchmark_traj
#         )[0]
    
#     pre_part = benchmark_traj[:,1:,:]
#     post_part = np.array([predictions[0], predictions[1], benchmark_wot[t,2], benchmark_wot[t,3], benchmark_wot[t,4]]).reshape(1,1,np.shape(benchmark_traj)[2])
#     benchmark_traj = np.concatenate((pre_part, post_part), axis=1)
#     outputs_ = np.vstack((outputs_, predictions))



# x_axis = benchmark[:,0][:timesteps]
# true_hf = benchmark[:,1][:timesteps]*config.scaler[0]
# true_temp = benchmark[:,2][:timesteps]*config.scaler[1]
# est_hf = outputs_[:,0]*config.scaler[0]
# est_temp = outputs_[:,1]*config.scaler[1]
# plt.figure()
# plt.subplot(211)
# plt.plot(x_axis, true_hf, 'b-', label='True heat flux')
# plt.plot(x_axis, est_hf, 'g--', label='Estimated heat flux')
# plt.legend()
# plt.subplot(212)
# plt.plot(x_axis, true_temp, 'r-', label='True temperature')
# plt.plot(x_axis, est_temp, 'k--', label='Estimated temperature')
# plt.legend()
# # plt.savefig('./Figures/LSTM_{0}ep_{1}steps_{2}.png'.format(ep, n_steps, bechmark_name), dpi=300)
# plt.show()

# # plt.figure()
# # plt.plot(benchmark[:,0], benchmark[:,1], 'r')
# # plt.plot(benchmark[:,0], benchmark[:,2], 'b')
# # plt.show()
