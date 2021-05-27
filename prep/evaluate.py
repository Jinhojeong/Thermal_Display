import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import datetime
import os
from tqdm import tqdm
from config import config

#'LSTM40_2hidden_tanhtanh_6432_201109'

date = 201130
model_name = '10_2LSTM4020_4hidden_tanhs_641286432_sgfon_201116'

model = tf.keras.models.load_model(
    './models/' + model_name
)

benchmark_name = 'rawdata_Al_400_2.npy'
benchmark_data = np.load(config.npy_dest_dir+benchmark_name)

if config.eval_windowing:
    init_data = benchmark_data[0,1:].reshape(1,1,config.num_features)
    benchmark_traj = np.repeat(init_data, config.n_steps, axis=1)
    extp_size = 1
    outputs_ = benchmark_data[0,1:3].reshape(-1,2)
else:
    benchmark_traj = benchmark_data[:config.n_steps,1:].reshape(1,config.n_steps,config.num_features)
    extp_size = config.n_steps
    outputs_ = benchmark_data[:config.n_steps,1:3].reshape(-1,2)
    

timesteps = np.shape(benchmark_data)[0]
for idx, item in enumerate(config.scaler):
    benchmark_traj[:,:,idx] = benchmark_traj[:,:,idx]/item

for t in tqdm(range(timesteps-extp_size)):
    
    predictions = model.predict(
        benchmark_traj
        )[0]
    

    pre_part = benchmark_traj[:,1:,:]
    post_part = np.array([predictions[0], predictions[1], benchmark_data[t+extp_size,3]/config.scaler[2], benchmark_data[t+extp_size,4]/config.scaler[3], benchmark_data[t+extp_size,5]/config.scaler[4]]).reshape(1,1,np.shape(benchmark_traj)[2])
    benchmark_traj = np.concatenate((pre_part, post_part), axis=1)
    outputs_ = np.vstack((outputs_, predictions))

x_axis = benchmark_data[:,0][:timesteps]
true_hf = benchmark_data[:,1][:timesteps]
true_temp = benchmark_data[:,2][:timesteps]
est_hf = outputs_[:,0]*config.scaler[0]
est_temp = outputs_[:,1]*config.scaler[1]
plt.figure()
plt.subplot(211)
plt.plot(x_axis, true_hf, 'b-', label='True heat flux')
plt.plot(x_axis, est_hf, 'g--', label='Estimated heat flux')
plt.legend()
plt.subplot(212)
plt.plot(x_axis, true_temp, 'r-', label='True temperature')
plt.plot(x_axis, est_temp, 'k--', label='Estimated temperature')
plt.legend()
plt.savefig('../figure/{0}/{1}_{2}_extp{3}.png'.format(date, benchmark_name[:-4], model_name, extp_size), dpi=300)
plt.show()
