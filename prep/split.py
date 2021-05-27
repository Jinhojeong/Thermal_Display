import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import datetime
import os
from tqdm import tqdm 
from config import config


def split_seq(data_, label_, name_, n_steps=config.n_steps, save=True):
        nf , nl= np.shape(data_)[1], np.shape(label_)[1]
        x_split = np.array([]).reshape(-1, n_steps, nf)
        y_split = np.array([]).reshape(-1, nl)
        for i in tqdm(range(np.shape(data_)[0]-n_steps+1)):
            data_1l = data_[i:i+n_steps,:].reshape(1, n_steps, nf)
            label_1l = label_[i+n_steps-1,:].reshape(1, nl)
            x_split = np.concatenate((x_split, data_1l), axis=0)
            y_split = np.concatenate((y_split, label_1l), axis=0)
        print(np.shape(x_split), np.shape(y_split))
        if save:
            with open('../data/splitted/{0}/data/data_split{1}_{2}'.format(n_steps, n_steps, name_), 'wb') as file1:
                np.save(file1, x_split)
            with open('../data/splitted/{0}/label/label_split{1}_{2}'.format(n_steps, n_steps, name_), 'wb') as file2:
                np.save(file2, y_split)
        else:
            return x_split, y_split


for npy_name in os.listdir(config.npy_dest_dir):
    raw_data = np.load(config.npy_dest_dir+npy_name)
    data_ = raw_data[:-1,1:]
    label_ = raw_data[1:,1:3]
    split_seq(data_, label_, npy_name[8:])