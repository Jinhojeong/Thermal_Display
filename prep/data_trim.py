import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from scipy.signal import (
    argrelextrema, argrelmax, argrelmin, find_peaks, peak_prominences, find_peaks_cwt, savgol_filter
)
import pandas as pd
import os, shutil
from config import config


def ms_only(arr_of_str, idx):
    ms = int(arr_of_str[idx][0][-3:-1])
    return ms

def filtering(data):
    ret = savgol_filter(
        data, 
        window_length=9,
        polyorder=5,
        mode='nearest'    
    )
    return ret

def annotate_and_delete(
        ax, 
        data_arr, 
        time_arr, 
        preindexes, 
        textspacing
    ):
    list_ = []
    for i, t_annotated in enumerate(time_arr[preindexes]):
        ax.annotate(
            str(i)+', '+str(t_annotated)+'\n value:'+str(data_arr[preindexes][i]),
            xy = (t_annotated, min(data_arr)),
            xytext = (t_annotated+textspacing, min(data_arr)),
            arrowprops=dict(facecolor='black', shrink=0.02)
        )
        list_.append(str(t_annotated))
    return list_

def secant(xs, 
           ys, 
           width,
    ):
    list_ = []
    for t_idx in range(0, len(xs)-width):
        dt = xs[t_idx+width] - xs[t_idx]
        dy = ys[t_idx+width] - ys[t_idx]
        slope = dy[0]/dt
        list_.append(slope*100)

    list_.extend([0]*width)
    secant_ = np.array(list_)
    return secant_


for xlsx_name in os.listdir(config.rawdata_dir):
    attributes = xlsx_name.split("_")
    del attributes[-1]
    print(attributes)
    
    time_l = pd.read_excel(
        config.rawdata_dir+xlsx_name, 
        usecols=[0], 
        header=[0]
    )
    time_l = np.array(time_l)[1:]
    ms_start = ms_only(time_l, 0)
    t_list = [0]
    for idx in range(1,len(time_l)):
        t_start = t_list[0]
        ms_prev = ms_only(time_l, idx-1)
        ms = ms_only(time_l, idx)
        t_diff = int(ms-ms_prev)
        if t_diff > 0:
            pass
        elif t_diff < 0:
            t_diff = t_diff + 100
        t_list.append(t_list[-1]+t_diff*10)
    t_array = np.asarray(t_list)

    # Heat flux (V)
    heat_flux = pd.read_excel(
        config.rawdata_dir+xlsx_name, 
        usecols=[1], 
        header=[0]
    )
    heat_flux = np.array(heat_flux)[1:]
    heat_flux = np.array(heat_flux, dtype=np.float64)

    # Temperature (Celcius)
    temperature = pd.read_excel(
        config.rawdata_dir+xlsx_name, 
        usecols=[2], 
        header=[0]
    )
    temperature = np.array(temperature)[1:]
    temperature = np.array(temperature, dtype=np.float64) 

    # Arduino Time list 
    time_a = pd.read_excel(
        config.rawdata_dir+xlsx_name, 
        usecols=[4], 
        header=[0]
    )
    time_a = np.array(time_a)[1:]
    time_a = time_a[~np.isnan(time_a)]
    time_a = time_a - time_a[0]
    time_a = np.array(time_a, dtype=np.float64)
    
    # Force
    force = pd.read_excel(
        config.rawdata_dir+xlsx_name, 
        usecols=[5]
    )
    # print(force.dropna().to_numpy()[-1])
    if force.dropna().to_numpy()[-1] == ' ':
        force = force.dropna().to_numpy()[:-1]
    else:
        force = force.dropna().to_numpy()
    force = np.array(force, dtype=np.float64)

    hf_secant = secant(t_array, heat_flux, 10)

    # force_filtered = filtering(force)
    # temperature_filtered = filtering(temperature)
    # hf_filtered = filtering(heat_flux)

    ##force trimming
    force_idx_list = []
    force_bias = 5
    for idx in range(np.shape(force)[0]-1):
        if (force[idx]-force_bias < 0) and (force[idx+1]-force_bias > 0):
            force_idx_list.append(idx)
        else:
            pass


    ##hf trimming
    hf_idx_list = []
    hf_bias = 0.001
    for idx in range(np.shape(heat_flux)[0]-1):
        if (heat_flux[idx]+hf_bias > 0) and (heat_flux[idx+1]+hf_bias < 0):
            hf_idx_list.append(idx)    
        else:
            pass

    ##hf secant trimming
    hf_sec_idx_list = []
    hf_sec_bias = 0.001
    for idx in range(np.shape(hf_secant)[0]-1):
        if (hf_secant[idx]+hf_sec_bias > 0) and (hf_secant[idx+1]+hf_sec_bias < 0):
            hf_sec_idx_list.append(idx)    
        else:
            pass
    

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.plot(time_a, force)
    ax1.vlines(time_a[force_idx_list], min(force), max(force), 'k')
    t_anno_str_list_force = []
    for i, t_anno in enumerate(time_a[force_idx_list]):
        ax1.annotate(
            str(i)+', '+str(t_anno)+
            '\n value:'+str(
                force[force_idx_list][i]
            ), 
            xy=(t_anno, max(force)),
            xytext=(t_anno-2000, 
                    max(force)-(max(force)-min(force))/4
                    ),
            arrowprops=dict(facecolor='black', shrink=0.02),
            fontsize = 7
        )
        t_anno_str_list_force.append(str(t_anno))

    
    ax2 = fig.add_subplot(312)
    ax2.plot(t_array, heat_flux)
    ax2.vlines(t_array[hf_idx_list], min(heat_flux), max(heat_flux), 'k')
    
    ax3 = fig.add_subplot(313)
    ax3.plot(t_array, hf_secant)
    ax3.vlines(t_array[hf_sec_idx_list], min(hf_secant), max(hf_secant), 'k')
    t_anno_str_list_hfsec = []
    for i, t_anno in enumerate(t_array[hf_sec_idx_list]): 
        ax3.annotate(
            str(i)+', '+str(t_anno)+
            '\n value:'+str(
                round(hf_secant[hf_sec_idx_list][i],6)
            ), 
            xy=(t_anno, min(hf_secant)),
            xytext=(t_anno+2000, 
                    min(hf_secant)+(max(hf_secant)-min(hf_secant))/4
                    ),
            arrowprops=dict(facecolor='black', shrink=0.02),
            fontsize = 7
        )
        t_anno_str_list_hfsec.append(str(t_anno))

    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    plt.show()

    print('force: ',t_anno_str_list_force)
    print('hf secant: ',t_anno_str_list_hfsec)

    del_idxs_str_force = input('Enter multiple force indexes to delete, space-separated:')
    if del_idxs_str_force == 'n':
        pass
    else:
        del_idxs_force = list(map(int, del_idxs_str_force.split()))
        count1 = 0 
        for item in del_idxs_force:
            del force_idx_list[item-count1]
            count1+=1

    print('remain force list: ',time_a[force_idx_list])


    del_idxs_str_hfsec = input('Enter multiple heat flux indexes to delete, space-separated:')
    if del_idxs_str_hfsec == 'n':
        pass
    else:
        del_idxs_hfsec = list(map(int, del_idxs_str_hfsec.split()))
        count2 = 0
        for item in del_idxs_hfsec:
            del hf_sec_idx_list[item-count2]
            count2+=1

    print('remain hf_sec list: ',t_array[hf_sec_idx_list])

    if len(force_idx_list) == len(hf_sec_idx_list):
        num_seq = len(force_idx_list) 
        pass
    else:
        print('len(force_idx_list)({0}) and len(hf_sec_idx_list)({1}) does not match'.format(len(force_idx_list), len(hf_sec_idx_list)))
        exit(0)
    
    for idx in range(num_seq):
        if idx == num_seq-1:
            pass
        else:
            time_a_single_seq = time_a[
                force_idx_list[idx]:force_idx_list[idx+1]
            ] - time_a[force_idx_list[idx]]
            force_single_seq = force[
                force_idx_list[idx]:force_idx_list[idx+1]
            ]
            t_array_single_seq = t_array[
                hf_sec_idx_list[idx]:hf_sec_idx_list[idx+1]
            ] - t_array[hf_sec_idx_list[idx]]
            hf_single_seq = heat_flux[
                hf_sec_idx_list[idx]:hf_sec_idx_list[idx+1]
            ]
            temp_single_seq = temperature[
                hf_sec_idx_list[idx]:hf_sec_idx_list[idx+1]
            ]
            multiplier = t_array_single_seq[-1]/time_a_single_seq[-1]

            time_a_single_seq = time_a_single_seq*multiplier

            # print(np.shape(time_a_single_seq))
            # print(np.shape(force_single_seq))

            f = interpolate.interp1d(
                np.squeeze(time_a_single_seq), 
                np.squeeze(force_single_seq),
                kind='linear', 
                fill_value='extrapolate'
            )
            force_single_seq_new = f(t_array_single_seq)

            data = np.hstack((
                t_array_single_seq.reshape(-1,1),
                hf_single_seq.reshape(-1,1),
                temp_single_seq.reshape(-1,1),
                force_single_seq_new.reshape(-1,1),
                config.materials[attributes[1]]
                *np.ones_like(temp_single_seq).reshape(-1,1),
                config.Ra_dict[attributes[1]][attributes[2]]
                *np.ones_like(temp_single_seq).reshape(-1,1)
            ))

            with open(config.npy_dest_dir+'rawdata_{0}_{1}_{2}.npy'.format(attributes[1], attributes[2], idx), 'wb') as file:
                np.save(file, data)

    shutil.move(config.rawdata_dir+xlsx_name, config.rawdata_dir+'../done/')

    