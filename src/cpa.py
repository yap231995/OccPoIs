import itertools
import h5py
import numpy as np
from tqdm import tqdm

from tf_run import AES_Sbox, AES_Sbox_inv

def second_order_combine_multiplication(number_of_traces,X_profiling):
    comb_spt_lst = list(itertools.combinations([i for i in range(X_profiling.shape[1])],2))

    total_samplept =len(comb_spt_lst)
    traces_combine = np.zeros((number_of_traces, total_samplept))
    print("Total number of sample points:", total_samplept)
    print("Multiply each points with each points")
    for idx, spts in enumerate(comb_spt_lst):
        # print(idx,spts)
        if idx %10 == 0:
            print("Sample points: {}/{}".format(idx,total_samplept))
        pt1 = X_profiling[:number_of_traces, spts[0]]
        pt2 = X_profiling[:number_of_traces, spts[1]]
        traces_combine[:, idx] = pt1 * pt2 #not sure the indexing

    return traces_combine,total_samplept

def HW(s):
    return bin(s).count("1")
def HD(s):
    pass #(????)
def LSB(s):
    return s& 0x1
def MSB(s):
    return (s>>7)& 0x1

def aes_label_with_mask_cpa(plt,correct_key,leakage,dataset, root,byte):
    in_file = h5py.File(root, "r")
    masks_profiling = np.array(in_file['Profiling_traces/metadata'][:]['masks'])[:plt.shape[0], :]

    if dataset == "ASCAD":
        if byte == 0 or byte == 1:
            masks_values = np.zeros((masks_profiling.shape[0],))
        else:
            mask_values = masks_profiling[:, byte-2]
    elif dataset == "ASCAD_variable":
        print("MASK FOR ASCAD_variable: ")
        print(masks_profiling)
        mask_values = masks_profiling[:,byte]
    num_traces = plt.shape[0]
    labels_for_snr = np.zeros(num_traces)
    for i in range(num_traces):
        if leakage == 'HW':
            labels_for_snr[i] = HW(AES_Sbox[plt[i] ^ correct_key] ^ mask_values[i])
            mask_values[i]= HW(mask_values[i])
        else:
            labels_for_snr[i] = AES_Sbox[plt[i] ^ correct_key] ^ mask_values[i]

    return labels_for_snr, mask_values

def aes_label_cpa(plt,correct_key,leakage):

    num_traces = plt.shape[0]
    labels_for_snr = np.zeros(num_traces)
    for i in range(num_traces):
        if leakage == 'HW':
            labels_for_snr[i] = HW(AES_Sbox[plt[i] ^ correct_key])
        elif leakage == 'ID':
            labels_for_snr[i] = AES_Sbox[plt[i] ^ correct_key]
        elif leakage == 'LSB':
            labels_for_snr[i] = LSB(AES_Sbox[plt[i] ^ correct_key])
        elif leakage == 'MSB':
            labels_for_snr[i] = MSB(AES_Sbox[plt[i] ^ correct_key])
    return labels_for_snr

def plaintext_label_cpa_cw(plt,k,leakage):
        num_traces = plt.shape[0]
        labels_for_pt = np.zeros(num_traces)
        for i in range(num_traces):
            if leakage == 'HW':

                labels_for_pt[i] = HW(plt[i] ^ k)
            elif leakage == 'ID':
                labels_for_pt[i] = plt[i]^k
            elif leakage == 'LSB':
                # print("plt", plt[i])
                # print("k", k)
                # print("plt[i] ^ k", plt[i] ^ k)
                labels_for_pt[i] = LSB(plt[i] ^ k)
                # print("labels_for_pt[i]", labels_for_pt[i])
            elif leakage == 'MSB':
                labels_for_pt[i] = MSB(plt[i] ^ k)
        return labels_for_pt


def HD_cpa_cw(plt, k, leakage):
    num_traces = plt.shape[0]
    labels_for_pt = np.zeros(num_traces)
    for i in range(num_traces):
        if leakage == 'HW':
            print("Not the right leakage model: HW (should be ID)")
            labels_for_pt[i] = HW(plt[i] ^ k)
        elif leakage == 'ID':
            # labels_for_pt[i] = HW(plt[i])
            labels_for_pt[i] = HW(plt[i]^k)
    return labels_for_pt

def plaintext_label_cpa_AES_HD(plt, k, leakage):
    num_traces = plt.shape[0]
    labels_for_pt = np.zeros(num_traces)
    for i in range(num_traces):
        if leakage == 'HW':
            labels_for_pt[i] = HW(int(plt[i, 15]) ^ k)
        elif leakage == 'ID':
            labels_for_pt[i] = int(plt[i, 15]) ^ k
    return labels_for_pt

def aes_label_cpa_AES_HD(plt, k, leakage):
    num_traces = plt.shape[0]
    labels_for_pt = np.zeros(num_traces)
    for i in range(num_traces):
        if leakage == 'HW':
            labels_for_pt[i] = HW(AES_Sbox_inv[k ^ int(plt[i, 15])] ^ plt[i, 11])
        elif leakage == 'ID':
            labels_for_pt[i] = AES_Sbox_inv[k ^ int(plt[i, 15])] ^ plt[i, 11]
    return labels_for_pt


def cpa_method(total_samplept,number_of_traces, label, traces):
    if total_samplept == 1:
        cpa = np.zeros(total_samplept)
        cpa[0] = abs(np.corrcoef(label[:number_of_traces], traces[:number_of_traces])[1, 0])
    else:
        cpa = np.zeros(total_samplept)
        print("Calculate CPA")
        for t in tqdm(range(total_samplept)):
            cpa[t] = abs(np.corrcoef(label[:number_of_traces], traces[:number_of_traces, t])[1, 0])
    return cpa


def online_cpa_one_shot(x, y, number_of_traces, total_samplept):
    bar_x_n = np.zeros(total_samplept)
    bar_x_n_1 = np.zeros(total_samplept)
    bar_y_n = 0
    var_x = np.zeros(total_samplept)
    cov_x_y = np.zeros(total_samplept)
    for trace_index in tqdm(range(number_of_traces)):
        y_i = y[trace_index]
        bar_y_n = bar_y_n + (y_i - bar_y_n) / (trace_index + 1)  ##sum y_i
        for spt in range(total_samplept):
            spt_value = x[trace_index, spt]
            bar_x_n[spt] = bar_x_n[spt] + (spt_value - bar_x_n[spt]) / (trace_index + 1)  ##sum x_i
            var_x[spt] = var_x[spt] + (spt_value - bar_x_n_1[spt]) * (spt_value - bar_x_n[spt])
            cov_x_y[spt] = cov_x_y[spt] + (spt_value - bar_x_n_1[spt]) * (y_i - bar_y_n)
        bar_x_n_1 = bar_x_n

    var_y = y.shape[0] * np.var(y)
    cpa = abs(cov_x_y / (np.sqrt(var_x) * np.sqrt(var_y)))
    return cpa

def online_cpa_each_trace(x, y, number_of_traces, total_samplept):
    bar_x_n = np.zeros(total_samplept)
    bar_x_n_1 = np.zeros(total_samplept)
    bar_y_n = 0
    var_x = np.zeros(total_samplept)
    cov_x_y = np.zeros(total_samplept)
    cpa = np.zeros((number_of_traces, total_samplept))
    for trace_index in tqdm(range(number_of_traces)):
        y_i = y[trace_index]
        bar_y_n = bar_y_n + (y_i - bar_y_n) / (trace_index + 1)  ##sum y_i
        for spt in range(total_samplept):
            spt_value = x[trace_index, spt]
            bar_x_n[spt] = bar_x_n[spt] + (spt_value - bar_x_n[spt]) / (trace_index + 1)  ##sum x_i
            var_x[spt] = var_x[spt] + (spt_value - bar_x_n_1[spt]) * (spt_value - bar_x_n[spt])
            cov_x_y[spt] = cov_x_y[spt] + (spt_value - bar_x_n_1[spt]) * (y_i - bar_y_n)
        bar_x_n_1 = bar_x_n
        var_y = trace_index * np.var(y[:trace_index])
        cpa[trace_index, :] = abs(cov_x_y / (np.sqrt(var_x) * np.sqrt(var_y)))
    return cpa



def Hamming_Distance(list1,list2):
    assert list1.shape == list2.shape
    new_lst = np.zeros(list1.shape)
    for i in range(list1.shape[0]):
        # print("list1[i]", list1[i])
        # print("list2[i]", list2[i])
        new_lst[i] = HW(int(list1[i])^int(list2[i]))
        # print("new_lst[i]", new_lst[i])
    return new_lst
