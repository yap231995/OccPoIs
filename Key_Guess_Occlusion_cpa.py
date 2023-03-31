import os
import pickle
import sys
from copy import deepcopy

import sklearn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge, lars_path
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import concurrent.futures




#Create own local dataset.
from tqdm import tqdm

from Key_Guess_Occlusion import backward_occulsion_input_given_pts
from src.cpa import cpa_method, aes_label_cpa, plaintext_label_cpa_cw, Hamming_Distance, HD_cpa_cw
from src.snr import second_order_combine_multiplication
from tf_run import load_ascad, load_chipwhisperer, attack_calculate_metrics, perform_attacks, load_aes_hd_ext, AES_Sbox, \
    AES_Sbox_inv




if __name__ == "__main__":

    np.random.seed(seed=100)
    root = './'

    dataset = 'Chipwhisperer' #'Chipwhisperer' #ASCAD #ASCAD_k0 #ASCAD_variable
    leakage = 'ID'
    iterative_type = "random" #forward #backward #full #random
    save_root = './Result/' + dataset + '_CNN_' + leakage+'/'
    print(tf.__version__)
    save_root_occlusion = save_root + 'key_guessing_occlusion/'
    save_root_occlusion_iterative_type = save_root_occlusion + iterative_type + '/'


    print("PATH save_root:", save_root_occlusion_iterative_type)
    image_root = save_root_occlusion_iterative_type + 'image/'
    attack_root = save_root_occlusion_iterative_type + 'attack/'

    if not os.path.exists(save_root_occlusion):
        os.mkdir(save_root_occlusion)
    if not os.path.exists(save_root_occlusion_iterative_type):
        os.mkdir(save_root_occlusion_iterative_type)
    if not os.path.exists(image_root):
        os.mkdir(image_root)
    if not os.path.exists(attack_root):
        os.mkdir(attack_root)

    seed = 0
    random_state = check_random_state(seed)
    nb_traces_attacks = 2000 #5000 for ASCAD, 2000 for CW
    nb_attacks = 100
    byte = 2
    if dataset == 'ASCAD':
        data_root = 'Dataset/ASCAD/ASCAD.h5'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_ascad(
            root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
            test_end=nb_traces_attacks)

    elif dataset == 'ASCAD_variable':
        data_root = 'Dataset/ASCAD/ASCAD_variable.h5'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_ascad(
            root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
            test_end=nb_traces_attacks)
    elif dataset == 'ASCAD_k0':
        data_root = 'Dataset/ASCAD/ASCAD_k0.h5'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_ascad(
            root + data_root, leakage_model=leakage, byte=0, train_begin=0, train_end=45000, test_begin=0,
            test_end=nb_traces_attacks)

    elif dataset == 'Chipwhisperer':
        data_root = 'Dataset/Chipwhisperer/'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_chipwhisperer(
            root + data_root + '/', leakage_model=leakage,train_begin = 0,train_end = 8000, test_begin = 8000,test_end  = 10000)
    elif dataset == 'AES_HD_ext':
        data_root = 'Dataset/AES_HD_ext/aes_hd_ext.h5'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_aes_hd_ext(root +data_root,leakage_model=leakage,train_begin=0, train_end=45000, test_begin=0,
            test_end=nb_traces_attacks)

    print("The dataset we using: ", data_root)
    # load the data and normalize it

    print("Number of X_attack used: ", X_attack.shape)
    if leakage == 'HW':
        print("Number of Y_attack used: ", len(Y_attack))
    elif leakage == 'ID':
        print("Number of Y_attack used: ", Y_attack.shape)
    input_length = len(X_profiling[0])
    print('Input length: {}'.format(input_length))




    if leakage == 'ID':
        print('ID leakage model')
        classes = 256
        print(save_root + '/' + dataset + '_' + leakage + '_CNN' + '.h5')
        model = load_model(save_root + '/' + dataset + '_' + leakage + '_CNN' + '.h5', compile=False)
    elif leakage == 'HW':
        print('HW leakage model')
        classes = 9
        model = load_model(save_root + '/' + dataset + '_' + leakage + '_CNN' + '.h5', compile=False)
    else:
        print("Error: incorrect leakage model")
        sys.exit(-1)

    model.summary()
    check_load_properly_model = False
    if check_load_properly_model == True:
        scaler = StandardScaler()
        X_profiling = scaler.fit_transform(X_profiling)
        X_attack = scaler.transform(X_attack)
        X_profiling_inside = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
        X_attack_inside = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
        print(X_attack_inside.shape)
        print(X_profiling_inside.shape)
        container = attack_calculate_metrics(model,X_attack=X_attack_inside, Y_attack = Y_attack, plt_attack=plt_attack,leakage= leakage, nb_attacks=nb_attacks, nb_traces_attacks=nb_traces_attacks,
                                            correct_key=correct_key, dataset=dataset)
        # plot GE
        plt.plot(container[257:])
        print(container[257:])

        plt.show()


    #START:
    # attack_type = "cpa"
    # X_more = np.concatenate((X_attack, X_profiling), axis = 0)
    X_more = X_attack
    scaler = StandardScaler()
    X_more = scaler.fit_transform(X_more)
    print("X_more", X_more.shape)
    x = X_more.reshape((X_more.shape[0], X_more.shape[1], 1))
    print("x", x.shape)
    plt_text = plt_attack #plt_profiling #plt_attack
    important_samplept_further = np.load(save_root_occlusion_iterative_type+"important_samplept_further.npy", ) #important_samplept_further
    print("correct key: ", correct_key)
    print("important_samplept_further: ", important_samplept_further)
    print("important_samplept_further", important_samplept_further.shape)
    important_samplept_further = sorted(important_samplept_further, reverse = False)
    cut_out_trace = deepcopy(x)[:,important_samplept_further,:]
    #cut_out_trace = deepcopy(x) #original
    cut_out_trace = cut_out_trace.squeeze()
    print("cut_out_trace", cut_out_trace.shape)
    part_1 = True
    part_2 = False
    if dataset == "Chipwhisperer":
        if part_1 == True:
            save_cpa = False
            label_type = "plaintext" #Sbox #plaintext
            if save_cpa == True:
                if len(cut_out_trace.shape) == 2:
                    total_samplept = cut_out_trace.shape[1]
                elif len(cut_out_trace.shape) == 1:
                    total_samplept = 1
                cpa = np.zeros((256,total_samplept,cut_out_trace.shape[0]))
                for k in range(256):
                    print("key:", k)
                    if label_type == "Sbox":
                        label = aes_label_cpa(plt_text, k, leakage)
                    elif label_type == "plaintext":
                        label = plaintext_label_cpa_cw(plt_text,k,leakage)
                    for number_of_traces in tqdm(range(0, cut_out_trace.shape[0])):
                        cpa[k, :, number_of_traces] = cpa_method(total_samplept, number_of_traces, label, cut_out_trace)

                np.save(attack_root + "cpa_key_trace_attack_traces_"+label_type+".npy",cpa)
            else:
                if label_type == "plaintext":
                    cpa = np.load(attack_root + "cpa_key_trace_attack_traces_"+label_type+".npy") #(key, sample points , number of traces)
                else:
                    cpa = np.load(attack_root + "cpa_key_trace_attack_traces.npy")
            print("cpa_spt_trace: ", cpa.shape)
            #1. cpa over sample point (for fixed number of trace, take the largest)
            fig, ax = plt.subplots(figsize=(15, 7))

            x_axis = [i for i in range(0, cpa.shape[1])]
            flag = False
            max_value = 0
            for k in range(256):
                number_of_samples = cpa.shape[1]

                cpa_vs_samplept = cpa[k,:,cpa.shape[2]-1]

                print("key: ", k)
                print("cpa_vs_samplept: ", cpa_vs_samplept)

                if k != correct_key:
                    if flag == True:
                        ax.plot(x_axis, cpa_vs_samplept, color="grey")
                    else:
                        ax.plot(x_axis, cpa_vs_samplept, label="wrong keys", color="grey")
                        flag = True
            ax.plot(x_axis,  cpa[correct_key,:,cpa.shape[2]-1], label="correct key", color="r")
            plt.ylabel("(Absolute) Correlation", fontsize=20)
            plt.xlabel("Samples", fontsize=20)
            # plt.xlim([0,0.5])
            ax.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0, prop={'size': 20})
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(15)
            if label_type == "plaintext":
                plt.savefig(attack_root+'/CPA_samplepoint_' + leakage + '_' + dataset + '_attack_traces_'+label_type+'.png')
            else:
                plt.savefig(attack_root+'/CPA_samplepoint_' + leakage + '_' + dataset + '_attack_traces.png')
            # plt.savefig(attack_root+'/CPA_samplepoint_' + leakage + '_' + dataset + '_original.png')
            plt.cla()

            #2. cpa over traces (largest correlation)
            fig, ax = plt.subplots(figsize=(15, 7))
            flag = False
            max_cpa = np.max(cpa, axis = 1)

            x_axis = [i for i in range(0, max_cpa.shape[1])]
            print(max_cpa.shape)
            for k in range(256):
                number_of_traces = max_cpa.shape[1]

                cpa_vs_time = max_cpa[k,:]

                if k != correct_key:

                    if flag == True:
                        ax.plot(x_axis, cpa_vs_time, color="grey")
                    else:
                        ax.plot(x_axis, cpa_vs_time, label="wrong keys", color="grey")
                        flag = True
            ax.plot(x_axis, max_cpa[correct_key,:], label="correct key", color="r")
            plt.ylabel("(Absolute) Correlation", fontsize=20)
            plt.xlabel("Number of traces", fontsize=20)

            ax.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0, prop={'size': 20})
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(15)
            if label_type == "plaintext":
                plt.savefig(attack_root+'/CPA_num_of_traces_' + leakage + '_' + dataset + '_attack_traces_'+label_type+'.png')
            else:
                plt.savefig(attack_root+'/CPA_num_of_traces_' + leakage + '_' + dataset + '_attack_traces.png')
        if part_2 == True:
            print("Part2: To use another leakage model (HW) -> (ID), (ID)-> (HW)")
            save_cpa = False
            label_type = 'plaintext'
            if leakage == 'ID':
                tried_leakage = 'HW'
            elif leakage == 'HW':
                tried_leakage = 'ID'
            # tried_leakage = 'LSB' #MSB, LSB
            if save_cpa == True:
                if len(cut_out_trace.shape) == 2:
                    total_samplept = cut_out_trace.shape[1]
                elif len(cut_out_trace.shape) == 1:
                    total_samplept = 1
                cpa = np.zeros((256, total_samplept, cut_out_trace.shape[0]))

                for k in range(256):
                    print("key:", k)
                    if label_type == "Sbox":
                        label = aes_label_cpa(plt_text, k, tried_leakage)
                    elif label_type == "plaintext":
                        label = plaintext_label_cpa_cw(plt_text,k,tried_leakage)
                        # np.set_printoptions(threshold=sys.maxsize)
                        # print("label:", label)
                    for number_of_traces in tqdm(range(100, cut_out_trace.shape[0])):
                        # print("number_of_traces:", number_of_traces)
                        cpa[k, :, number_of_traces] = cpa_method(total_samplept, number_of_traces, label, cut_out_trace)
                if label_type == "Sbox":
                    np.save(attack_root + "cpa_key_trace_actual_"+leakage+"_tried_"+tried_leakage+".npy", cpa)
                elif label_type == "plaintext":
                    np.save(attack_root + "cpa_key_trace_actual_" + leakage + "_tried_" + tried_leakage + "_" + label_type + ".npy", cpa)
            else:
                if label_type == "Sbox":
                    cpa = np.load(attack_root + "cpa_key_trace_actual_"+leakage+"_tried_"+tried_leakage+".npy")  # (key, sample points , number of traces)
                elif label_type == "plaintext":
                    cpa = np.load(attack_root + "cpa_key_trace_actual_" + leakage + "_tried_" + tried_leakage  + "_" + label_type + ".npy")
                # cpa = np.load(attack_root + "cpa_key_trace_original.npy") #(key, sample points , number of traces)
            print("cpa_spt_trace: ", cpa.shape)
            # 1. cpa over sample point (for fixed number of trace, take the largest)
            fig, ax = plt.subplots(figsize=(15, 7))

            x_axis = [i for i in range(0, cpa.shape[1])]
            flag = False
            max_value = 0
            for k in range(256):
                number_of_samples = cpa.shape[1]

                cpa_vs_samplept = cpa[k, :, cpa.shape[2] - 1]

                print("key: ", k)
                print("cpa_vs_samplept: ", cpa_vs_samplept)

                if k != correct_key:
                    if flag == True:
                        ax.plot(x_axis, cpa_vs_samplept, color="grey")
                    else:
                        ax.plot(x_axis, cpa_vs_samplept, label="wrong keys", color="grey")
                        flag = True
            ax.plot(x_axis, cpa[correct_key, :, cpa.shape[2] - 1], label="correct key", color="r")
            plt.ylabel("(Absolute) Correlation", fontsize=20)
            plt.xlabel("Samples", fontsize=20)
            # plt.xlim([0,0.5])
            ax.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0, prop={'size': 20})
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(15)
            if label_type == "Sbox":
                plt.savefig(attack_root + '/CPA_samplepoint_actual_' + leakage + '_tried_' +tried_leakage+'_'+ dataset + '.png')
            elif label_type == "plaintext":
                plt.savefig(attack_root + '/CPA_samplepoint_actual_' + leakage + '_tried_' +tried_leakage+'_'+ dataset + '_' + label_type+'.png')
            plt.cla()

            # 2. cpa over traces (largest correlation)
            fig, ax = plt.subplots(figsize=(15, 7))
            flag = False
            max_cpa = np.max(cpa, axis=1)

            x_axis = [i for i in range(0, max_cpa.shape[1])]
            print(max_cpa.shape)
            for k in range(256):
                number_of_traces = max_cpa.shape[1]

                cpa_vs_time = max_cpa[k, :]

                if k != correct_key:
                    if flag == True:
                        ax.plot(x_axis, cpa_vs_time, color="grey")
                    else:
                        ax.plot(x_axis, cpa_vs_time, label="wrong keys", color="grey")
                        flag = True
            ax.plot(x_axis,  max_cpa[correct_key, :], label="correct key", color="r")
            plt.ylabel("(Absolute) Correlation", fontsize=20)
            plt.xlabel("Number of traces", fontsize=20)
            # plt.xlim([0, number_of_traces])
            ax.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0, prop={'size': 20})
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(15)
            if label_type == "Sbox":
                plt.savefig(attack_root + '/CPA_num_of_traces_actual_' + leakage + '_tried_' +tried_leakage+'_'+ dataset + '.png')
            elif label_type == "plaintext":
                plt.savefig(attack_root + '/CPA_num_of_traces_actual_' + leakage + '_tried_' + tried_leakage + '_' + dataset + '_'+label_type+'.png')
            plt.show()

