import itertools
import math
import os
import sys
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import tensorflow as tf
from tensorflow.keras.models import load_model

from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *

from tqdm import tqdm
from Key_Guess_Occlusion import cal_NTGE
from src.cpa import cpa_method
from src.net import cnn_ascad_fix_ID, cnn_ascad_variable_ID
from tf_run import load_ascad, load_chipwhisperer, attack_calculate_metrics, load_aes_hd_ext, AES_Sbox, \
    AES_Sbox_inv, rk_key, load_chipwhisperer_desync
import random



def SOSD_SOST(data, labels,classes,num_features_wants, method):
    mean_class = []
    var_class = []
    count_class = []
    labels = np.array(labels)
    for class_index in range(classes):
        traces_class = data[np.where(labels== class_index)[0]]
        if traces_class.shape[0] != 0:
            mean_class.append(np.mean(traces_class, axis=0))
            count_class.append(traces_class.shape[0])
            if method == 'SOST':
                var_class.append(np.var(traces_class, axis=0))
        else:
            mean_class.append(np.zeros(data.shape[1]))
            count_class.append(0)
            if method == 'SOST':
                var_class.append(np.ones(data.shape[1]))

    mean_class = np.array(mean_class)
    var_class = np.array(var_class)
    count_class = np.array(count_class)
    # print("mean_class:", mean_class)
    # np.set_printoptions(threshold=sys.maxsize)
    print("var_class:", var_class)
    print("count_class:", count_class)

    if method == 'SOSD':
        SOSD = np.zeros(data.shape[1])
        for i in range(classes):
            for j in range(classes):
                if i < j:
                    # print((mean_class[i] - mean_class[j]) ** 2)
                    SOSD += (mean_class[i] - mean_class[j]) ** 2
        # 1. sort the index corresponds to the weights.
        index_sosd = np.argsort(-SOSD, axis=0)  # in descending order, use the negative to help reverse it
        # 2. then get the first  num_wanted_sample.
        # print("SOSD:",SOSD)
        return index_sosd[:num_features_wants]
    elif method == 'SOST':
        SOST = np.zeros(data.shape[1])
        for i in range(classes):
            for j in range(classes):
                if count_class[i] > 1 and count_class[j] > 1:  # ignore those classes that does not exist
                    if i < j:
                        # print("i:", i)
                        # print("((var_class[i]**2/count_class[i])", ((var_class[i]**2/count_class[i])))
                        #
                        # print("i:", j)
                        # print("((var_class[j]**2/count_class[j])", ((var_class[j]**2/count_class[j])))
                        SOST += (mean_class[i] - mean_class[j]) ** 2 / (
                                    (var_class[i] ** 2 / count_class[i]) + (var_class[j] ** 2 / count_class[j]))

        # 1. sort the index corresponds to the weights.
        index_sost = np.argsort(-SOST, axis=0)  # in descending order, use the negative to help reverse it
        # 2. then get the first  num_wanted_sample.
        # print("SOST:", SOST)
        return index_sost[:num_features_wants]

def perform_attacks_without_log( nb_traces, predictions, plt_attack,correct_key,leakage,dataset,nb_attacks=1, shuffle=True):
    '''
    :param nb_traces: number_traces used to attack
    :param predictions: output of the neural network i.e. prob of each class
    :param plt_attack: plaintext from attack traces
    :param nb_attacks: number of attack experiments
    :param byte: byte in questions
    :param shuffle: true then it shuffle
    :return: mean of the rank for each experiments, log_probability of the output for all key
    '''
    all_rk_evol = np.zeros((nb_attacks, nb_traces)) #(num_attack, num_traces used)
    all_key_log_prob = np.zeros(256)
    for i in tqdm(range(nb_attacks)):
        if shuffle:
            l = list(zip(predictions, plt_attack)) #list of [prediction, plaintext_attack]
            random.shuffle(l) #shuffle the each other prediction
            sp, splt = list(zip(*l)) #*l = unpacking, output: shuffled predictions and shuffled plaintext.
            sp = np.array(sp)
            splt = np.array(splt)
            att_pred = sp[:nb_traces] #just use the required number of traces
            att_plt = splt[:nb_traces]

        else:
            att_pred = predictions[:nb_traces]
            att_plt = plt_attack[:nb_traces]
        rank_evol, key_log_prob = rank_compute_without_log(att_pred, att_plt,correct_key,leakage=leakage,dataset=dataset)
        all_rk_evol[i] = rank_evol
        all_key_log_prob += key_log_prob
    print()
    return np.mean(all_rk_evol, axis=0), np.float32(all_key_log_prob)


def rank_compute_without_log(prediction, att_plt, correct_key,leakage, dataset):
    '''
    :param prediction: prediction by the neural network
    :param att_plt: attack plaintext
    :return: key_log_prob which is the log probability
    '''
    hw = [bin(x).count("1") for x in range(256)]
    (nb_traces, nb_hyp) = prediction.shape

    key_log_prob = np.zeros(256)
    rank_evol = np.full(nb_traces, 255)
    # print(nb_traces)
    for i in range(nb_traces):
        for k in range(256):
            if dataset == "AES_HD_ext":
                if leakage == 'ID':
                    key_log_prob[k] += prediction[i, AES_Sbox_inv[k ^ int(att_plt[i, 15])] ^ att_plt[i, 11] ]
                else:

                    key_log_prob[k] += prediction[i, hw[AES_Sbox_inv[k ^ int(att_plt[i, 15])] ^ att_plt[i, 11]] ]
            else:
                if leakage == 'ID':
                    key_log_prob[k] += prediction[i,  AES_Sbox[k ^ int(att_plt[i])]]
                else:
                    key_log_prob[k] += prediction[i,  hw[ AES_Sbox[k ^ int(att_plt[i])]]]

        rank_evol[i] =  rk_key(key_log_prob, correct_key) #this will sort it.

    return rank_evol, key_log_prob


def create_cov_mean_per_class(cut_out_trace_profiling,Y_profiling, num_of_features, classes):
    split_trace_classes = [[] for i in range(classes)]
    for i in range(cut_out_trace_profiling.shape[0]):
        split_trace_classes[int(Y_profiling[i])].append(cut_out_trace_profiling[i])
    mean_classes = [[] for i in range(classes)]
    cov_classes = [[] for i in range(classes)]
    for cl in range(classes):
        split_trace_classes[cl] = np.array(split_trace_classes[cl])
        mean_classes[cl].append(np.mean(split_trace_classes[cl], axis=0))
        cov_classes[cl].append(np.cov(split_trace_classes[cl].T))


    # 2. attack phase
    if num_of_features == 1:
        determinant_classes = np.zeros(classes)  # determinant = variance
        for cl in range(classes):
            determinant_classes[cl] = cov_classes[cl][0]
    else:
        determinant_classes = np.zeros(classes)
        for cl in range(classes):
            determinant_classes[cl] = np.linalg.det(cov_classes[cl][0])
    return determinant_classes, mean_classes, cov_classes

def template_attack_single_trace(trace, classes, determinant_classes, mean_classes, cov_classes):
    def cal_probability_one_class(trace, determinant, mean_cl, cov_cl):
        if len(trace) == 1:
            trace_minus_mean = (trace[0] - mean_cl[0])
            hi = trace_minus_mean ** 2 / cov_cl

            return -np.log(determinant + 1e-40) - (1 / 2) * hi

        # cal log of probability
        else:
            trace_minus_mean = (trace - mean_cl).reshape(trace.shape[0], 1)
            inv_cov_matrix = np.linalg.inv(cov_cl)
            hi = np.matmul(np.matmul(trace_minus_mean.T, inv_cov_matrix), trace_minus_mean)[0][0]
            return (-1 / 2) * np.log(determinant + 1e-40) - (1 / 2) * hi

    return np.array(list(
        map(lambda i: cal_probability_one_class(trace, determinant_classes[i], mean_classes[i][0], cov_classes[i][0]),
            [j for j in range(classes)])))

def template_attack(traces, classes, determinant_classes, mean_classes, cov_classes):
    log_prob = np.zeros((traces.shape[0], classes))
    for i in tqdm(range(traces.shape[0])):
        log_prob[i, :] = template_attack_single_trace(traces[i, :], classes, determinant_classes, mean_classes,
                                                      cov_classes)
    return log_prob

if __name__ == "__main__":

    np.random.seed(seed=100)
    root = './'

    dataset = 'ASCAD' #'Chipwhisperer' #ASCAD #ASCAD_desync50 #ASCAD_variable #AES_HD_ext
    leakage = 'ID'
    non_overfitting = False
    iterative_type = "random" #forward #backward #full #random
    if non_overfitting == False:
        save_root = './Result/' + dataset + '_CNN_' + leakage+'/'
    else:
        save_root = './Result/' + dataset + '_CNN_' + leakage+'_non_overfitting/'
    if dataset == "ASCAD_desync50":
        save_root = './Result/' + dataset + '_CNN_' + leakage + '_desync_50/'
    if dataset == "ASCAD_desync100":
        save_root = './Result/' + dataset + '_CNN_' + leakage + '_desync_100/'
    print(tf.__version__)
    save_root_occlusion = save_root + 'key_guessing_occlusion/'
    save_root_occlusion_iterative_type = save_root_occlusion + iterative_type + '/'


    print("PATH save_root:", save_root_occlusion_iterative_type)
    image_root = save_root_occlusion_iterative_type + 'image/'
    attack_root = save_root_occlusion_iterative_type + 'attack/'
    attribution_root = save_root + 'attribution/'

    if not os.path.exists(save_root_occlusion):
        os.mkdir(save_root_occlusion)
    if not os.path.exists(save_root_occlusion_iterative_type):
        os.mkdir(save_root_occlusion_iterative_type)
    if not os.path.exists(image_root):
        os.mkdir(image_root)
    if not os.path.exists(attack_root):
        os.mkdir(attack_root)
    if not os.path.exists(attribution_root):
        os.mkdir(attribution_root)

    seed = 0
    random_state = check_random_state(seed)
    nb_traces_attacks = 2000 #10000 for ASCAD, 2000 for CW, 20000 AES_HD_ext
    nb_attacks = 50
    byte = 2
    if dataset == 'ASCAD':
        data_root = 'Dataset/ASCAD/ASCAD.h5'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_ascad(
            root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
            test_end=nb_traces_attacks)
    elif dataset == 'ASCAD_desync50':
        byte = 2
        data_root = 'Dataset/ASCAD/ASCAD_desync50.h5'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_ascad(
            root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
            test_end=nb_traces_attacks)
    elif dataset == 'ASCAD_desync100':
        byte = 2
        data_root = 'Dataset/ASCAD/ASCAD_desync100.h5'
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

    elif dataset == 'Chipwhisperer_desync25':
        data_root = 'Dataset/Chipwhisperer/'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_chipwhisperer_desync(
            root + data_root + '/', leakage_model=leakage,desync_lvl = 25)

    elif dataset == 'Chipwhisperer_desync50':
        data_root = 'Dataset/Chipwhisperer/'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_chipwhisperer_desync(
            root + data_root + '/', leakage_model=leakage,desync_lvl = 50)

    elif dataset == 'AES_HD_ext':
        data_root = 'Dataset/AES_HD_ext/aes_hd_ext.h5'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_aes_hd_ext(root +data_root,leakage_model=leakage,train_begin=0, train_end=45000, test_begin=0,
            test_end=nb_traces_attacks)

    print("The dataset we using: ", data_root)
    # load the data and normalize it
    print("Number of X_profiling used: ", X_profiling.shape)
    if leakage == 'HW':
        print("Number of Y_profiling used: ", len(Y_profiling))
    elif leakage == 'ID':
        print("Number of Y_profiling used: ", Y_profiling.shape)

    print("Number of X_attack used: ", X_attack.shape)
    if leakage == 'HW':
        print("Number of Y_attack used: ", len(Y_attack))
    elif leakage == 'ID':
        print("Number of Y_attack used: ", Y_attack.shape)

    input_length = len(X_profiling[0])
    print('Input length: {}'.format(input_length))
    scaler = StandardScaler()
    X_profiling = scaler.fit_transform(X_profiling)
    X_attack = scaler.transform(X_attack)




    if leakage == 'ID':
        print('ID leakage model')
        classes = 256
        if non_overfitting == True:
            if dataset == "ASCAD":
                model = cnn_ascad_fix_ID(input_length, classes=classes)
                cp_path = save_root + "saved_model_06.h5"
                print("cp_path: ", cp_path)
                model.load_weights(cp_path)
            if dataset == "ASCAD_variable":
                model = cnn_ascad_variable_ID(input_length, classes=classes)
                cp_path = save_root + "saved_model_05.h5"
                print("cp_path: ", cp_path)
                model.load_weights(cp_path)
            else:
                print("DONT HAVE non_overfitting model! please looking into it: " + dataset)
        else:
            if dataset == "ASCAD_desync50":
                print(save_root + '/' + dataset + '_' + leakage + '_CNN_desync_50_162' + '.h5')
                model = load_model(save_root + '/' + dataset + '_' + leakage + '_CNN_desync_50_162' + '.h5',
                                   compile=False)
            elif dataset == "ASCAD_desync100":
                print(save_root + '/' + dataset + '_' + leakage + '_CNN_desync_100_801' + '.h5')
                model = load_model(save_root + '/' + dataset + '_' + leakage + '_CNN_desync_100_801' + '.h5', compile=False)
            elif dataset == "Chipwhisperer_desync25":
                print(save_root + '/' + dataset + '_' + leakage + '_CNN_405' + '.h5')
                model = load_model(save_root + '/' + dataset + '_' + leakage + '_CNN_405' + '.h5', compile=False)
            else:
                print(save_root + '/' + dataset + '_' + leakage + '_CNN' + '.h5')
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
    # attack_type = "ta"
    important_samplept_further = np.load(save_root_occlusion_iterative_type+"important_samplept_further.npy" ) #important_samplept_further
    important_samplept_further = np.array(sorted(important_samplept_further, reverse = False))
    print("important_samplept_further: ", important_samplept_further)
    #Just to check

    print("important_samplept_further", important_samplept_further.shape)
    cut_out_trace_profiling = deepcopy(X_profiling)[:,important_samplept_further]
    cut_out_trace_attack = deepcopy(X_attack)[:,important_samplept_further]
    print("cut_out_trace_profiling", cut_out_trace_profiling.shape)
    print("cut_out_trace_attack", cut_out_trace_attack.shape)



    #We shall build templates
    #1. seperate them to their class
    save_kgo = False
    apply_sosd = False
    apply_sost = False
    apply_cpa = False
    apply_ho_cpa = False
    apply_grad = False
    apply_lrp = False
    apply_one_occlusion = False
    if save_kgo == True:
        split_trace_classes = [[] for i in range(classes)]
        for i in range(cut_out_trace_profiling.shape[0]):
            split_trace_classes[int(Y_profiling[i])].append(cut_out_trace_profiling[i])
        mean_classes = [[] for i in range(classes)]
        cov_classes = [[] for i in range(classes)]
        for cl in range(classes):
            print("cl:", cl)
            split_trace_classes[cl] = np.array(split_trace_classes[cl])
            print(split_trace_classes[cl].shape)
            mean_classes[cl].append(np.mean(split_trace_classes[cl], axis = 0))
            cov_classes[cl].append(np.cov(split_trace_classes[cl].T))


        print("mean vector class 0:", np.array(mean_classes[0][0]).shape)
        print("cov matrix class 0:", np.array(cov_classes[0][0]).shape) #if only one sample, the cov  = variance.

        #2. attack phase
        if important_samplept_further.shape[0] ==1:
            determinant_classes = np.zeros(classes) #determinant = variance
            for cl in range(classes):
                determinant_classes[cl] = cov_classes[cl][0]
        else:
            determinant_classes = np.zeros(classes)
            for cl in range(classes):
                determinant_classes[cl] = np.linalg.det(cov_classes[cl][0])
            print("determinant: ", determinant_classes.shape)


        predictions_ta_log = template_attack(cut_out_trace_attack , classes, determinant_classes, mean_classes, cov_classes)
        GE, _ = perform_attacks_without_log(nb_traces_attacks, predictions_ta_log, plt_attack, correct_key, leakage, nb_attacks=50,
                                shuffle=True, dataset=dataset)
        print("GE:", GE)

        NTGE = cal_NTGE(GE)

        np.save(attack_root + "GE_"+dataset+"_"+leakage+"_"+iterative_type+".npy", GE)
        np.save(attack_root + "NTGE_"+dataset+"_"+leakage+"_"+iterative_type+".npy", np.array(NTGE))
    else:
        GE = np.load(attack_root + "GE_"+dataset+"_"+leakage+"_"+iterative_type+".npy")
        NTGE = np.load(attack_root + "NTGE_"+dataset+"_"+leakage+"_"+iterative_type+".npy")
        print(GE)
        print(GE.shape)
        print("NTGE KGO:", NTGE)
        pass


    extract_num_features = important_samplept_further.shape[0]

    print("extract_num_features:", extract_num_features)


    # sosd
    if apply_sosd == True:
        features_index_sosd = SOSD_SOST(X_attack, Y_attack, classes, num_features_wants = extract_num_features, method= 'SOSD') #Using profiling traces to find the features then train the template attack
        features_index_sosd = np.array(sorted(features_index_sosd, reverse=False))
        print("features_index_sosd:", features_index_sosd)
        print("features_index_sosd shape:", features_index_sosd.shape)
        cut_out_trace_profiling_sosd = deepcopy(X_profiling)[:,features_index_sosd]
        cut_out_trace_attack_sosd = deepcopy(X_attack)[:,features_index_sosd]
        determinant_classes_sosd, mean_classes_sosd, cov_classes_sosd = create_cov_mean_per_class(cut_out_trace_profiling_sosd,Y_profiling, extract_num_features, classes)
        predictions_ta_log_sosd = template_attack(cut_out_trace_attack_sosd, classes, determinant_classes_sosd, mean_classes_sosd, cov_classes_sosd)
        GE_sosd, _ = perform_attacks_without_log(nb_traces_attacks, predictions_ta_log_sosd, plt_attack, correct_key, leakage,
                                            nb_attacks=50,
                                            shuffle=True, dataset=dataset)
        print("GE_sosd:", GE_sosd)

        NTGE_sosd = cal_NTGE(GE_sosd)

        np.save(attack_root + "features_index_" + dataset + "_" + leakage + "_" + iterative_type + "_sosd.npy", features_index_sosd)
        np.save(attack_root + "GE_" + dataset + "_" + leakage + "_" + iterative_type + "_sosd.npy", GE_sosd)
        np.save(attack_root + "NTGE_" + dataset + "_" + leakage + "_" + iterative_type + "_sosd.npy", np.array(NTGE_sosd))
    else:
        features_index_sosd = np.load(attack_root + "features_index_" + dataset + "_" + leakage + "_" + iterative_type + "_sosd.npy")
        GE_sosd = np.load(attack_root + "GE_" + dataset + "_" + leakage + "_" + iterative_type + "_sosd.npy")
        NTGE_sosd = np.load(attack_root + "NTGE_" + dataset + "_" + leakage + "_" + iterative_type + "_sosd.npy")
        print("features_index_sosd:", features_index_sosd)
        print("GE_sosd:", GE_sosd)
        print("NTGE_sosd:", NTGE_sosd)
    # sost
    if apply_sost == True:
        features_index_sost = SOSD_SOST(X_attack, Y_attack, classes, num_features_wants=extract_num_features,
                                        method='SOST')  # Using profiling traces to find the features then train the template attack
        features_index_sost = np.array(sorted(features_index_sost, reverse = False))
        print("features_index_sost:", features_index_sost)
        print("features_index_sost shape:", features_index_sost.shape)
        cut_out_trace_profiling_sost = deepcopy(X_profiling)[:, features_index_sost]
        cut_out_trace_attack_sost = deepcopy(X_attack)[:, features_index_sost]
        determinant_classes_sost, mean_classes_sost, cov_classes_sost = create_cov_mean_per_class(cut_out_trace_profiling_sost, Y_profiling,extract_num_features,
                                                                                                  classes)
        predictions_ta_log_sost = template_attack(cut_out_trace_attack_sost, classes, determinant_classes_sost,
                                                  mean_classes_sost, cov_classes_sost)
        GE_sost, _ = perform_attacks_without_log(nb_traces_attacks, predictions_ta_log_sost, plt_attack, correct_key,
                                                 leakage,
                                                 nb_attacks=50,
                                                 shuffle=True, dataset=dataset)
        print("GE_sost:", GE_sost)

        NTGE_sost = cal_NTGE(GE_sost)

        np.save(attack_root + "features_index_" + dataset + "_" + leakage + "_" + iterative_type + "_sost.npy",
                features_index_sost)
        np.save(attack_root + "GE_" + dataset + "_" + leakage + "_" + iterative_type + "_sost.npy", GE_sost)
        np.save(attack_root + "NTGE_" + dataset + "_" + leakage + "_" + iterative_type + "_sost.npy", np.array(NTGE_sost))
    else:
        features_index_sost = np.load(attack_root + "features_index_" + dataset + "_" + leakage + "_" + iterative_type + "_sost.npy")
        GE_sost = np.load(attack_root + "GE_" + dataset + "_" + leakage + "_" + iterative_type + "_sost.npy")
        NTGE_sost = np.load(attack_root + "NTGE_" + dataset + "_" + leakage + "_" + iterative_type + "_sost.npy")
        print("features_index_sost:", features_index_sost)
        print("GE:", GE_sost)
        print("NTGE_sost:", NTGE_sost)
    #cpa
    if apply_cpa == True:
        cpa_root = './Dataset/' + dataset + '/CPA/'
        if dataset == "Chipwhisperer" or dataset == "AES_HD_ext":
            cpa = np.load(cpa_root + '/CPA_attack' + leakage + '_' + dataset + '.npy')
            features_index_cpa= np.argsort(-cpa, axis=0)[:extract_num_features]
        elif dataset == "Chipwhisperer_desync25" or dataset == "Chipwhisperer_desync50":
            cpa = np.load('./Dataset/Chipwhisperer/CPA/CPA_attack' + leakage + '_' + dataset + '_Sbox.npy')
            features_index_cpa= np.argsort(-cpa, axis=0)[:extract_num_features]

        if dataset == "ASCAD" or dataset == "ASCAD_variable" or dataset == "ASCAD_desync50"or dataset == "ASCAD_desync100":
            # cpa_second_order = np.load(cpa_root + 'second_order' + '/CPA_order2_' + leakage + '_' + dataset + '.npy')
            # features_index_cpa = np.argsort(-cpa_second_order, axis=0)[:extract_num_features]
            # print("features_index_cpa: ", features_index_cpa)

            number_of_traces = X_attack.shape[0]  # number of sample points in a trace
            total_samplept = X_attack.shape[1]
            traces = X_attack
            label = Y_attack
            cpa_univariate = cpa_method(total_samplept, number_of_traces, label, traces)
            np.save("Dataset/ASCAD/CPA/first_order/cpa_univariate_"+dataset+"_"+leakage+".npy", cpa_univariate)
            features_index_cpa = np.argsort(-cpa_univariate, axis=0)[:extract_num_features]
        features_index_cpa = np.array(sorted(features_index_cpa, reverse = False))
        print("features_index_cpa: ", features_index_cpa)
        cut_out_trace_profiling_cpa = deepcopy(X_profiling)[:, features_index_cpa]
        cut_out_trace_attack_cpa = deepcopy(X_attack)[:, features_index_cpa]
        # print(cut_out_trace_attack_cpa)
        determinant_classes_cpa, mean_classes_cpa, cov_classes_cpa = create_cov_mean_per_class(cut_out_trace_profiling_cpa, Y_profiling,
                                                                                                  extract_num_features,
                                                                                                  classes)



        predictions_ta_log_cpa= template_attack(cut_out_trace_attack_cpa, classes, determinant_classes_cpa,
                                                  mean_classes_cpa, cov_classes_cpa)
        GE_cpa, _ = perform_attacks_without_log(nb_traces_attacks, predictions_ta_log_cpa, plt_attack, correct_key,
                                                 leakage,
                                                 nb_attacks=50,

                                                     shuffle=True, dataset=dataset)



        print("GE_cpa:", GE_cpa)

        NTGE_cpa = cal_NTGE(GE_cpa)

        np.save(attack_root + "features_index_" + dataset + "_" + leakage + "_" + iterative_type + "_cpa.npy",
                features_index_cpa)
        np.save(attack_root + "GE_" + dataset + "_" + leakage + "_" + iterative_type + "_cpa.npy", GE_cpa)
        np.save(attack_root + "NTGE_" + dataset + "_" + leakage + "_" + iterative_type + "_cpa.npy", np.array(NTGE_cpa))
    else:
        features_index_cpa = np.load(attack_root + "features_index_" + dataset + "_" + leakage + "_" + iterative_type + "_cpa.npy")
        GE_cpa = np.load(attack_root + "GE_" + dataset + "_" + leakage + "_" + iterative_type + "_cpa.npy")
        NTGE_cpa = np.load(attack_root + "NTGE_" + dataset + "_" + leakage + "_" + iterative_type + "_cpa.npy")
        print("features_index_cpa:", features_index_cpa)
        print("GE_cpa:", GE_cpa)
        print("GE_cpa:", GE_cpa.shape)
        print("NTGE_cpa:", NTGE_cpa)
    #ho_cpa
    if dataset == "ASCAD" or dataset == "ASCAD_variable" or dataset == "ASCAD_desync50" or dataset == "ASCAD_desync100":
        if apply_ho_cpa == True:
            #1. Load the cpa
            cpa_second_order = np.load('./Dataset/ASCAD/CPA/second_order' + '/CPA_order2_' + leakage + '_' + dataset + '_attack.npy')
            print("cpa_second_order.shape:", cpa_second_order.shape)
            print("extract_num_features:", extract_num_features)
            print("extract_num_features/2:", extract_num_features/2)
            print("int(extract_num_features/2):", math.ceil(extract_num_features/2))
            if dataset == "ASCAD_desync100": #Here we figured out that we can use itertool for combination instead to speed up our progress.
                num_features_combine = math.ceil(2 * extract_num_features)
                combine_features_index_ho_cpa = np.argsort(-cpa_second_order, axis=0)[:(num_features_combine)]
                print("combine_features_index_ho_cpa: ", combine_features_index_ho_cpa)
                comb_spt_lst = np.array(list(itertools.combinations([i for i in range(X_attack.shape[1])],2)))
                chosen_original_features = comb_spt_lst[combine_features_index_ho_cpa] #[(spt1,spt2),(spt1,spt2),.... ]
                features_index_ho_cpa = np.unique(chosen_original_features.flatten()) # flatten it to one list and then unique.
                print("features_index_ho_cpa: ", features_index_ho_cpa)
                print("features_index_ho_cpa: ", features_index_ho_cpa.shape)
                pass

            else:
                if dataset == "ASCAD" and leakage == "ID" and non_overfitting == False:
                    num_features_combine = math.ceil(2*extract_num_features ) ##chose 6 from original trace while actual 5
                elif dataset == "ASCAD" and leakage == "ID" and non_overfitting == True:
                    num_features_combine = math.ceil(2*extract_num_features) - 2 ##chose 32 from original while actual 31
                elif dataset == "ASCAD_variable" and leakage == "ID" and non_overfitting == False:
                    num_features_combine = math.ceil(2*extract_num_features)-2  ##chose 6 from original while actual 6
                elif dataset == "ASCAD_desync50" and leakage == "ID":
                    num_features_combine = math.ceil(3*extract_num_features) +4  ##chose 32 from original while actual 32

                combine_features_index_ho_cpa = np.argsort(-cpa_second_order, axis=0)[:(num_features_combine)]

                print("combine_features_index_ho_cpa: ", combine_features_index_ho_cpa)
                # 2. Pull back to the original traces
                # note: could have duplicate because we multiply pointwise instead of using combination for these cases.
                j_index =  combine_features_index_ho_cpa%X_profiling.shape[1]
                i_index = ((combine_features_index_ho_cpa-j_index)/X_profiling.shape[1]).astype(int)
                features_index_ho_cpa = np.unique(np.concatenate((i_index,j_index)))
                features_index_ho_cpa = np.array(sorted(features_index_ho_cpa, reverse=False))
                print("j_index: ", j_index)
                print("i_index: ", i_index)
                print("features_index_ho_cpa: ", features_index_ho_cpa)
                print("features_index_ho_cpa: ", features_index_ho_cpa.shape)
            extract_num_features_ho_cpa = features_index_ho_cpa.shape[0]
            cut_out_trace_profiling_ho_cpa = deepcopy(X_profiling)[:, features_index_ho_cpa]
            cut_out_trace_attack_ho_cpa = deepcopy(X_attack)[:, features_index_ho_cpa]
            determinant_classes_ho_cpa, mean_classes_ho_cpa, cov_classes_ho_cpa = create_cov_mean_per_class(
                cut_out_trace_profiling_ho_cpa, Y_profiling,
                extract_num_features_ho_cpa,
                classes)

            predictions_ta_log_ho_cpa = template_attack(cut_out_trace_attack_ho_cpa, classes, determinant_classes_ho_cpa,
                                                     mean_classes_ho_cpa, cov_classes_ho_cpa)
            GE_ho_cpa, _ = perform_attacks_without_log(nb_traces_attacks, predictions_ta_log_ho_cpa, plt_attack, correct_key,
                                                    leakage,
                                                    nb_attacks=50,

                                                    shuffle=True, dataset=dataset)

            print("GE_ho_cpa:", GE_ho_cpa)
            NTGE_ho_cpa = cal_NTGE(GE_ho_cpa)

            np.save(attack_root + "features_index_" + dataset + "_" + leakage + "_" + iterative_type + "_ho_cpa.npy",
                    features_index_ho_cpa)
            np.save(attack_root + "combine_features_index_" + dataset + "_" + leakage + "_" + iterative_type + "_ho_cpa.npy",
                    combine_features_index_ho_cpa)
            np.save(attack_root + "GE_" + dataset + "_" + leakage + "_" + iterative_type + "_ho_cpa.npy", GE_ho_cpa)
            np.save(attack_root + "NTGE_" + dataset + "_" + leakage + "_" + iterative_type + "_ho_cpa.npy",
                    np.array(NTGE_ho_cpa))
        else:
            features_index_ho_cpa = np.load(attack_root + "features_index_" + dataset + "_" + leakage + "_" + iterative_type + "_ho_cpa.npy")
            GE_ho_cpa = np.load(attack_root + "GE_" + dataset + "_" + leakage + "_" + iterative_type + "_ho_cpa.npy")
            NTGE_ho_cpa = np.load(attack_root + "NTGE_" + dataset + "_" + leakage + "_" + iterative_type + "_ho_cpa.npy")
            print("features_index_ho_cpa:", features_index_ho_cpa)
            print("GE_ho_cpa:", GE_ho_cpa)
            print("GE_ho_cpa:", GE_ho_cpa.shape)
            print("NTGE_ho_cpa:", NTGE_ho_cpa)




    if apply_grad == True:
        attribution = np.load(attribution_root+"attribution_grad_inn.npy")
        if dataset == "Chipwhispherer":
            num_profiling_trace_attrib = 8000
        else:
            num_profiling_trace_attrib = 10000
        attribution_grad = attribution / num_profiling_trace_attrib  # forget to put inside
        features_index_grad = np.argsort(-attribution, axis=0)[:extract_num_features]
        features_index_grad = np.array(sorted(features_index_grad, reverse=False))
        print("features_index_grad: ", features_index_grad)
        cut_out_trace_profiling_grad = deepcopy(X_profiling)[:, features_index_grad]
        cut_out_trace_attack_grad = deepcopy(X_attack)[:, features_index_grad]



        determinant_classes_grad, mean_classes_grad, cov_classes_grad = create_cov_mean_per_class(
            cut_out_trace_profiling_grad, Y_profiling,
            extract_num_features,
            classes)

        predictions_ta_log_grad = template_attack(cut_out_trace_attack_grad, classes, determinant_classes_grad,
                                                 mean_classes_grad, cov_classes_grad)
        GE_grad, _ = perform_attacks_without_log(nb_traces_attacks, predictions_ta_log_grad, plt_attack, correct_key,
                                                leakage,
                                                nb_attacks=50,

                                                shuffle=True, dataset=dataset)

        print("GE_grad:", GE_grad)

        NTGE_grad = cal_NTGE(GE_grad)

        np.save(attribution_root + "attribution_grad.npy",attribution)
        np.save(attack_root + "features_index_" + dataset + "_" + leakage + "_" + iterative_type + "_grad.npy",
                features_index_grad)
        np.save(attack_root + "GE_" + dataset + "_" + leakage + "_" + iterative_type + "_grad.npy", GE_grad)
        np.save(attack_root + "NTGE_" + dataset + "_" + leakage + "_" + iterative_type + "_grad.npy", np.array(NTGE_grad))
    else:
        features_index_grad = np.load(
            attack_root + "features_index_" + dataset + "_" + leakage + "_" + iterative_type + "_grad.npy")
        GE_grad = np.load(attack_root + "GE_" + dataset + "_" + leakage + "_" + iterative_type + "_grad.npy")
        NTGE_grad = np.load(attack_root + "NTGE_" + dataset + "_" + leakage + "_" + iterative_type + "_grad.npy")
        print("features_index_grad:", features_index_grad)
        print("GE_grad:", GE_grad)
        print("GE_grad:", GE_grad.shape)
        print("NTGE_grad:", NTGE_grad)

    if apply_lrp == True:
        attribution = np.load(attribution_root + "attribution_lrp.npy")
        if dataset == "Chipwhispherer":
            num_profiling_trace_attrib = 8000
        else:
            num_profiling_trace_attrib = 10000
        attribution_grad = attribution / num_profiling_trace_attrib  # forget to put inside
        features_index_lrp = np.argsort(-abs(attribution), axis=0)[:extract_num_features]
        features_index_lrp = np.array(sorted(features_index_lrp, reverse = False))
        cut_out_trace_profiling_lrp = deepcopy(X_profiling)[:, features_index_lrp]
        cut_out_trace_attack_lrp = deepcopy(X_attack)[:, features_index_lrp]
        # print(cut_out_trace_attack_cpa)

        print("features_index_lrp: ", features_index_lrp)
        determinant_classes_lrp, mean_classes_lrp, cov_classes_lrp = create_cov_mean_per_class(
            cut_out_trace_profiling_lrp, Y_profiling,
            extract_num_features,
            classes)

        predictions_ta_log_lrp = template_attack(cut_out_trace_attack_lrp, classes, determinant_classes_lrp,
                                                  mean_classes_lrp, cov_classes_lrp)
        GE_lrp, _ = perform_attacks_without_log(nb_traces_attacks, predictions_ta_log_lrp, plt_attack, correct_key,
                                                 leakage,
                                                 nb_attacks=50,
                                                 shuffle=True, dataset=dataset)

        print("GE_lrp:", GE_lrp)

        NTGE_lrp = cal_NTGE(GE_lrp)

        np.save(attribution_root + "attribution_lrp.npy",attribution)
        np.save(attack_root + "features_index_" + dataset + "_" + leakage + "_" + iterative_type + "_lrp.npy",
                features_index_lrp)
        np.save(attack_root + "GE_" + dataset + "_" + leakage + "_" + iterative_type + "_lrp.npy", GE_lrp)
        np.save(attack_root + "NTGE_" + dataset + "_" + leakage + "_" + iterative_type + "_lrp.npy",
                np.array(NTGE_lrp))
    else:
        features_index_lrp = np.load(
            attack_root + "features_index_" + dataset + "_" + leakage + "_" + iterative_type + "_lrp.npy")
        GE_lrp = np.load(attack_root + "GE_" + dataset + "_" + leakage + "_" + iterative_type + "_lrp.npy")
        NTGE_lrp = np.load(attack_root + "NTGE_" + dataset + "_" + leakage + "_" + iterative_type + "_lrp.npy")
        print("features_index_lrp:", features_index_lrp)
        print("GE_lrp:", GE_lrp)
        print("NTGE_lrp:", NTGE_lrp)
        pass

    if apply_one_occlusion == True:
        attribution = np.load(attribution_root+"attribution_occ.npy")
        if dataset == "Chipwhispherer":
            num_profiling_trace_attrib = 8000
        else:
            num_profiling_trace_attrib = 10000
        attribution_grad = attribution / num_profiling_trace_attrib  # forget to put inside
        features_index_occ = np.argsort(-abs(attribution), axis=0)[:extract_num_features]
        features_index_occ = np.array(sorted(features_index_occ, reverse = False))
        print("features_index_occ: ", features_index_occ)
        cut_out_trace_profiling_occ = deepcopy(X_profiling)[:, features_index_occ]
        cut_out_trace_attack_occ = deepcopy(X_attack)[:, features_index_occ]
        # print(cut_out_trace_attack_cpa)
        determinant_classes_occ, mean_classes_occ, cov_classes_occ = create_cov_mean_per_class(
            cut_out_trace_profiling_occ, Y_profiling,
            extract_num_features,
            classes)

        predictions_ta_log_occ = template_attack(cut_out_trace_attack_occ, classes, determinant_classes_occ,
                                                 mean_classes_occ, cov_classes_occ)
        GE_occ, _ = perform_attacks_without_log(nb_traces_attacks, predictions_ta_log_occ, plt_attack, correct_key,
                                                leakage,
                                                nb_attacks=50,
                                                shuffle=True, dataset=dataset)

        print("GE_occ:", GE_occ)

        NTGE_occ = cal_NTGE(GE_occ)
        np.save(attribution_root + "attribution_occ.npy",attribution)
        np.save(attack_root + "features_index_" + dataset + "_" + leakage + "_" + iterative_type + "_occ.npy",
                features_index_occ)
        np.save(attack_root + "GE_" + dataset + "_" + leakage + "_" + iterative_type + "_occ.npy", GE_occ)
        np.save(attack_root + "NTGE_" + dataset + "_" + leakage + "_" + iterative_type + "_occ.npy",
                np.array(NTGE_occ))
    else:
        features_index_occ = np.load(
            attack_root + "features_index_" + dataset + "_" + leakage + "_" + iterative_type + "_occ.npy")
        GE_occ = np.load(attack_root + "GE_" + dataset + "_" + leakage + "_" + iterative_type + "_occ.npy")
        NTGE_occ = np.load(attack_root + "NTGE_" + dataset + "_" + leakage + "_" + iterative_type + "_occ.npy")
        print("features_index_occ:", features_index_occ)
        print("GE_occ:", GE_occ)
        print("NTGE_occ:", NTGE_occ)



    fig, ax = plt.subplots(figsize=(15, 7))
    x_axis = [i for i in range(nb_traces_attacks)]
    print("GE", GE.shape)
    print("GE_sosd", GE_sosd.shape)
    print("GE_sost", GE_sost.shape)
    print("GE_cpa", GE_cpa.shape)
    print("GE_grad", GE_grad.shape)
    print("GE_lrp", GE_lrp.shape)
    print("GE_occ", GE_occ.shape)
    ax.plot(x_axis, GE[:nb_traces_attacks], 'b', label = "KGO")
    ax.plot(x_axis, GE_sosd[:nb_traces_attacks], 'r', label = "SOSD")
    ax.plot(x_axis, GE_sost[:nb_traces_attacks], 'g', label = "SOST")
    ax.plot(x_axis, GE_cpa[:nb_traces_attacks], 'y', label = "CPA (first-order)")
    if dataset == "ASCAD" or dataset == "ASCAD_variable":
        ax.plot(x_axis, GE_ho_cpa[:nb_traces_attacks], 'purple', label = "CPA (multi.)")
    ax.plot(x_axis, GE_grad[:nb_traces_attacks], 'c', label = "Saliency Map")
    ax.plot(x_axis, GE_lrp[:nb_traces_attacks], 'orange', label = "LRP")
    ax.plot(x_axis, GE_occ[:nb_traces_attacks], 'grey', label = "1-Occlusion")


    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Number of Traces', fontsize=20)
    ax.set_ylabel('Key Rank', fontsize=20)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(15)
    ax.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0, prop={'size': 15})
    save_image = True
    if save_image == True:
        plt.savefig(image_root + 'Template_attack_' + dataset + "_" + leakage + "_" + "_" + iterative_type + "_GE.png")
    plt.show()




