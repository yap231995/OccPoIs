import os
import random
import sys
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from src.net import cnn_ascad_fix_ID, cnn_ascad_variable_ID
from tf_run import load_ascad, load_chipwhisperer, attack_calculate_metrics, perform_attacks, load_aes_hd_ext, load_chipwhisperer_desync


def cal_NTGE(GE):
    NTGE = float('inf')
    for i in range(GE.shape[0] - 1, -1, -1):
        if GE[i] >= 1:
            break
        elif GE[i] < 1:
            NTGE = i
    print("NTGE:", NTGE)
    return NTGE



def KGO(x,lst_points, model,nb_traces,plt_attack,dataset,correct_key,leakage, threshold = 1, path ="", do_not_shuffle_first = False, fix_iteration = []):

    y = deepcopy(x)
    new_x = np.zeros(x.shape)
    new_x[:, lst_points, :] = y[:, lst_points, :]
    predictions = model.predict(new_x)
    GE, _ = perform_attacks(nb_traces, predictions, plt_attack, correct_key, leakage, nb_attacks=50,
                            shuffle=True, dataset=dataset)
    print("START GE:", GE)


    flag = False
    flag_counter = 0
    flag_fix_iteration = 0

    while flag == False:
        flag = True
        important_index = []
        y = deepcopy(x)
        new_x = np.zeros(x.shape)
        new_x[:, lst_points, :] = y[:, lst_points, :]
        GE_vis_metric = []
        # print("START:", new_x.shape)

        if do_not_shuffle_first == True and flag_fix_iteration < len(fix_iteration) and (sorted(fix_iteration[flag_fix_iteration]) == sorted(lst_points)):
            print("using previous: iteration "+ str(flag_fix_iteration))
            lst_points = fix_iteration[flag_fix_iteration]
            flag_fix_iteration += 1
        else:
            random.shuffle(lst_points)
        # print("hi:", len(lst_points))
        print("lst_points iteration"+str(flag_counter)+": ", lst_points)
        np.save(path + "shuffle_lst_iteration_"+str(flag_counter), lst_points)
        flag_counter+=1
        for spt in lst_points:
            print("spt:", spt)
            original_value = deepcopy(new_x[:, spt, :]) #keep the original value
            new_x[:, spt, :] = 0 #Replace with 0


            # print("new_x", new_x[0])
            predictions = model.predict(new_x)
            GE, _ = perform_attacks(nb_traces, predictions, plt_attack, correct_key, leakage, nb_attacks=50,
                                    shuffle=True, dataset=dataset)

            print("GE:", GE)

            GE_vis_metric.append(GE[-1])
            if GE[-1] >= threshold:
                #put it back (these are important)
                new_x[:, spt, :] = original_value
                important_index.append(spt)
                print("Added into important_index")
            elif GE[-1] <threshold:
                flag = False

            print("important_index:", important_index)
        lst_points = important_index

    predictions = model.predict(new_x)
    GE, _ = perform_attacks(nb_traces, predictions, plt_attack, correct_key, leakage, nb_attacks=50,
                            shuffle=True, dataset=dataset)
    print("END GE:", GE)
    important_index = np.array(important_index)
    GE_vis_metric = np.array(GE_vis_metric)
    return important_index, GE_vis_metric



if __name__ == "__main__":

    np.random.seed(seed=100)
    root = './'

    dataset = 'Chipwhisperer' #'Chipwhisperer' #ASCAD #ASCAD_desync50 #ASCAD_variable #AES_HD_ext
    leakage = 'ID'
    non_overfitting = False
    occlusion_type = "input"
    iterative_type = "random"  #random
    if non_overfitting == False:
        save_root = './Result/' + dataset + '_CNN_' + leakage+'/'
    else:
        save_root = './Result/' + dataset + '_CNN_' + leakage+'_non_overfitting/'
    if dataset == "ASCAD_desync50":
        save_root = './Result/' + dataset + '_CNN_' + leakage + '_desync_50/'
    elif dataset == "ASCAD_desync100":
        save_root = './Result/' + dataset + '_CNN_' + leakage + '_desync_100/'
    print(tf.__version__)
    save_root_occlusion = save_root + 'key_guessing_occlusion/'
    save_root_occlusion_iterative_type = save_root_occlusion+iterative_type+'/'

    print("PATH save_root:", save_root_occlusion_iterative_type)
    image_root = save_root_occlusion_iterative_type + 'image/'

    if not os.path.exists(save_root_occlusion):
        os.mkdir(save_root_occlusion)
    if not os.path.exists(save_root_occlusion_iterative_type):
        os.mkdir(save_root_occlusion_iterative_type)
    if not os.path.exists(image_root):
        os.mkdir(image_root)

    seed = 0
    random_state = check_random_state(seed)
    nb_traces_attacks = 55000 #5000 for ASCAD, 2000 for CW #10000 for AES_HD, 55000 for ASCAD_var
    nb_attacks = 100
    byte = 2
    if dataset == 'ASCAD':
        data_root = 'Dataset/ASCAD/ASCAD.h5'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_ascad(
            root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
            test_end=nb_traces_attacks)
    if dataset == 'ASCAD_desync50':
        byte = 2
        data_root = 'Dataset/ASCAD/ASCAD_desync50.h5'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_ascad(
            root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
            test_end=nb_traces_attacks)
    if dataset == 'ASCAD_desync100':
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
                model = load_model(save_root + '/' + dataset + '_' + leakage + '_CNN_desync_50_162' + '.h5', compile=False)
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
    x = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
    if occlusion_type == "input":
        print("ITERATIVE_TYPE: ", iterative_type)
        if iterative_type == "random":
            lst_points = [i for i in range(X_attack.shape[1])]
            start_time = time.time()
            important_samplept_further, GE_vis_metric_further = KGO(x, lst_points, model, nb_traces_attacks, plt_attack, dataset, correct_key,leakage, threshold=1,
                                             path=save_root_occlusion_iterative_type)
            end_time = time.time() - start_time

            np.save(save_root_occlusion_iterative_type + "elapse_time.npy", end_time)
            np.save(save_root_occlusion_iterative_type+"important_samplept_further.npy", important_samplept_further)
            np.save(save_root_occlusion_iterative_type+"GE_vis_metric_further.npy", GE_vis_metric_further)
            # Load.....
            important_samplept_further = np.load(save_root_occlusion_iterative_type + "important_samplept_further.npy", )
            GE_vis_metric_further = np.load(save_root_occlusion_iterative_type + "GE_vis_metric_further.npy")
            end_time = np.load(save_root_occlusion_iterative_type + "elapse_time.npy")
        print("important_samplept_further: ", important_samplept_further)
        print("important_samplept_further: ", important_samplept_further.shape)
        print("elapse_time: ", end_time)
        print(important_samplept_further.shape)
        fig, ax = plt.subplots(figsize=(15, 7))
        x_axis = [i for i in range(X_attack.shape[1])]

        feature_importances_GE = np.zeros(X_attack.shape[1])
        feature_importances_GE[important_samplept_further] = GE_vis_metric_further
        print(GE_vis_metric_further)
        ax.plot(x_axis, feature_importances_GE, 'b')

        ax.set_xlabel('Sample Points', fontsize=20)
        ax.set_ylabel('GE', fontsize=20)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(15)

        plt.savefig(image_root + 'Input_Occlusion_' + dataset + "_" + leakage + "_" + occlusion_type + "_" + iterative_type + "_GE_further_in.png")
        plt.show()






