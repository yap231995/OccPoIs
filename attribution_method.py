import os
import sys
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model
from tqdm import tqdm

from src.net import cnn_ascad_fix_ID, cnn_ascad_variable_ID
from src.utils import batchNormToDense, insert_layer_nonseq
from tf_run import load_ascad, load_chipwhisperer, attack_calculate_metrics, load_aes_hd_ext,load_chipwhisperer_desync

tf.compat.v1.disable_eager_execution()
import innvestigate #In order to use this, one need to fixed the bug stated in README.md


def one_occulsion(x, model, correct_labels):
    input_length = x.shape[1] #x.shape = (num_traces, num_features, 1)

    # Start
    counter = 0
    new_x = deepcopy(x)

    predictions_original = model.predict(x)
    attribution_lst = []
    for counter in tqdm(range(input_length)):
        original_value = deepcopy(new_x[:, counter, :]) #keep the original value
        new_x[:, counter, :] = 0 #Replace with 0
        predictions_perturb_index = model.predict(new_x)


        new_x[:, counter, :] = original_value #put back
        relevancy = predictions_original - predictions_perturb_index
        relevancy_correct_class = np.zeros(relevancy.shape[0])
        for i in range(relevancy.shape[0]):
            relevancy_correct_class[i] = relevancy[i, int(correct_labels[i])]
        heatmap = np.sum(relevancy_correct_class)
        attribution_lst.append(heatmap)
    attribution_lst = np.array(attribution_lst)
    return attribution_lst





def attribution_method(x, model, correct_labels,attribution_type, path, flag = False):
    new_model = None
    if flag == True:
        for layer in model.layers:
            counter = 0
            if isinstance(layer, BatchNormalization):
                print("HELLO! ")
                replace_layer = batchNormToDense(layer)
                model = insert_layer_nonseq(model,layer._name, replace_layer,insert_layer_name=str(layer._name)+"_dense", position = "replace")
                model.save(path + "DNN_update_"+str(counter)+".h5")
                model = load_model(path + "DNN_update_"+str(counter)+".h5", compile=False)
        new_model = model
    model.summary()


    if attribution_type == 'one_occlusion':
        print("test one_occlusion")
        attribution = one_occulsion(x, model,correct_labels)
    elif attribution_type == 'lrp':
        print("test lrp")
        model_wo_sm = innvestigate.model_wo_softmax(model)
        lrp_analyzer = innvestigate.create_analyzer("lrp.epsilon", model_wo_sm, neuron_selection_mode="index")
        attribution = lrp_analyzer.analyze(x, correct_labels)
        attribution = attribution.squeeze()
        attribution = np.sum(attribution, axis = 0)
    elif attribution_type == 'gradient': #also known as Saliency Map
        print("test grad")
        model_wo_sm = innvestigate.model_wo_softmax(model)
        grad_analyzer = innvestigate.create_analyzer("gradient", model_wo_sm, neuron_selection_mode="index", postprocess = "abs")
        attribution = grad_analyzer.analyze(x, correct_labels)
        attribution = attribution.squeeze()
        attribution = np.sum(attribution, axis = 0)
    return attribution,new_model





if __name__ == "__main__":

    root = './'

    dataset = 'ASCAD' #'Chipwhisperer #ASCAD #ASCAD_desync50 #ASCAD_variable #AES_HD_ext
    leakage = 'ID'
    non_overfitting = False
    if non_overfitting == False:
        save_root = './Result/' + dataset + '_CNN_' + leakage+'/'
    else:
        save_root = './Result/' + dataset + '_CNN_' + leakage+'_non_overfitting/'
    if dataset == "ASCAD_desync50":

        save_root = './Result/' + dataset + '_CNN_' + leakage+'_desync_50/'
    elif dataset == "ASCAD_desync100":
        save_root = './Result/' + dataset + '_CNN_' + leakage+'_desync_100/'
    print(tf.__version__)
    save_root_attribution = save_root + 'attribution/'


    print("PATH save_root:", save_root_attribution)
    image_root = save_root_attribution + 'image/'


    if not os.path.exists(save_root_attribution):
        os.mkdir(save_root_attribution)
    if not os.path.exists(image_root):
        os.mkdir(image_root)

    seed = 0
    random_state = check_random_state(seed)
    nb_traces_attacks = 2000
    nb_attacks = 50
    byte = 2
    if dataset == 'ASCAD':
        data_root = 'Dataset/ASCAD/ASCAD.h5'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_ascad(
            root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
            test_end=nb_traces_attacks)
    elif dataset == 'ASCAD_desync50':
        data_root = 'Dataset/ASCAD/ASCAD_desync50.h5'
        (X_profiling, X_attack), (Y_profiling, Y_attack), (plt_profiling, plt_attack), correct_key = load_ascad(
            root + data_root, leakage_model=leakage, byte=byte, train_begin=0, train_end=45000, test_begin=0,
            test_end=nb_traces_attacks)
    elif dataset == 'ASCAD_desync100':
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
            elif dataset == "ASCAD_variable":
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
    apply_grad = False
    apply_lrp = False
    apply_occ = False

    x = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
    num_trace_attrib = x.shape[0]
    x = x[:num_trace_attrib,:,:]

    y = np.array(Y_attack)[:num_trace_attrib]
    old_model = model
    #Gradient
    if apply_grad == True:
        #TODO: put flag = True if this is the first time calling attribution_method for Gradient (aka Salienct Map) and LRP. Work around innvestigate bug.
        num_iteration = int(num_trace_attrib/1000)
        x_attri = x[0:1000,:,:]
        y_attri = y[0:1000]
        attribution_grad, model = attribution_method(x_attri, model, attribution_type='gradient',correct_labels=y_attri, path = save_root_attribution, flag = True) #gradient #one_occlusion #lrp
        print("GRADIENT")
        for iterate in tqdm(range(1,num_iteration)):
            print("counter: {}/{}".format(iterate, num_iteration))
            x_attri = x[iterate*1000:(iterate+1)*1000,:,:]
            y_attri = y[iterate * 1000:(iterate + 1) * 1000]
            attribution_grad_next, _ = attribution_method(x_attri, model, attribution_type='gradient',correct_labels=y_attri, path = save_root_attribution, flag = False) #gradient #one_occlusion #lrp
            attribution_grad =attribution_grad + attribution_grad_next
        np.save(save_root_attribution + "attribution_grad_inn.npy", attribution_grad)
    else:
        attribution_grad = np.load(save_root_attribution + "attribution_grad_inn.npy")
    attribution_grad = attribution_grad/num_trace_attrib #forget to put inside
    fig, ax = plt.subplots(figsize=(15, 7))
    # x_axis = [i for i in range(4980, X_attack.shape[1])]
    x_axis = [i for i in range(X_attack.shape[1])]
    ax.set_xlabel('Sample Points', fontsize=20)
    ax.set_ylabel('Attribution', fontsize=20)
    ax.plot(x_axis, attribution_grad, 'r')
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(15)
    plt.savefig(image_root + 'attribution_grad_inn.png')
    plt.show()
    plt.cla()

    #LRP
    if apply_lrp == True:
        #TODO: put flag = True if this is the first time calling attribution_method for Gradient (aka Salienct Map) and LRP. Work around innvestigate bug.
        num_iteration = int(num_trace_attrib / 1000)
        x_attri = x[0: 1000, :, :]
        y_attri = y[0:1000]
        attribution_lrp, _ = attribution_method(x_attri, model, attribution_type='lrp', correct_labels=y_attri,
                                              path=save_root_attribution, flag=False)  # gradient #one_occlusion #lrp
        print("LRP")
        for iterate in range(1, num_iteration):
            print("counter: {}/{}".format(iterate, num_iteration))
            x_attri = x[iterate * 1000:(iterate + 1) * 1000, :, :]
            y_attri = y[iterate * 1000:(iterate + 1) * 1000]
            attribution_lrp_next, _ = attribution_method(x_attri, model, attribution_type='lrp',correct_labels=y_attri, path = save_root_attribution, flag = False) #gradient #one_occlusion #lrp
            attribution_lrp += attribution_lrp + attribution_lrp_next
        np.save(save_root_attribution + "attribution_lrp.npy", attribution_lrp)
    else:
        attribution_lrp = np.load(save_root_attribution + "attribution_lrp.npy")
    print("attribution_lrp.shape: ", attribution_lrp.shape)
    attribution_lrp = attribution_lrp / num_trace_attrib  # forget to put inside

    fig, ax = plt.subplots(figsize=(15, 7))
    # x_axis = [i for i in range(4980, X_attack.shape[1])]
    x_axis = [i for i in range(X_attack.shape[1])]
    ax.set_xlabel('Sample Points', fontsize=20)
    ax.set_ylabel('Attribution', fontsize=20)
    ax.plot(x_axis, attribution_lrp, 'y')
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(15)
    plt.savefig(image_root + 'attribution_lrp.png')
    plt.show()
    plt.cla()

    #Occlusion
    if apply_occ == True:
        model = old_model
        num_iteration = int(num_trace_attrib / 1000)
        x_attri = x[0: 1000, :, :]
        y_attri = y[0:1000]
        attribution_occ, _ = attribution_method(x_attri, model, attribution_type='one_occlusion', correct_labels=y_attri,
                                                path=save_root_attribution)  # gradient #one_occlusion #lrp
        print("1-Occ")
        for iterate in range(1, num_iteration):
            print("counter: {}/{}".format(iterate,num_iteration) )
            x_attri = x[iterate * 1000:(iterate + 1) * 1000, :, :]
            y_attri = y[iterate * 1000:(iterate + 1) * 1000]
            attribution_occ_next, _ = attribution_method(x_attri, model, attribution_type='one_occlusion', correct_labels=y_attri,
                                                    path=save_root_attribution)  # gradient #one_occlusion #lrp
            attribution_occ += attribution_occ_next
        np.save(save_root_attribution + "attribution_occ.npy", attribution_occ)
    else:
        attribution_occ = np.load(save_root_attribution + "attribution_occ.npy")
    attribution_occ = attribution_occ / num_trace_attrib  # forget to put inside

    fig, ax = plt.subplots(figsize=(15, 7))
    # x_axis = [i for i in range(4980, X_attack.shape[1])]
    x_axis = [i for i in range(X_attack.shape[1])]
    ax.set_xlabel('Sample Points', fontsize=20)
    ax.set_ylabel('Attribution', fontsize=20)
    ax.plot(x_axis, attribution_occ, 'g')
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(15)
    plt.savefig(image_root + 'attribution_occ.png')
    plt.show()
    plt.cla()

