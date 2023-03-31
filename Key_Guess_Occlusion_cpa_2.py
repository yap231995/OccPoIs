import os
import sys
from copy import deepcopy

import h5py
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm
from src.cpa import cpa_method,HW
from src.net import cnn_ascad_fix_ID, cnn_ascad_variable_ID
from tf_run import load_ascad, load_chipwhisperer, attack_calculate_metrics, load_aes_hd_ext, load_chipwhisperer_desync, AES_Sbox


if __name__ == "__main__":

    np.random.seed(seed=100)
    root = './'

    dataset = 'ASCAD' #'Chipwhisperer' #ASCAD #ASCAD_variable #AES_HD_ext
    leakage = 'ID'
    non_overfitting = False
    occlusion_type = "input"
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
    save_root_occlusion_iterative_type = save_root_occlusion+iterative_type+'/'

    print("PATH save_root:", save_root_occlusion_iterative_type)
    image_root = save_root_occlusion_iterative_type + 'image/'
    explainability_root = save_root_occlusion_iterative_type + 'explainability/'


    if not os.path.exists(save_root_occlusion):
        os.mkdir(save_root_occlusion)
    if not os.path.exists(save_root_occlusion_iterative_type):
        os.mkdir(save_root_occlusion_iterative_type)
    if not os.path.exists(image_root):
        os.mkdir(image_root)
    if not os.path.exists(explainability_root):
        os.mkdir(explainability_root)

    seed = 0
    random_state = check_random_state(seed)
    nb_traces_attacks = 10000 #10000 #5000 for ASCAD, 2000 for CW #10000 for AES_HD, 55000 for ASCAD_var
    nb_attacks = 100
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
                model = load_model(save_root + '/' + dataset + '_' + leakage + '_CNN_desync_100_801' + '.h5',
                                   compile=False)
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
    important_samplept_further = np.load(save_root_occlusion_iterative_type + "important_samplept_further.npy")  # important_samplept_further

    visualizing_KGO_pt_with_CPA = True
    check_ascad_fixed_id = True
    check_ascad_variable_id = True
    if visualizing_KGO_pt_with_CPA:
        GE_vis = np.zeros(X_attack.shape[1])
        GE_vis[important_samplept_further] = 1
        fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
        x_axis = [i for i in range(X_attack.shape[1])]
        if dataset == "ASCAD_variable":
            CPA_path = './Dataset/ASCAD/CPA/'+dataset+'/'
            pt_r_in_ASCAD_variable = np.load(CPA_path + "pt_r_in_"+dataset+"_"+leakage+".npy")
            r_ASCAD_variable = np.load(CPA_path + "r_"+dataset+"_"+leakage+".npy")
            r_in_ASCAD_variable = np.load(CPA_path + "r_in_"+dataset+"_"+leakage+".npy")
            r_out_ASCAD_variable = np.load(CPA_path + "r_out_"+dataset+"_"+leakage+".npy")
            sb_ASCAD_variable = np.load(CPA_path + "sb_"+dataset+"_"+leakage+".npy")
            sb_r_ASCAD_variable = np.load(CPA_path + "sb_r_"+dataset+"_"+leakage+".npy")
            sb_r_out_ASCAD_variable = np.load(CPA_path + "sb_r_out_"+dataset+"_"+leakage+".npy")

            print("r_ASCAD_variable[187]:", r_ASCAD_variable[187])
            print("sb_r_out_ASCAD_variable[187]:", sb_r_out_ASCAD_variable[187])

            plt.bar(x_axis, GE_vis, label="DeepPoIs", color= "tab:red", width=4)
            plt.plot(x_axis, sb_r_out_ASCAD_variable,label="$Sbox(pt_3\oplus k_3^*) \oplus r_{out}$",color= "tab:purple")
            plt.plot(x_axis, r_out_ASCAD_variable,label="$r_{out}$",color= "tab:blue")
            plt.plot(x_axis, sb_r_ASCAD_variable,label="$Sbox(pt_3\oplus k_3^*) \oplus r$",color= "tab:green")
            plt.plot(x_axis, r_ASCAD_variable, label="$r$",color= "tab:orange")
            plt.plot(x_axis, pt_r_in_ASCAD_variable, label="$pt_3 \oplus k_3^* \oplus r_{in}$", color="tab:brown")
            plt.plot(x_axis, r_in_ASCAD_variable, label="$r_{in}$", color="tab:cyan")
            # plt.plot(x_axis, sb_ASCAD_variable,label="$Sbox(pt_3\oplus k_3^*)$",color= "rosybrown")
            ax.set_xlabel('Sample Points', fontsize=20)
            ax.set_ylabel('(Absolute) Correlation', fontsize=20)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(15)
            ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                      mode="expand", borderaxespad=0, ncol=3, prop={'size': 20})
            plt.savefig(image_root + 'CPA_with_KGO_points_' + dataset + '_' + leakage + '.png')
            plt.show()
        elif dataset == "ASCAD" or dataset == "ASCAD_desync50" or dataset == "ASCAD_desync100":
            dataset_main = "ASCAD"
            CPA_path = './Dataset/ASCAD/CPA/' + dataset_main + '/'
            pt_r_in_ASCAD = np.load(CPA_path + "pt_r_in_" + dataset_main + "_" + leakage + ".npy")
            r_ASCAD = np.load(CPA_path + "r_" + dataset_main + "_" + leakage + ".npy")
            r_in_ASCAD = np.load(CPA_path + "r_in_" + dataset_main + "_" + leakage + ".npy")
            r_out_ASCAD = np.load(CPA_path + "r_out_" + dataset_main + "_" + leakage + ".npy")
            sb_ASCAD = np.load(CPA_path + "sb_" + dataset_main + "_" + leakage + ".npy")
            sb_r_ASCAD = np.load(CPA_path + "sb_r_" + dataset_main + "_" + leakage + ".npy")
            sb_r_out_ASCAD = np.load(CPA_path + "sb_r_out_" + dataset_main + "_" + leakage + ".npy")

            plt.xticks(np.arange(0, X_attack.shape[1], 50))


            plt.bar(x_axis, GE_vis, label="DeepPoIs", color= "tab:red", width=1.7)
            plt.plot(x_axis, sb_r_out_ASCAD, label="$Sbox(pt_3\oplus k^*) \oplus r_{out}$",color= "tab:purple")
            plt.plot(x_axis, r_out_ASCAD, label="$r_{out}$",color= "tab:blue")
            plt.plot(x_axis, r_in_ASCAD, label="$r_{in}$", color="tab:olive")
            plt.plot(x_axis, sb_r_ASCAD, label="$Sbox(pt_3\oplus k^*) \oplus r$",color= "tab:green")
            plt.plot(x_axis, r_ASCAD, label="$r$",color= "tab:orange")
            plt.ylim(0, 1)
            ax.set_xlabel('Sample Points', fontsize=20)
            ax.set_ylabel('(Absolute) Correlation', fontsize=20)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(15)
            ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                      mode="expand", borderaxespad=0, ncol=3, prop={'size': 20})
            plt.savefig(image_root + 'CPA_with_KGO_points_' + dataset + '_' + leakage + '.png')
            plt.show()
        elif dataset == "Chipwhisperer" or dataset == "Chipwhisperer_desync50" or dataset == "Chipwhisperer_desync25":
            CPA_path = './Dataset/Chipwhisperer/CPA/'
            # CPA_CW = np.load(CPA_path + "CPA_attack" + leakage + "_" + dataset+ ".npy")
            CPA_CW = np.load(CPA_path + "CPA_attack" + leakage + "_Chipwhisperer.npy")
            print("important_samplept_further:", important_samplept_further)
            print("CPA_CW[x]:", CPA_CW[important_samplept_further])
            # print(ok)
            # CPA_CW_pt = np.load(CPA_path + "CPA_attack" + leakage + "_" + dataset+"_plaintext.npy")
            CPA_CW_pt = np.load(CPA_path + "CPA_attack" + leakage + "_Chipwhisperer_plaintext.npy")

            plt.plot(x_axis, CPA_CW_pt, label="$pt\oplus k^*$", color = "tab:blue")
            plt.plot(x_axis, CPA_CW, label="$Sbox(pt\oplus k^*)$", color = "tab:grey")
            if leakage == "ID":
                plt.bar(x_axis, GE_vis, label="DeepPoIs", color="tab:red", width=15)
            elif leakage == "HW":
                plt.bar(x_axis, GE_vis, label="DeepPoIs", color="tab:green", width=15)

            ax.set_xlabel('Sample Points', fontsize=20)
            ax.set_ylabel('(Absolute) Correlation', fontsize=20)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(15)
            ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                      mode="expand", borderaxespad=0, ncol=3, prop={'size': 20})
            # plt.tight_layout(rect=[0, 0, 0.75, 1])
            plt.savefig(image_root + 'CPA_with_KGO_points_' + dataset + '_' + leakage + '.png')
            plt.show()
            plt.cla()
        elif dataset == "AES_HD_ext":
            cpa_last_round_only = False
            if cpa_last_round_only == True:
                CPA_path = './Dataset/AES_HD_ext/CPA/'
                CPA_CW = np.load(CPA_path + "CPA_attack" + leakage + "_" + dataset+ "_all_key.npy")
                plt.bar(x_axis, GE_vis, label="DeepPoIs", color= "tab:red", width=3.8)
                flag = False
                for k in range(256):
                    if k != correct_key and flag == False:
                        plt.plot(x_axis, CPA_CW[k, :], color="lightblue",
                                 label=r"$Sbox(ct_{15}\oplus k) \oplus ct_{11}$ with $k\neq k^*_{15}$")
                        flag = True
                    elif k != correct_key and flag == True:
                        plt.plot(x_axis, CPA_CW[k, :], color="lightblue")
                plt.plot(x_axis, CPA_CW[correct_key,:],color = "b", label="$Sbox(ct_{15}\oplus k^*_{15}) \oplus ct_{11}$")
                plt.ylim(0, 0.1)

                ax.set_xlabel('Sample Points', fontsize=20)
                ax.set_ylabel('(Absolute) Correlation', fontsize=20)
                for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                    label.set_fontsize(15)
                ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                        mode="expand", borderaxespad=0, ncol=3, prop={'size': 20})
                plt.savefig(image_root + 'CPA_with_KGO_points_'+dataset+'_'+leakage+'.png')
                plt.show()
                plt.cla()


    #Visualize what are the sample points related to ASCADf (ID)
    if check_ascad_fixed_id  == True:
        def rout_label_cpa(plt,k,leakage,root = 'Dataset/ASCAD/ASCAD.h5'):
            in_file = h5py.File(root, "r")
            mask_values = np.array(in_file['Attack_traces/metadata'][:]['masks'])[:plt.shape[0], :]
            r_out = mask_values[:, 15] ##r_out for ASCAD.h5
            num_traces = plt.shape[0]
            labels_for_m1 = np.zeros(num_traces)
            labels_for_m2 = np.zeros(num_traces)
            for i in range(num_traces):
                if leakage == 'HW':
                    labels_for_m1[i] = HW(AES_Sbox[plt[i] ^ k]^r_out[i])
                    labels_for_m2[i] = HW(r_out[i])
                elif leakage == 'ID':
                    labels_for_m1[i] = AES_Sbox[plt[i] ^ k]^r_out[i]
                    labels_for_m2[i] = r_out[i]
            return labels_for_m1, labels_for_m2

        def r_label_cpa(plt,k,leakage,root = 'Dataset/ASCAD/ASCAD.h5'):
            in_file = h5py.File(root, "r")
            mask_values = np.array(in_file['Attack_traces/metadata'][:]['masks'])[:plt.shape[0], :]
            r = mask_values[:, 0]
            num_traces = plt.shape[0]
            labels_for_m1 = np.zeros(num_traces)
            labels_for_m2 = np.zeros(num_traces)
            for i in range(num_traces):
                if leakage == 'HW':
                    labels_for_m1[i] = HW(AES_Sbox[plt[i] ^ k]^r[i])
                    labels_for_m2[i] = HW(r[i])
                elif leakage == 'ID':
                    labels_for_m1[i] = AES_Sbox[plt[i] ^ k]^r[i]
                    labels_for_m2[i] = r[i]
            return labels_for_m1, labels_for_m2

        if dataset == "ASCAD" and leakage == "ID":
            important_samplept_further = sorted(important_samplept_further, reverse = False)
            print("important_samplept: ", important_samplept_further)
            cut_out_trace = deepcopy(X_attack)[:, important_samplept_further]
            total_samplept = cut_out_trace.shape[1]
            number_of_traces = cut_out_trace.shape[0]
            print("total_samplept", total_samplept)
            print("number_of_traces", number_of_traces)


            cpa_m2 = np.zeros((256, total_samplept))
            cpa_m1 = np.zeros((256, total_samplept))
            fig, ax = plt.subplots(figsize=(15, 7))
            x_axis = [i for i in range(total_samplept)]
            print("CPA per key:")
            save_value = False
            mask_type = "rout" #r #rout
            if save_value == True:
                for k in tqdm(range(256)):
                    if mask_type == "rout":
                        labels_for_m1, labels_for_m2 = rout_label_cpa(plt_attack, k, leakage, root='Dataset/ASCAD/ASCAD.h5')
                    elif mask_type == "r":
                        labels_for_m1, labels_for_m2 = r_label_cpa(plt_attack, k, leakage,
                                                                      root='Dataset/ASCAD/ASCAD.h5')

                    # for number_of_traces in tqdm(range(0, cut_out_trace.shape[0])):
                    cpa_m1[k, :] = cpa_method(total_samplept, number_of_traces, labels_for_m1, cut_out_trace)
                    cpa_m2[k, :] = cpa_method(total_samplept, number_of_traces, labels_for_m2, cut_out_trace)
                np.save(explainability_root + "cpa_attack_sbox_pt_k_"+mask_type+".npy", cpa_m1)
                np.save(explainability_root +  "cpa_attack_"+mask_type+".npy" , cpa_m2)
            else:
                cpa_m1 = np.load(explainability_root + "cpa_attack_sbox_pt_k_"+mask_type+".npy")
                cpa_m2 = np.load(explainability_root + "cpa_attack_"+mask_type+".npy" )


            flag = False
            if mask_type == "r":
                mask2_name_other_key =  r"$Sbox(pt_3\oplus k) \oplus r$ with $k \neq k_3^*$"
                mask2_name_correct_key = "$Sbox(pt_3\oplus k_3^*) \oplus r$"
                mask1_name = "$r$"
                color_other_key = "lightgreen"
                color_Sbox = "tab:green"
                color_r = "tab:orange"
            elif mask_type == "rout":
                mask2_name_other_key = r"$Sbox(pt_3\oplus k) \oplus r_{out}$ with $k \neq k_3^*$"
                mask2_name_correct_key = "$Sbox(pt_3\oplus k_3^*) \oplus r_{out}$"
                mask1_name = "$r_{out}$"
                color_other_key = "plum"
                color_Sbox = "tab:purple"
                color_r = "tab:blue"


            for k in tqdm(range(256)):
                if k != correct_key and flag == False:
                    ax.plot(x_axis, cpa_m1[k, :], color=color_other_key, label=mask2_name_other_key)
                    flag = True
                elif k != correct_key and flag == True:
                    ax.plot(x_axis, cpa_m1[k, :], color=color_other_key)
            ax.plot(x_axis, cpa_m1[correct_key, :], color=color_Sbox, label=mask2_name_correct_key)
            ax.plot(x_axis, cpa_m2[correct_key, :], color=color_r, label=mask1_name)

            print( np.argmax(cpa_m1[:, 1]))
            print( correct_key)

            ax.set_xlabel('Sample Point', fontsize=20)
            ax.set_ylabel('(Absolute) Correlation', fontsize=20)
            plt.xticks(x_axis, important_samplept_further)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                    label.set_fontsize(10)
            ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                      mode="expand", borderaxespad=0, ncol=4, prop={'size': 20})
            plt.savefig(image_root + 'CPA_sbox_pt_k_'+mask_type+'_' + dataset + "_" + leakage+".png")
            plt.show()
        else:
            print("This is not ASCAD (ID), so will pass this part.")



    if check_ascad_variable_id  == True:

        def plaintext_label_cpa(plt,k,leakage,root = 'Dataset/ASCAD/ASCAD_variable.h5'):
            in_file = h5py.File(root, "r")
            mask_values = np.array(in_file['Attack_traces/metadata'][:]['masks'])[:plt.shape[0], :]
            r_in = mask_values[:, 16]  ##r_in for ASCAD_variable.h5
            num_traces = plt.shape[0]
            labels_for_m1 = np.zeros(num_traces)
            labels_for_m2 = np.zeros(num_traces)
            for i in range(num_traces):
                if leakage == 'HW':
                    labels_for_m1[i] = HW(plt[i] ^ k^r_in[i])
                    labels_for_m2[i] = HW(r_in[i])
                elif leakage == 'ID':
                    labels_for_m1[i] = plt[i] ^ k^r_in[i]
                    labels_for_m2[i] = r_in[i]
            return labels_for_m1, labels_for_m2

        if dataset == "ASCAD_variable" and leakage == "ID":
            important_samplept_further = sorted(important_samplept_further, reverse = False)
            print("important_samplept: ", important_samplept_further)
            cut_out_trace = deepcopy(X_attack)[:, important_samplept_further]
            total_samplept = cut_out_trace.shape[1]
            number_of_traces = cut_out_trace.shape[0]
            print("total_samplept", total_samplept)
            print("number_of_traces", number_of_traces)


            cpa_m2 = np.zeros((256, total_samplept))
            cpa_m1 = np.zeros((256, total_samplept))
            fig, ax = plt.subplots(figsize=(15, 7))
            x_axis = [i for i in range(total_samplept)]
            print("CPA per key:")
            save_value = False
            if save_value == True:
                for k in tqdm(range(256)):
                    labels_for_m1, labels_for_m2 = plaintext_label_cpa(plt_attack, k, leakage, root='Dataset/ASCAD/ASCAD_variable.h5')
                    # for number_of_traces in tqdm(range(0, cut_out_trace.shape[0])):
                    cpa_m1[k, :] = cpa_method(total_samplept, number_of_traces, labels_for_m1, cut_out_trace)
                    cpa_m2[k, :] = cpa_method(total_samplept, number_of_traces, labels_for_m2, cut_out_trace)
                np.save(explainability_root + "cpa_attack_pt_k_rin.npy", cpa_m1)
                np.save(explainability_root + "cpa_attack_rin.npy", cpa_m2)
            else:
                cpa_m1 = np.load(explainability_root + "cpa_attack_pt_k_rin.npy" )
                cpa_m2 = np.load(explainability_root + "cpa_attack_rin.npy" )


            flag = False
            for k in tqdm(range(256)):
                if k != correct_key and flag == False:
                    ax.plot(x_axis, cpa_m1[k, :], 'tan', label = r"$pt_3\oplus k \oplus r_{in}$ with $k \neq k_3^*$")
                    flag = True
                elif k != correct_key and flag == True:
                    ax.plot(x_axis, cpa_m1[k, :], 'tan')
            ax.plot(x_axis, cpa_m1[correct_key, :], 'tab:brown', label="$pt_3\oplus k_3^* \oplus r_{in}$")
            ax.plot(x_axis, cpa_m2[correct_key, :], 'tab:cyan', label="$r_{in}$")
            ax.set_xlabel('Sample Point', fontsize=20)
            ax.set_ylabel('(Absolute) Correlation', fontsize=20)
            label_x = [item.get_text() for item in ax.get_xticklabels()]
            # print("label_x", label_x)
            for i in range(len(important_samplept_further)):
                label_x[i+1] = str(important_samplept_further[i])
            ax.set_xticklabels(label_x)
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(15)
            # ax.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0, prop={'size': 25})
            ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                      mode="expand", borderaxespad=0, ncol=3, prop={'size': 20})
            plt.savefig(image_root + 'CPA_pt_k_rin_' + dataset + "_" + leakage+".png")
            plt.show()
        else:
            print("This is not ASCAD_variable ID (overfitting), so will pass this part.")
