# main file for fmgene project
# status: OK, tuning

import sys, os
import json

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve, auc
from networks import CNN_Wrapper, MLP_Wrapper, RNN_Wrapper, M_Wrapper, CNN_paper, MLP_fMRI_Wrapper
from utils import read_json
from numpy import linspace, interp

def plot_AUC(auc_data, dataset_name):
    plt.figure()
    mean_fpr = np.linspace(0, 1, 100)  # Common set of thresholds for interpolation

    for model_name, roc_data in auc_data.items():
        tprs_interp = []
        aucs = []

        for fpr, tpr in roc_data:
            for fpr, tpr in zip(fpr, tpr):
                if not np.isnan(fpr).any() and not np.isnan(tpr).any():
                    tpr_interp = np.interp(mean_fpr, fpr, tpr)
                    tpr_interp[0] = 0.0  # Ensure that the first value starts from 0
                    tprs_interp.append(tpr_interp)
                    aucs.append(auc(fpr, tpr))

        mean_tpr = np.mean(tprs_interp, axis=0)
        mean_tpr[-1] = 1.0  # Ensure that the last value ends at 1
        std_tpr = np.std(tprs_interp, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        plt.plot(mean_fpr, mean_tpr, label=f'{model_name}: {mean_auc:.4f} (± {std_auc:.4f})')
        plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {dataset_name}')
    plt.legend(loc='lower right')
    plt.savefig(f'plots/roc_curve_{dataset_name}.png')
    plt.close()

def print_net(net):
    print('train:', len(net.train_data))
    print('valid:', len(net.valid_data))
    print('test :', len(net.test_data))
    print('samples:', net.train_data.index_list)

def NeuralNet(config, train, wrapper):
    print('Dataset', config['type'])
    reports = []
    accuracies = []
    roc_aucs = []
    precisions = []
    recalls = []
    f1_scores = []
    mccs = []
    fpr_list = []
    tpr_list = []
    
    config['model_name'] += config['type']
    model_name = config['model_name']

    for exp_idx in range(config['num_exps']):
        config['model_name'] = model_name + str(exp_idx)
        config['seed'] += exp_idx*2
        net = wrapper(config)
        
        if train:
            net.train()
        else:
            net.load()

        _, _, preds, labels = net.test(raw=True)
        
        preds_rounded = np.round(preds)
        
        accuracy = accuracy_score(labels, preds_rounded)
        labels_binary = [np.argmax(label) for label in labels]
        preds_binary = [np.argmax(pred) for pred in preds]
        fpr, tpr, _ = roc_curve(labels[:, 1], preds[:, 1])
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        precision = precision_score(labels_binary, preds_binary)
        recall = recall_score(labels_binary, preds_binary)
        f1 = f1_score(labels_binary, preds_binary)
        mcc = matthews_corrcoef(labels_binary, preds_binary)

        try:
            roc_auc = roc_auc_score(labels[:,1], preds[:,1])
        except:
            print('skipped one')
            continue

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        mccs.append(mcc)
        roc_aucs.append(roc_auc)

        reports.append({'accuracy': accuracy, 'roc_auc': roc_auc, 'precision': precision, 
                        'recall': recall, 'f1_score': f1, 'mcc': mcc})
        config['model_name'] = model_name

    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    avg_precision = np.mean(precisions)
    std_precision = np.std(precisions)
    avg_recall = np.mean(recalls)
    std_recall = np.std(recalls)
    avg_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    avg_mcc = np.mean(mccs)
    std_mcc = np.std(mccs)
    avg_roc_auc = np.mean(roc_aucs)
    std_roc_auc = np.std(roc_aucs)

    model_info_str = (f"{model_name} Average Accuracy: {avg_accuracy:.4f} (±{std_accuracy:.4f}), "
                      f"Average Precision: {avg_precision:.4f} (±{std_precision:.4f}), "
                      f"Average Recall: {avg_recall:.4f} (±{std_recall:.4f}), "
                      f"Average F1 Score: {avg_f1:.4f} (±{std_f1:.4f}), "
                      f"Average MCC: {avg_mcc:.4f} (±{std_mcc:.4f}), "
                      f"Average ROC AUC: {avg_roc_auc:.4f} (±{std_roc_auc:.4f})\n\n")
    return reports, avg_accuracy, std_accuracy, avg_precision, std_precision, avg_recall, std_recall, avg_f1, std_f1, avg_mcc, std_mcc, avg_roc_auc, std_roc_auc, fpr_list, tpr_list, model_info_str
    # print(net.net)
    # net.random_forest()
    # net.dataset_ratio()

def DecisionNet(configs, train, wrappers):
    print('Dataset', configs[0]['type'])
    reports = []
    for i, config in enumerate(configs):
        all_preds = []
        labels = None
        model_name = config['model_name']
        wrapper = wrappers[i]
        for exp_idx in range(config['num_exps']):
            config['model_name'] = model_name + str(exp_idx)
            net = wrapper(config)
            net.load()
            _, _, preds, label = net.test(raw=True)
            all_preds.append(preds)
            if labels is None:
                labels = label

        # Perform majority voting
        majority_votes = np.round(np.mean(all_preds, axis=0))
        
        # Calculate accuracy and ROC AUC score
        accuracy = accuracy_score(labels, majority_votes)
        roc_auc = roc_auc_score(labels, majority_votes)

        reports.append((accuracy, roc_auc))

    # Aggregate the results over all experiments
    avg_accuracy = np.mean([report[0] for report in reports])
    avg_roc_auc = np.mean([report[1] for report in reports])
    print(f'Average Accuracy: {avg_accuracy:.4f}, Average ROC AUC: {avg_roc_auc:.4f}')
    return avg_accuracy, avg_roc_auc

# Usage example:
# configs = [config1, config2, ...] # Your list of configs for each model
# wrappers = [CNN_Wrapper, RNN_Wrapper, ...] # Your list of model wrappers
# decision_net_accuracy, decision_net_roc_auc = DecisionNet(configs, train=False, wrappers=wrappers)

def main():
    auc_data = {
        'Resting': {"Our's": [], "Gupta's": [], "Baseline_RNN": [], "Baseline_MLP": []},
        'MoCo': {"Our's": [], "Gupta's": [], "Baseline_RNN": [], "Baseline_MLP": []},
        'All': {"Our's": [], "Gupta's": [], "Baseline_RNN": [], "Baseline_MLP": []}
    }

    train = True
    train = False

    out = ''
    # Train CNN
    # Resting
    config_cnn = read_json('./config.json')['cnn_105']
    result = NeuralNet(config_cnn, train=train, wrapper=CNN_Wrapper)
    out += result[-1]
    auc_data['Resting']["Our's"].append((result[-3], result[-2]))

    # MoCo
    config_cnn = read_json('./config.json')['cnn_6720']
    result = NeuralNet(config_cnn, train=train, wrapper=CNN_Wrapper)
    out += result[-1]
    auc_data['MoCo']["Our's"].append((result[-3], result[-2]))

    # All
    config_cnn = read_json('./config.json')['cnn_1']
    result = NeuralNet(config_cnn, train=train, wrapper=CNN_Wrapper)
    out += result[-1]
    auc_data['All']["Our's"].append((result[-3], result[-2]))
    # print(out)
    # sys.exit()


    # Train Gupta's CNN
    # Resting
    config_cnn = read_json('./config.json')['cnn_paper']
    config_cnn['type'] = 'Resting'
    result = NeuralNet(config_cnn, train=train, wrapper=CNN_paper)
    out += result[-1]
    auc_data['Resting']["Gupta's"].append((result[-3], result[-2]))

    # MoCo
    config_cnn = read_json('./config.json')['cnn_paper']
    config_cnn['type'] = 'MoCo'
    result = NeuralNet(config_cnn, train=train, wrapper=CNN_paper)
    out += result[-1]
    auc_data['MoCo']["Gupta's"].append((result[-3], result[-2]))
    
    # All
    config_cnn = read_json('./config.json')['cnn_paper']
    config_cnn['type'] = 'all'
    result = NeuralNet(config_cnn, train=train, wrapper=CNN_paper)
    out += result[-1]
    auc_data['All']["Gupta's"].append((result[-3], result[-2]))
    # print(out)


    # Train RNN
    # Resting
    config_rnn = read_json('./config.json')['rnn_fmri']
    config_rnn['type'] = 'Resting'
    config_rnn['input_size'] = 64 * 64 * 48
    result = NeuralNet(config_rnn, train=train, wrapper=RNN_Wrapper)
    out += result[-1]
    auc_data['Resting']["Baseline_RNN"].append((result[-3], result[-2]))
    
    # MoCo
    config_rnn = read_json('./config.json')['rnn_fmri']
    config_rnn['type'] = 'MoCo'
    config_rnn['input_size'] = 64 * 64 * 24
    result = NeuralNet(config_rnn, train=train, wrapper=RNN_Wrapper)
    out += result[-1]
    auc_data['MoCo']["Baseline_RNN"].append((result[-3], result[-2]))
    
    # All
    config_rnn = read_json('./config.json')['rnn_fmri']
    config_rnn['type'] = 'all'
    config_rnn['input_size'] = 64 * 64 * 24
    result = NeuralNet(config_rnn, train=train, wrapper=RNN_Wrapper)
    out += result[-1]
    auc_data['All']["Baseline_RNN"].append((result[-3], result[-2]))
    # print(out)


    # Train MLP
    # Resting
    config_mlp = read_json('./config.json')['mlp_fmri']
    config_mlp['type'] = 'Resting'
    config_mlp['input_size'] = 64 * 64 * 48
    result = NeuralNet(config_mlp, train=train, wrapper=MLP_fMRI_Wrapper)
    out += result[-1]
    auc_data['Resting']["Baseline_MLP"].append((result[-3], result[-2]))
    
    # MoCo
    config_mlp = read_json('./config.json')['mlp_fmri']
    config_mlp['type'] = 'MoCo'
    config_mlp['input_size'] = 64 * 64 * 24
    result = NeuralNet(config_mlp, train=train, wrapper=MLP_fMRI_Wrapper)
    out += result[-1]
    auc_data['MoCo']["Baseline_MLP"].append((result[-3], result[-2]))
    
    # All
    config_mlp = read_json('./config.json')['mlp_fmri']
    config_mlp['type'] = 'all'
    config_mlp['input_size'] = 64 * 64 * 24
    result = NeuralNet(config_mlp, train=train, wrapper=MLP_fMRI_Wrapper)
    out += result[-1]
    auc_data['All']["Baseline_MLP"].append((result[-3], result[-2]))
    print(out)

    # print(auc_data)

    for dataset in auc_data:
        plot_AUC(auc_data[dataset], dataset)


    # Train MLP
    # config_rnn = read_json('./config.json')['mlp_gene']
    # result = NeuralNet(config_rnn, train=train, wrapper=MLP_Wrapper)

    # Train RNN
    # config_rnn = read_json('./config.json')['rnn_gene']
    # result = NeuralNet(config_rnn, train=train, wrapper=RNN_Wrapper)

    # Train fused network - feature level
    # config = read_json('./config.json')['merged_105']
    # result = NeuralNet(config, train=train, wrapper=M_Wrapper)
    # print('merged loss, accuracy', np.mean(result, axis=0))

    # Fused network - decision level

    # result = DecisionNet(configs=[config_cnn, config_rnn], train=train, wrappers=[CNN_Wrapper, RNN_Wrapper])
    # print('decision net: avg_accuracy, avg_roc_auc', result)

    
    '''
    # # net1.get_map() #visualize the feature maps generated by convolutional layers
    # # net.get_shap()
    fpr1, tpr1, threshold1, auc1  = net1.get_curve()
    fpr2, tpr2, threshold2, auc2 = net2.get_curve(config2)
    fpr, tpr, threshold, auc = net.get_curve(config)

    plt.plot(fpr,tpr,label="gene model, AUC="+str(auc))
    plt.plot(fpr1, tpr1, label = "fMRI model, AUC="+str(auc1))
    plt.plot(fpr2, tpr2, label = "combined model, AUC="+str(auc2))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate' )
    plt.legend()
    plt.savefig('gene_AUC.png')
    '''
if __name__ == '__main__':
    main()