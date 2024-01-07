# main file for fmgene project
# status: in developing 

import sys, os

import numpy as np

from networks import CNN_Wrapper, RNN_Wrapper, M_Wrapper
from utils import read_json
import matplotlib.pyplot as plt


def NeuralNet(config, train):
    print('Dataset', config['type'])
    reports = []
    for exp_idx in range(config['num_exps']):
        config['model_name'] = 'RNN_{}'.format(exp_idx)
        net = RNN_Wrapper(config)
        if train:
            net.train(3)
        else:
            net.load()
        reports += [net.test()]
    return reports
    
def main():
    config = read_json('./config.json')['rnn_gene']
    #config2 = read_json('./config.json')['merged_105']
    
    config['model_name'] = 'RNN_{}'.format(2)
    '''
    config1['model_name'] = 'CNN_{}'.format(2)
    config2['model_name'] = 'merged_{}'.format(2)
    #print(config)
    net = RNN_Wrapper(config)
    net1 = CNN_Wrapper(config1)
    net2 = M_Wrapper(config2)
    #config_gene = read_json('./config.json')['rnn_gene']
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

    net = RNN_Wrapper(config)
    print('train:', len(net.train_data))
    print('valid:', len(net.valid_data))
    print('test :', len(net.test_data))
    # net.get_shap()

    result = NeuralNet(config)
    #print(result)
    print('loss, accuracy', np.mean(result, axis=0))

if __name__ == '__main__':
    main()