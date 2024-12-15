# network wrapper for fmgene project
# status: in-developing

import torch
import os, sys
import shutil
import math
import shap
import pickle
import glob

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from matplotlib import patches
from torch.utils.data import DataLoader

from models import _FCCNN, _MLP_gene, _Merged, _RNN_gene, _RNN_fMRI, _MLP_fMRI
from dataloader import fmri_gene_data
from utils import read_json
# from dataloader import fmri_gene_data_v2 as fmri_gene_data #this has less samples, since it's id are from commons.
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

device = torch.device('cuda:0' if (torch.cuda.is_available()) else "cpu")


from models import BinaryClassifier3D

def dataloader_to_xy(dataloader):
    X, y = [], []
    for fmri, gene, label in dataloader:
        gene = gene.view(-1, 500)  # Flatten the gene data
        X.extend(gene.numpy())
        y.extend(label.numpy())
    return np.array(X), np.array(y)

class Wrapper:
    # SWEEP here for placeholders
    def __init__(self, config, SWEEP=False):
        self.config = config
        self.checkpoint_dir = "./checkpoint_dir/"
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.checkpoint_dir += "{}/".format(config['model_name'])
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.output_dir = self.checkpoint_dir + "output_dir/"
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        torch.manual_seed(config['seed'])
        self.prepare_dataloader()

        self.net = _FCCNN(config).to(device, dtype=torch.float)
        

        self.criterion = nn.BCELoss().to(device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=config["lr_cnn"], weight_decay=config["weight_decay_cnn"])

    def prepare_dataloader(self):
        # only train data, for test, valid, wait until ADNI data are ready
        self.train_data = fmri_gene_data(self.config, 'train', 'fmri')
        self.train_dataloader = DataLoader(self.train_data, batch_size=self.config['batch_size'], shuffle=False, drop_last=True)

        self.valid_data = fmri_gene_data(self.config, 'valid', 'fmri')
        self.valid_dataloader = DataLoader(self.valid_data, batch_size=1, shuffle=False, drop_last=True)

        self.test_data = fmri_gene_data(self.config, 'test', 'fmri')
        self.test_dataloader = DataLoader(self.test_data, batch_size=1, shuffle=False, drop_last=True)
    
    def count_trainable_params(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def dataset_ratio(self):
        # Function to calculate and print the label ratio for each dataset split
        def calculate_ratio(dataloader, dataloader_name):
            count = 0
            label_str = ''
            for _, _, labels in dataloader:
                for label in labels:
                    label_str += str(int(label.item())) + ' '
                    count += label.item()
            total = len(dataloader) * dataloader.batch_size
            print(f'{dataloader_name} Set - Labels: {label_str.strip()}; 1s report: Count: {count}, Total: {total}, Percentage: {count / total if total > 0 else 0}\n')

        # Calculate and print stats for each split
        calculate_ratio(self.train_dataloader, 'Train')
        calculate_ratio(self.valid_dataloader, 'Validation')
        calculate_ratio(self.test_dataloader, 'Test')

    def load(self, dir=None, fixed=False, verb=0):
        if dir:
            # need to update
            if verb:
                print("searching models from", dir)
            dir = [
                glob.glob(dir[0] + "*_*.pth")[0],
            ]
            print("might need update")
        else:
            if verb:
                print("searching models from", self.checkpoint_dir)
            dir = [
                glob.glob(self.checkpoint_dir + "*_*.pth")[0],
            ]
        if verb:
            print('\tloading from ', dir[0])
        self.epoch = dir[0].split("_")[-1].split(".")[0]
        st_d = torch.load(dir[0])
        # del st['l2.weight']
        self.net.load_state_dict(st_d, strict=False)
        if fixed:
            print("need update")
            sys.exit()
            ps = []
            for n, p in self.model.named_parameters():
                if (
                    n == "l2.weight"
                    or n == "l2.bias"
                    or n == "l1.weight"
                    or n == "l1.bias"
                ):
                    ps += [p]
                    # continue
                else:
                    pass
                    p.requires_grad = False
            self.optimizer = optim.SGD(ps, lr=self.lr, weight_decay=0.01)

    def train(self, verbose=1):
        print('training...')

        self.optimal_valid_metric = np.inf
        self.optimal_epoch = -1

        for self.epoch in range(1, self.config['epochs']+1):
            self.train_model_epoch()
            val_loss = self.valid_model_epoch()
            self.save_checkpoint(val_loss)
            if self.epoch % 10 == 0:
                print('epoch {}: valid_loss ='.format(self.epoch), '%.3f' % (val_loss))
            if self.config['SWEEP']:
                pass
            
        print('best model saved at {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric.item())

    def train_model_epoch(self):
        self.net.train(True)
        for fmri, gene, label in self.train_dataloader:
            #fmri, label = fmri.to(device, dtype=torch.float), F.one_hot(label.to(device, dtype=torch.long), num_classes=2).float()
            fmri, gene, label = fmri.to(device, dtype=torch.float), gene.to(device, dtype=torch.float),  F.one_hot(label.to(device, dtype=torch.long), num_classes=2).float()
            # fmri, gene, label = fmri.to(device, dtype=torch.float), gene.to(device, dtype=torch.float), F.one_hot(label.to(device, dtype=torch.float))
            self.net.zero_grad()
            preds = self.net(fmri, gene)
            # print(preds)
            # print(label)
            loss = self.criterion(preds, label)
            loss.backward()
            self.optimizer.step()
    
    def valid_model_epoch(self):
        #basically same as train, but on valid_dataloader
        self.net.eval() #so dr is off
        loss_all = []
        with torch.no_grad():
            for fmri, gene, label in self.valid_dataloader:
                #fmri, label = fmri.to(device, dtype=torch.float), F.one_hot(label.to(device, dtype=torch.long), num_classes=2).float()
                fmri, gene, label = fmri.to(device, dtype=torch.float), gene.to(device, dtype=torch.float),  F.one_hot(label.to(device, dtype=torch.long), num_classes=2).float()

                preds = self.net(fmri, gene)
                loss = self.criterion(preds, label)
                # print(loss)
                # print(preds)
                # print(label)
                loss_all += [loss.item()]
        return np.mean(loss_all)

    def test(self, raw=False):
        self.net.eval() # so dropout is off
        loss_all = []
        accu_all = []
        predictions = []
        true_labels = []
        with torch.no_grad():
            for fmri, gene, label in self.test_dataloader:
                fmri, gene, label = fmri.to(device, dtype=torch.float), gene.to(device, dtype=torch.float), F.one_hot(label.to(device, dtype=torch.long), num_classes=2).float()

                preds = self.net(fmri, gene)
                loss = self.criterion(preds, label)
                loss_all.append(loss.item())
                acc_test = accuracy_score(label.detach().cpu().numpy(), preds.detach().cpu().numpy()>0.5)
                accu_all.append(acc_test)

                # Store raw predictions and labels
                predictions.append(preds.detach().cpu().numpy())
                true_labels.append(label.detach().cpu().numpy())

        # Flatten the lists of predictions and true labels
        predictions = np.concatenate(predictions, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)

        if raw:
            return np.mean(loss_all), np.mean(accu_all), predictions, true_labels
        return np.mean(loss_all), np.mean(accu_all)

    def save_checkpoint(self, loss):
        score = loss
        if score <= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_metric = score

            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith(".pth"):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(
                self.net.state_dict(),
                "{}{}_{}.pth".format(self.checkpoint_dir, self.config['model_name'], self.optimal_epoch)
            )
    
    def get_curve(self):
        model = _FCCNN(self.config)
        model.load_state_dict(torch.load('/gpfs/gibbs/pi/gerstein/xz584/fmgene/checkpoint_dir/CNN_1/CNN_1_d_3.pth', map_location=device))
        model = model.to(device)
        model.eval()
        probs = []
        labels = []
        for fmri, gene, label in self.test_dataloader:
            #label = F.one_hot(label.to(device, dtype=torch.long), num_classes=2).float()
            fmri = fmri.to(device, dtype=torch.float)
            gene = 0
            gene = torch.tensor(gene, dtype=torch.int8)
            gene = gene.to(device, dtype=torch.float)
            pred = model(fmri, gene)
            pred = pred.squeeze(-1).detach().numpy()
            if label.numpy()[0] == True:
                labels.append(1)
            else:
                labels.append(0)
            probs.append(pred[0][1])
        auc = roc_auc_score(labels, probs)
        fpr, tpr, threshold = roc_curve(labels, probs)
        return fpr, tpr, threshold, auc
    
    def get_map(self):
        #load model 
        model = _FCCNN(self.config)
        model.load_state_dict(torch.load('/gpfs/gibbs/pi/gerstein/xz584/fmgene/checkpoint_dir/CNN_1/CNN_1_d_3.pth', map_location=device))
        model = model.to(device)
        model.eval()
        weights = []
        layers = []
        count = 0
        children = list(model.children())
        for i in range(len(children)):
            if type(children[i]) == nn.Conv3d:
                count += 1
                weights.append(children[i].weight)
                layers.append(children[i])
      
        results = []
        stage = []
        count = 0
        for fmri, gene, label in self.test_dataloader:
            new = []
            path = '/gpfs/gibbs/pi/gerstein/xz584/fmgene/feature_maps/' + str(count) + '/'
            try:
                os.mkdir(path)
            except:
                pass
            fmri = fmri.to(device, dtype=torch.float)
            #print(fmri.shape)
            for layer in layers:
                fmri = layer(fmri)
                new.append(fmri)
                stage.append(str(layer))
                            
            after = []
            for img in new:
                #This is one participants 
                img = img.squeeze(0)
                gray = torch.sum(img, 0)
                gray = gray/ img.shape[0]
                after.append(gray.data.cpu().numpy())
            
            for i in range(len(after)):
                img = after[i]
                timepoints = int(img.shape[2]) 
                nrows = int(math.ceil(int(timepoints)/5))
                fig, axes = plt.subplots(ncols=5, nrows=nrows, figsize=(timepoints, 10))
                for t, ax in enumerate(axes.flatten()):
                    if t < timepoints:
                        im = ax.imshow(img[ : , :, t].T, origin='lower')
                        box = patches.Rectangle((38, timepoints), 2, 2,linewidth=2, edgecolor='r', facecolor='none')
                        ax.add_patch(box)
                        ax.axis('off')
                        ax.set_title('t = %i' % t, fontsize=20)
                fig.tight_layout()
                #plt.imshow()
                plt.colorbar(im, ax=axes.ravel().tolist())
                plt.savefig(path + str(i) + '-' + stage[i].split('(')[0] + '.png', bbox_inches='tight')
                plt.close()
            count += 1

class CNN_Wrapper(Wrapper):
    pass

class RNN_Wrapper(Wrapper):
    def __init__(self, config, SWEEP=False):
        self.config = config
        self.checkpoint_dir = "./checkpoint_dir/"
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.checkpoint_dir += "{}/".format(config['model_name'])
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.output_dir = self.checkpoint_dir + "output_dir/"
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        torch.manual_seed(config['seed'])
        self.prepare_dataloader()

        self.net = _RNN_fMRI(config).to(device, dtype=torch.float)
        self.criterion = nn.BCELoss().to(device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

class MLP_fMRI_Wrapper(Wrapper):
    def __init__(self, config, SWEEP=False):
        self.config = config
        self.checkpoint_dir = "./checkpoint_dir/"
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.checkpoint_dir += "{}/".format(config['model_name'])
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.output_dir = self.checkpoint_dir + "output_dir/"
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        torch.manual_seed(config['seed'])
        self.prepare_dataloader()

        self.net = _MLP_fMRI(config).to(device, dtype=torch.float)
        self.criterion = nn.BCELoss().to(device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

class MLP_Wrapper(Wrapper):
    # SWEEP here for placeholders
    def __init__(self, config, SWEEP=False):
        self.config = config
        self.checkpoint_dir = "./checkpoint_dir/"
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.checkpoint_dir += "{}/".format(config['model_name'])
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.output_dir = self.checkpoint_dir + "output_dir/"
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        self.prepare_dataloader()

        self.net = _MLP_gene(config).to(device, dtype=torch.float)

        #self.criterion = nn.CrossEntropyLoss().to(device)
        #self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config["lr_rnn"])
        '''
        self.criterion = nn.BCELoss().to(device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, weight_decay=0.1)
        '''
        #self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, weight_decay=0.1)


        self.criterion = nn.BCELoss().to(device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, weight_decay=0.1)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, weight_decay=0.1)

    def prepare_dataloader(self):
        # only train data, for test, valid, wait until ADNI data are ready
        self.train_data = fmri_gene_data(self.config, 'train', 'gene')
        self.train_dataloader = DataLoader(self.train_data, batch_size=self.config['batch_size'], shuffle=False, drop_last=True)

        self.valid_data = fmri_gene_data(self.config, 'valid', 'gene')
        self.valid_dataloader = DataLoader(self.valid_data, batch_size=1, shuffle=False, drop_last=True)

        self.test_data = fmri_gene_data(self.config, 'test', 'gene')
        self.test_dataloader = DataLoader(self.test_data, batch_size=1, shuffle=False, drop_last=True)

    def train_model_epoch(self):
        self.net.train(True)
        loss_all = []
        for fmri, gene, label in self.train_dataloader:

            self.net.zero_grad()
            label = F.one_hot(label.to(device, dtype=torch.long), num_classes=2).float()
            gene = gene.view(5, 500)
            gene = gene.to(device, dtype=torch.float)
            output = self.net(gene)
            loss = self.criterion(output, label)
            loss_all += [loss.item()]
            loss.backward()
            self.optimizer.step()
        print('loss', np.mean(loss_all))
    
    def valid_model_epoch(self):
        #basically same as train, but on valid_dataloader
        self.net.eval() #so dr is off
        loss_all = []
        with torch.no_grad():
            for fmri, gene, label in self.valid_dataloader:
                label = F.one_hot(label.to(device, dtype=torch.long), num_classes=2).float()
                gene = gene.view(1, 500)
                gene = gene.to(device, dtype=torch.float)
                output = self.net(gene)
                loss = self.criterion(output, label)
                # print(loss)
                loss_all += [loss.item()]

        return np.mean(loss_all)

    def test(self, raw=False):
        self.net.eval()
        loss_all = []
        accu_all = []
        predictions = []
        true_labels = []
        with torch.no_grad():
            for fmri, gene, label in self.test_dataloader:
                label = F.one_hot(label.to(device, dtype=torch.long), num_classes=2).float()
                gene = gene.view(1, 500)
                gene = gene.to(device, dtype=torch.float)
                output = self.net(gene)
                loss = self.criterion(output, label)
                loss_all.append(loss.item())
                acc_test = accuracy_score(label.detach().cpu().numpy(), output.detach().cpu().numpy()>0.5)
                accu_all.append(acc_test)

                # Store raw predictions and labels
                predictions.append(output.detach().cpu().numpy())
                true_labels.append(label.detach().cpu().numpy())

        # Flatten the lists of predictions and true labels
        predictions = np.concatenate(predictions, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)

        if raw:
            return np.mean(loss_all), np.mean(accu_all), predictions, true_labels
        return np.mean(loss_all), np.mean(accu_all)

    def get_curve(self, config):
        model = _MLP_gene(config)
        model.load_state_dict(torch.load('/gpfs/gibbs/pi/gerstein/xz584/fmgene/RNN_2/RNN_2_d_2_0.6627748436049411.pth', map_location=device))
        model = model.to(device)
        model.eval()
        probs = []
        labels = []
        hidden = torch.zeros(2, 1, 64)
        hidden = hidden.to(torch.device('cuda:0' if (torch.cuda.is_available()) else "cpu"), dtype=torch.float)
        for fmri, gene, label in self.test_dataloader:
            #label = F.one_hot(label.to(device, dtype=torch.long), num_classes=2).float()
            gene = gene.view(1, 500)
            gene = gene.to(device, dtype=torch.float)
            pred = model(gene, hidden)
            pred = pred.squeeze(-1).detach().numpy()
            #pred = pred.detach().numpy()
            if label.numpy()[0] == True:
                labels.append(1)
            else:
                labels.append(0)
            probs.append(pred[0][1])
        
        
        print('labels', labels)
        auc = roc_auc_score(labels, probs)
        fpr, tpr, threshold = roc_curve(labels, probs)
        return fpr, tpr, threshold, auc
    
    def get_shap(self):
        self.model_shap = _MLP_gene(self.config)
        self.model_shap.load_state_dict(torch.load('/gpfs/gibbs/pi/gerstein/xz584/fmgene/checkpoint_dir/RNN_0/RNN_0_d_2_0.6888428586997734.pth', map_location=device))
        self.model_shap = self.model_shap.to(device)
        self.model_shap.eval()
        batch = next(iter(self.train_dataloader))
        train, _ = batch
        train = train.to(device, dtype=torch.float)
        
        for i in range(20):
            batch = next(iter(self.train_dataloader))
            curr, _ = batch
            curr = curr.to(device, dtype=torch.float)
            train = torch.cat((train, curr), 0)

        batch1 = next(iter(self.test_dataloader))
        test, _ = batch1
        test =  test.to(device, dtype=torch.float)
        
        #print(train)
        #print(test)
        '''
        train = []
        test = []
        count = 0
        print("TEST")
        for gene in self.test_dataloader:
            print(count)
            if count < 5:
                train.append(gene)
                count += 1
            elif count < 2:
                test.append(gene)
                count += 1
            else:
                break
        '''

        #print(self.test_dataloader)

        #train, test =  self.train_data.data_shap()

        explainer = shap.DeepExplainer(self.model_shap, train)
        #print(test.shape)
        shap_values = explainer.shap_values(test)
        '''
        explainer = shap.KernelExplainer(self.model_shap.predict, shap.sample(train, 25))
        shap_values = explainer.shap_values(test, nsamples=10)
        '''
        #shap.sample(train_data_x_np, 25) 
        print(shap_values)
        with open('features.pk', 'rb') as fi:
                #These are the SNPs that have been found to be the most significant
                features = pickle.load(fi)
        #shap.summary_plot(shap_values)
        print("pt2")
        shap.summary_plot(shap_values, features, show=False)
        # print("TEST")
        #shap.summary_plot(shap_values, features, show='False')
        plt.savefig('shap_gene.png')

    def random_forest(self):
        # Prepare the data
        X_train, y_train = dataloader_to_xy(self.train_dataloader)
        X_valid, y_valid = dataloader_to_xy(self.valid_dataloader)
        X_test, y_test = dataloader_to_xy(self.test_dataloader)

        # Parameters for Grid Search
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Initialize the Random Forest model
        rf = RandomForestClassifier(random_state=42)

        # Grid Search with Cross-Validation
        CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)

        # Train the model with Grid Search
        CV_rf.fit(X_train, y_train)

        # Best parameters
        print("Best parameters:", CV_rf.best_params_)

        # Use the best estimator
        best_rf = CV_rf.best_estimator_

        # Make predictions
        y_pred = best_rf.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        print(f"Test Accuracy: {accuracy}")
        print(f"Test ROC AUC: {roc_auc}")

class M_Wrapper(Wrapper):
    # SWEEP here for placeholders
    def __init__(self, config, SWEEP=False):
        super().__init__(config)

        self.net = _Merged(config).to(device, dtype=torch.float)
        self.criterion = nn.BCELoss().to(device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=config["lr_cnn"], weight_decay=config["weight_decay_cnn"])

    def prepare_dataloader(self):
        # only train data, for test, valid, wait until ADNI data are ready
        self.train_data = fmri_gene_data(self.config, 'train', 'all')
        self.train_dataloader = DataLoader(self.train_data, batch_size=self.config['batch_size'], shuffle=False, drop_last=True)

        self.valid_data = fmri_gene_data(self.config, 'valid', 'all')
        self.valid_dataloader = DataLoader(self.valid_data, batch_size=1, shuffle=False, drop_last=True)

        self.test_data = fmri_gene_data(self.config, 'test', 'all')
        self.test_dataloader = DataLoader(self.test_data, batch_size=1, shuffle=False, drop_last=True)

    def train(self, verbose=1):
        print('training...')

        self.optimal_valid_metric = np.inf
        self.optimal_epoch = -1

        for self.epoch in range(1, self.config['epochs']+1):
            self.train_model_epoch()
            val_loss = self.valid_model_epoch()
            self.save_checkpoint(val_loss)
            print('epoch {}: valid_loss ='.format(self.epoch), '%.3f' % (val_loss))
            if self.config['SWEEP']:
                pass
            
        print('best model saved at {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric.item())

    def train_model_epoch(self):
        self.net.train(True)
        # print(len(self.train_dataloader))
        # sys.exit()
        for fmri, gene, label in self.train_dataloader:
            # print(label.to(device, dtype=torch.float))
            fmri, gene, label = fmri.to(device, dtype=torch.float), gene.to(device, dtype=torch.float), F.one_hot(label.to(device, dtype=torch.long), num_classes=2).float()
            # fmri, gene, label = fmri.to(device, dtype=torch.float), gene.to(device, dtype=torch.float), F.one_hot(label.to(device, dtype=torch.float))
            #print(gene)
            self.net.zero_grad()
            # print(label)

            preds = self.net(fmri, gene)
            # print("PRED")
            # print(preds)
            loss = self.criterion(preds, label)
            # print(loss)
            loss.backward()
            self.optimizer.step()
    
    def valid_model_epoch(self):
        #basically same as train, but on valid_dataloader
        self.net.eval() #so dr is off
        loss_all = []
        with torch.no_grad():
            for fmri, gene, label in self.valid_dataloader:
                # print(label)
                fmri, gene, label = fmri.to(device, dtype=torch.float), gene.to(device, dtype=torch.float), F.one_hot(label.to(device, dtype=torch.long), num_classes=2).float()
                #fmri, gene, label = fmri.to(device, dtype=torch.float), F.one_hot(label.to(device, dtype=torch.long), num_classes=2).float()
                # fmri, gene, label = fmri.to(device, dtype=torch.float), gene.to(device, dtype=torch.float), F.one_hot(label, num_classes=2).to(device, dtype=torch.float)

                preds = self.net(fmri, gene)
                loss = self.criterion(preds, label)
                # print(loss)
                # print(preds)
                # print(label)
                loss_all += [loss.item()]
        return np.mean(loss_all)

        #basically same as train, but on valid_dataloader
        self.net.eval() #so dr is off
        loss_all = []
        with torch.no_grad():
            for fmri, gene, label in self.valid_dataloader:
                #fmri, label = fmri.to(device, dtype=torch.float), F.one_hot(label.to(device, dtype=torch.long), num_classes=2).float()
                fmri, gene, label = fmri.to(device, dtype=torch.float), gene.to(device, dtype=torch.float),  F.one_hot(label.to(device, dtype=torch.long), num_classes=2).float()

                preds = self.net(fmri, gene)
                loss = self.criterion(preds, label)
                # print(loss)
                # print(preds)
                # print(label)
                loss_all += [loss.item()]
        return np.mean(loss_all)

    def test(self, raw=False):
        self.net.eval() # so dropout is off
        loss_all = []
        accu_all = []
        predictions = []
        true_labels = []
        with torch.no_grad():
            for fmri, gene, label in self.test_dataloader:
                fmri, gene, label = fmri.to(device, dtype=torch.float), gene.to(device, dtype=torch.float), F.one_hot(label.to(device, dtype=torch.long), num_classes=2).float()

                preds = self.net(fmri, gene)
                loss = self.criterion(preds, label)
                loss_all.append(loss.item())
                acc_test = accuracy_score(label.detach().cpu().numpy(), preds.detach().cpu().numpy()>0.5)
                accu_all.append(acc_test)

                # Store raw predictions and labels
                predictions.append(preds.detach().cpu().numpy())
                true_labels.append(label.detach().cpu().numpy())

        # Flatten the lists of predictions and true labels
        predictions = np.concatenate(predictions, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)

        if raw:
            return np.mean(loss_all), np.mean(accu_all), predictions, true_labels
        return np.mean(loss_all), np.mean(accu_all)

    def save_checkpoint(self, loss):
        score = loss
        if score <= self.optimal_valid_metric:
            self.optimal_epoch = self.epoch
            self.optimal_valid_metric = score
            #save model

            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith(".pth"):
                        try:
                            os.remove(self.checkpoint_dir + File)
                        except:
                            pass
            torch.save(
                self.net.state_dict(),
                "{}{}_{}.pth".format(self.checkpoint_dir, self.config['model_name'], self.optimal_epoch)
            )
    
    def get_curve(self, config):
        model = _Merged(config)
        model.load_state_dict(torch.load('/gpfs/gibbs/pi/gerstein/xz584/fmgene/checkpoint_dir/merged_2/merged_2_d_3.pth', map_location=device))
        model = model.to(device)
        model.eval()
        probs = []
        labels = []
        for fmri, gene, label in self.test_dataloader:
            #label = F.one_hot(label.to(device, dtype=torch.long), num_classes=2).float()
            fmri = fmri.to(device, dtype=torch.float)
            gene = gene.to(device, dtype=torch.float)
            pred = model(fmri, gene)
            pred = pred.squeeze(-1).detach().numpy()
            if label.numpy()[0] == True:
                labels.append(1)
            else:
                labels.append(0)
            probs.append(pred[0][1])
        auc = roc_auc_score(labels, probs)
        fpr, tpr, threshold = roc_curve(labels, probs)
        return fpr, tpr, threshold, auc

class CNN_paper(Wrapper):
    def __init__(self, config, SWEEP=False):
        self.config = config
        self.checkpoint_dir = "./checkpoint_dir/"
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.checkpoint_dir += "{}/".format(config['model_name'])
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.output_dir = self.checkpoint_dir + "output_dir/"
        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        torch.manual_seed(config['seed'])
        self.prepare_dataloader()

        # Initialize the model
        self.net = BinaryClassifier3D(config).to(device, dtype=torch.float)
        self.criterion = nn.BCEWithLogitsLoss().to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config["lr_cnn"], weight_decay=config["weight_decay_cnn"])


if __name__ == '__main__':

    sys.exit()