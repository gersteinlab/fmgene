# dataloader for fmgene project
# status: in-developing
import random
import os, sys
import dicom2nifti
import torch
import glob
import pickle

import numpy as np
import pydicom as dicom
import matplotlib.pylab as plt
import nibabel as nb
import pandas as pd
import torch.nn as nn

from nilearn.image import resample_img
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2
from torch.utils.data import Dataset, DataLoader

#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
#from torch.utils.data import Dataset

class fmri_gene_data_v1(Dataset):
    def __init__(self):
        pass

# class fmri_gene_data_v1(Dataset):
class fmri_gene_data(Dataset):
    def __init__(self, config, stage, mode):
        # mode controls the data - i.e. fmri = 60, all = 54(cross with gene and fmri)
        random.seed(config['seed'])
        # random.seed(0)
        self.config = config
        self.df_info = pd.read_csv("/gpfs/gibbs/pi/gerstein/sk2776/data/images/ADNI_2_fMRI_label_3_24_2023.csv")
        self.stage = stage
        self.mode = mode
        # train:valid:test set = 3:1:1

        with open('common.pk', 'rb') as fi:
            #This holds the fmri filenames that have people with gene data
            self.common = pickle.load(fi)

        #list of file locations for first set of MRI images
        self.dir = ''
        if config['type'] == 'all':
            self.dir = '/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/all'
            self.common = self.common[0]
        elif config['type'] == 'Resting':
            self.dir = '/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/Resting_State_fMRI'
            self.common = self.common[1]
        elif config['type'] == 'MoCo':
            self.dir = '/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/MoCoSeries'
            self.common = self.common[2]
        
        self.fmri_list = os.listdir(self.dir)
        self.gene_label = pd.read_csv("/gpfs/gibbs/pi/gerstein/sk2776/data/DXSUM_PDXCONV_ADNIALL.csv")
        self.gene_label = self.gene_label[['PTID', 'DXCURREN', 'DXCHANGE']]
        self.gene_dir = "/gpfs/gibbs/pi/gerstein/sk2776/data/genetic"
        self.gene_list= []
        for name in os.listdir(self.gene_dir):
            if self.get_gene_label(name) != -1:
                self.gene_list.append(name)
        
        if mode == 'all' or mode == 'gene':
            with open('features.pk', 'rb') as fi:
                #These are the SNPs that have been found to be the most significant
                self.features = pickle.load(fi)
        
        random.shuffle(self.fmri_list)
        random.shuffle(self.gene_list)
        random.shuffle(self.common)

        split1 = len(self.fmri_list)*3//5
        split2 = len(self.fmri_list)*4//5

        split1_gene = len(self.gene_list)*3//5
        split2_gene = len(self.gene_list)*4//5
        split1_common = len(self.common)*3//5
        split2_common = len(self.common)*4//5
        # print('v1', len(self.common))
        # print(split1_common)
        # print(split2_common)
        # sys.exit()

        if stage == 'train':
            if mode == 'fmri':
                self.index_list = self.fmri_list[:split1]
            elif mode == 'gene':
                self.index_list = self.gene_list[:split1_gene]
            else:
                self.index_list = self.common[:split1_common]
        elif stage == 'valid':
            if mode == 'fmri':
                self.index_list = self.fmri_list[split1:split2]
            elif mode == 'gene':
                self.index_list = self.gene_list[split1_gene:split2_gene]
            else:
                self.index_list = self.common[split1_common:split2_common]
        elif stage == 'test':
            if mode == 'fmri':
                self.index_list = self.fmri_list[split2:]
                labels = []
                for item in self.index_list:
                    labels.append(self.get_fmri_label(item))
            elif mode == 'gene':
                self.index_list = self.gene_list[split2_gene: ]
            else:
                self.index_list = self.common[split2_common: ]
    
        # if self.stage == 'train':
        #     print(self.index_list[:20], len(self.index_list))
    
    def __len__(self):
        return len(self.index_list)
    
    def data_shap(self):
        split1_gene = len(self.gene_list)*3//5
        split2_gene = len(self.gene_list)*4//5
        train = []
        test = []

        for i in range(split1_gene, split2_gene):
            gene, label = self.gene_process(self.gene_list[i])
            train.append(gene)
        
        for i in range(split2_gene, len(self.gene_list)):
            gene, label = self.gene_process(self.gene_list[i])
            test.append(gene)
        
        return np.asarray(train), np.asarray(test)
        
    def get_gene_label(self, gene_id):
        part_id = gene_id.split(".")[0]
        patient_info1 = self.gene_label[self.gene_label['PTID'] == part_id]
        patient_info = patient_info1[["DXCURREN"]]
        patient_info = patient_info.dropna()
        if patient_info.empty:
            patient_info = patient_info1[['DXCHANGE']]
            patient_info = patient_info.dropna()
            if patient_info.empty:
                label_num = int(patient_info.iloc[-1]["DXCHANGE"])
                return label_num
            else:
                return -1
        else:
            label_num = int(patient_info.iloc[-1]["DXCURREN"])
            return label_num
                
    def fmri_process(self, id):
        item_name = os.listdir(self.dir + '/' + id)[0]
        path_to_file = self.dir + '/' + id + '/' + item_name
        item = nb.load(path_to_file)
        image_num = id.split("_")[-1]
        label = self.df_info[self.df_info["Image Data ID"] == image_num]['Group'].values[0]
        fmri = item.get_fdata()
        if len(fmri.shape) == 3:
            fmri = np.expand_dims(fmri, axis=0)
        else:
            fmri = torch.tensor(fmri).permute((3, 0, 1, 2))
        return fmri, str(label)
    
    def get_fmri_label(self, id):
        image_num = id.split("_")[-1]
        return self.df_info[self.df_info["Image Data ID"] == image_num]['Group'].values[0]

    def gene_process(self, gene_id):
        part_id = gene_id.split(".")[0]
        label_num = self.get_gene_label(part_id)
        patient_info = self.gene_label[self.gene_label['PTID'] == part_id]
        patient_info = patient_info[["DXCURREN"]]
        patient_info = patient_info.dropna()
        if patient_info.empty:
            label_num = 3
        else:
            label_num = int(patient_info.iloc[-1]["DXCURREN"])
    
        if label_num == 1 :
        #or label_num == 7 or label_num == 9:
            label = "NC"
        elif label_num == 2 :
        #or label_num == 4 or label_num == 8:
            label = "MCI"
        elif label_num == 3:
        #or label_num == 5 or label_num == 6:
            label = "AD"
        else:
            label = "AD"

        gene_file = pd.read_csv("/gpfs/gibbs/pi/gerstein/sk2776/data/genetic/" + gene_id)
        gene_file = gene_file.dropna()
        gene_file['B Allele Freq'] = gene_file['B Allele Freq'].apply(lambda x: int(x * 2))
        gene_file = gene_file.pivot_table('B Allele Freq', ['Sample ID'], 'SNP Name')
        #print(gene_file.head)
        gene = []
        #features = self.features.append('Theta')
        for feature in self.features:
            if feature in gene_file.columns:
                gene.append(gene_file[feature].values[0])
            else:
                gene.append(0)
        #print(gene)

        #gene_file = gene_file[self.features[0]]
        #gene = gene_file.to_numpy()
        #gene = np.concatenate(gene, axis=0)
        return gene, label

    def __getitem__(self, idx):
        #this mode is to find the fmri data
        if self.mode == "fmri":
            id = self.index_list[idx]
            fmri, label = self.fmri_process(id)
            gene = 0
            return fmri, gene, label=='AD'

        #this is only to load the gene data
        if self.mode == "gene":

                #This tries to get the participant id
                gene_id = self.gene_list[idx]
                gene, label = self.gene_process(gene_id)
                return np.asarray(gene), label=='AD'
                
                '''
                #This finds the disease stage
                patient_info = self.gene_label[self.gene_label['PTID'] == part_id]
                patient_info = patient_info[["DXCURREN"]]
                patient_info = patient_info.dropna()
                label_num = int(patient_info.iloc[-1]["DXCURREN"])
                if label_num == 1:
                    label = "CN"
                elif label_num == 2:
                    label = "MCI"
                else:
                    label = "AD"

                #This finds the selects the participants gene data
                gene_file = pd.read_csv("/gpfs/gibbs/pi/gerstein/sk2776/data/genetic/" + gene_id)
                features = self.features.append('Theta')
                gene_file = gene_file[[self.features]]
               
                #The Log R ratio is a ratio to the value of the allele to the expect; filtering here for only ones that are significant to reduce size
                #self.gene_file['Log R Ratio'] = self.gene_file['Log R Ratio'].abs()
                #subset1 = self.gene_file[self.gene_file['Log R Ratio'] > 5]
                #subset = subset1[["Theta"]]

                gene = gene_file.to_numpy()
                gene = np.concatenate(gene, axis=0)

                #here clipping it so that we only have 500 values
                if len(gene) > 500:
                    gene = gene[0:500]
                    #print(len(gene))
                if len(gene) < 500:
                    pad_width_left = int(500 - len(gene))
                    gene = np.pad(gene, (0, pad_width_left), 'constant')
                
                fmri = 0
                '''
             
        #Here we find the gene file for the person with fMRI based on the first if statement
        if self.mode == "all":
            #print(len(self.common))
            #print("T2")
            fmri_id = self.common[idx][0:10]
            for name in os.listdir(self.gene_dir):
                part_id = name.split(".")[0]
                if part_id == fmri_id:
                    gene, _ = self.gene_process(name)
                    fmri, label = self.fmri_process(self.common[idx])
                    return fmri, np.asarray(gene), label=='AD'
                    '''
                    gene_file = pd.read_csv("/gpfs/gibbs/pi/gerstein/sk2776/data/genetic/" + name)
                    features = self.features.append('Theta')
                    gene_file = gene_file[[self.features]]
                    '''
                    '''
                    self.gene_file['Log R Ratio'] = self.gene_file['Log R Ratio'].abs()
                    subset1 = self.gene_file[self.gene_file['Log R Ratio'] > 5]
                    subset = subset1[["Theta"]]
                    '''
                    '''
                    gene = subset.to_numpy()
                    gene = np.concatenate(gene, axis=0)
                    gene = gene_file.to_numpy()
                    gene = np.concatenate(gene, axis=0)
                
                    if len(gene) > 500:
                        gene = gene[0:500]
                    if len(gene) < 500:
                        pad_width_left = int(500 - len(gene))
                        gene = np.pad(gene, (0, pad_width_left), 'constant')
                        print(len(gene))
                    '''           
        '''
        fmri_id = id[0:10]
        gene = [0] * 8000
        for name in os.listdir(self.gene_dir):
        #Ex1: name = 002_S_0295.csv
            part_id = name.split(".")[0]
            #print(part_id)
           # print(fmri_id)
            index = 0
            #this means that we have found the corresponding file
            if part_id == fmri_id:
                self.gene_file = pd.read_csv("/gpfs/gibbs/pi/gerstein/sk2776/data/genetic/" + name)
            #self.gene_file = pd.read_csv("/gpfs/gibbs/pi/gerstein/sk2776/data/genetic/002_S_0295.csv")
                subset = self.gene_file[["SNP Name", "Theta"]]
                size = subset.shape[0]
                for i in range(size):
                    if abs(self.gene_file.iloc[i]['Log R Ratio']) > 5:
                        gene[index] = float(self.gene_file.iloc[i]['Theta'])
                        #print(gene[index])
                        index += 1
                break
        '''

class fmri_gene_data_v2(fmri_gene_data_v1):
    def __init__(self, config, stage, mode):
        random.seed(config['seed'])
        # random.seed(0)
        self.config = config
        self.df_info = pd.read_csv("/gpfs/gibbs/pi/gerstein/sk2776/data/images/ADNI_2_fMRI_label_3_24_2023.csv")
        self.stage = stage
        self.mode = mode
        # train:valid:test set = 3:1:1

        with open('common.pk', 'rb') as fi:
            #This holds the fmri filenames that have people with gene data
            self.common = pickle.load(fi)

        #list of file locations for first set of MRI images
        self.dir = '/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/Resting_State_fMRI'
        self.common = self.common[1]
        
        self.fmri_list = os.listdir(self.dir)
        self.gene_label = pd.read_csv("/gpfs/gibbs/pi/gerstein/sk2776/data/DXSUM_PDXCONV_ADNIALL.csv")
        self.gene_label = self.gene_label[['PTID', 'DXCURREN', 'DXCHANGE']]
        self.gene_dir = "/gpfs/gibbs/pi/gerstein/sk2776/data/genetic"
        self.gene_list= []
        for name in os.listdir(self.gene_dir):
            if self.get_gene_label(name) != -1:
                self.gene_list.append(name)
        
        with open('features.pk', 'rb') as fi:
            #These are the SNPs that have been found to be the most significant
            self.features = pickle.load(fi)
        
        random.shuffle(self.fmri_list)
        random.shuffle(self.gene_list)
        random.shuffle(self.common)

        split1_common = len(self.common)*3//5
        split2_common = len(self.common)*4//5
        # print('v2', len(self.common))
        # print(split1_common)
        # print(split2_common)
        # sys.exit()

        if stage == 'train':
            self.index_list = self.common[:split1_common]
        elif stage == 'valid':
            self.index_list = self.common[split1_common:split2_common]
        elif stage == 'test':
            self.index_list = self.common[split2_common:]
    
    def __len__(self):
        return len(self.index_list)
    
    def data_shap(self):
        split1_gene = len(self.gene_list)*3//5
        split2_gene = len(self.gene_list)*4//5
        train = []
        test = []

        for i in range(split1_gene, split2_gene):
            gene, label = self.gene_process(self.gene_list[i])
            train.append(gene)
        
        for i in range(split2_gene, len(self.gene_list)):
            gene, label = self.gene_process(self.gene_list[i])
            test.append(gene)
        
        return np.asarray(train), np.asarray(test)
        
    def get_gene_label(self, gene_id):
        part_id = gene_id.split(".")[0]
        patient_info1 = self.gene_label[self.gene_label['PTID'] == part_id]
        patient_info = patient_info1[["DXCURREN"]]
        patient_info = patient_info.dropna()
        if patient_info.empty:
            patient_info = patient_info1[['DXCHANGE']]
            patient_info = patient_info.dropna()
            if patient_info.empty:
                label_num = int(patient_info.iloc[-1]["DXCHANGE"])
                return label_num
            else:
                return -1
        else:
            label_num = int(patient_info.iloc[-1]["DXCURREN"])
            return label_num
                
    def fmri_process(self, id):
        item_name = os.listdir(self.dir + '/' + id)[0]
        path_to_file = self.dir + '/' + id + '/' + item_name
        item = nb.load(path_to_file)
        image_num = id.split("_")[-1]
        label = self.df_info[self.df_info["Image Data ID"] == image_num]['Group'].values[0]
        fmri = item.get_fdata()
        if len(fmri.shape) == 3:
            fmri = np.expand_dims(fmri, axis=0)
        else:
            fmri = torch.tensor(fmri).permute((3, 0, 1, 2))
        return fmri, str(label)
    
    def get_fmri_label(self, id):
        image_num = id.split("_")[-1]
        return self.df_info[self.df_info["Image Data ID"] == image_num]['Group'].values[0]

    def gene_process(self, gene_id):
        part_id = gene_id.split(".")[0]
        label_num = self.get_gene_label(part_id)
        patient_info = self.gene_label[self.gene_label['PTID'] == part_id]
        patient_info = patient_info[["DXCURREN"]]
        patient_info = patient_info.dropna()
        if patient_info.empty:
            label_num = 3
        else:
            label_num = int(patient_info.iloc[-1]["DXCURREN"])
    
        if label_num == 1 :
        #or label_num == 7 or label_num == 9:
            label = "NC"
        elif label_num == 2 :
        #or label_num == 4 or label_num == 8:
            label = "MCI"
        elif label_num == 3:
        #or label_num == 5 or label_num == 6:
            label = "AD"
        else:
            label = "AD"

        gene_file = pd.read_csv("/gpfs/gibbs/pi/gerstein/sk2776/data/genetic/" + gene_id)
        gene_file = gene_file.dropna()
        gene_file['B Allele Freq'] = gene_file['B Allele Freq'].apply(lambda x: int(x * 2))
        gene_file = gene_file.pivot_table('B Allele Freq', ['Sample ID'], 'SNP Name')
        #print(gene_file.head)
        gene = []
        #features = self.features.append('Theta')
        for feature in self.features:
            if feature in gene_file.columns:
                gene.append(gene_file[feature].values[0])
            else:
                gene.append(0)
        #print(gene)
        return gene, label

    def __getitem__(self, idx):
        fmri_id = self.index_list[idx][0:10]
        for name in os.listdir(self.gene_dir):
            part_id = name.split(".")[0]
            if part_id == fmri_id:
                gene, _ = self.gene_process(name)
                fmri, label = self.fmri_process(self.common[idx])
                return fmri, np.asarray(gene), label=='AD'


if __name__ == '__main__':

    print(len(os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/all')))
    print(len(os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/Resting_State_fMRI')))
    print(len(os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/MoCoSeries')))

    sys.exit()

    with open('common.pk', 'rb') as fi:
        #This holds the fmri filenames that have people with gene data
        common = pickle.load(fi)
    common = common[0]
    SMC = 0
    EMCI = 0
    LMCI = 0
    CN = 0
    male_SMC = 0
    female_SMC = 0
    female_EMCI = 0
    male_EMCI = 0
    female_LMCI = 0
    male_LMCI = 0
    female_CN = 0
    male_CN = 0
    CN_age = 0
    LMCI_age = 0
    EMCI_age = 0
    SMC_age = 0 
    df_info = pd.read_csv("/gpfs/gibbs/pi/gerstein/sk2776/data/images/ADNI_2_fMRI_label_3_24_2023.csv")
    unique = []
    for item in common:
        fmri_id = item[0:10]
        #print(fmri_id)
        label = df_info[df_info["Subject"] == fmri_id]['Group'].values[0]
        #print(label)
        age = int(df_info[df_info["Subject"] == fmri_id]['Age'].values[0])
        gender = df_info[df_info["Subject"] == fmri_id]['Sex'].values[0]
        if fmri_id not in unique:
            unique.append(fmri_id)
            if label == "AD":
                SMC += 1
                SMC_age += age
                if gender == 'F':
                    female_SMC += 1
                else:
                    male_SMC += 1
            elif label == "EMCI":
                EMCI += 1
                EMCI_age += age
                if gender == 'F':
                    female_EMCI += 1
                else:
                    male_EMCI += 1
            elif label == "LMCI":
                LMCI += 1
                LMCI_age += age
                if gender == 'M':
                    male_LMCI += 1
                else:
                    female_LMCI += 1
            else:
                CN += 1
                CN_age += age
                if gender == 'M':
                    male_CN += 1
                else:
                    female_CN += 1
    print("Male SMC")
    print(male_SMC)
    print("FEMALE SMC")
    print(female_SMC)
    print("SMC")
    print(SMC)
    print("Female EMCI")
    print(female_EMCI)
    print("Male EMCI")
    print(male_EMCI)
    print("EMC")
    print(EMCI)
    print("CN")
    print(CN)
    print("FEMALE CN")
    print(female_CN)
    print("Male CN")
    print(male_CN)
    print("LMCI")
    print(LMCI)
    print("FEMALE_LMCI")
    print(female_LMCI)
    print("MALE LMCI")
    print(male_LMCI)
    print("CN AGE")
    print(CN_age)
    print("LMCI Age")
    print(LMCI_age)
    print("EMCI Age")
    print(EMCI_age)
    print("SMC_age")
    print(SMC_age)
    print(len(unique))
            

    #GENETIC open files with labels and with demographic info
    '''
    gene_label = pd.read_csv("/gpfs/gibbs/pi/gerstein/sk2776/data/DXSUM_PDXCONV_ADNIALL.csv")
    gene_label = gene_label[['PTID', 'DXCURREN', 'DXCHANGE']]
    info = pd.read_csv("/gpfs/gibbs/pi/gerstein/sk2776/data/ADNIMERGE.csv")
    NC_age = 0
    NC_female = 0
    NC_male = 0
    NC = 0
    MCI = 0
    MCI_male = 0
    MCI_female = 0
    MCI_age = 0
    AD = 0
    AD_female = 0
    AD_male = 0
    AD_age = 0
     #Go through ever item in gene directory
    for item in os.listdir("/gpfs/gibbs/pi/gerstein/sk2776/data/genetic"):
        part_id = item.split(".")[0]
        #Get age and gender
        age = info[info["PTID"] == part_id]['AGE']
        if len(age) > 0:
            age = age.values[0]
        else:
            break
        gender = info[info["PTID"] == part_id]['PTGENDER']
        if len(gender) > 0:
            gender = gender.values[0]
        else:
            break
        #[0]
        patient_info1 = gene_label[gene_label['PTID'] == part_id]
        patient_info = patient_info1[["DXCURREN"]]
        patient_info = patient_info.dropna()
        if patient_info.empty:
            patient_info = patient_info1[['DXCHANGE']]
            patient_info = patient_info.dropna()
            if patient_info.empty:
                label_num = 3
            else:
                label_num = int(patient_info.iloc[-1]["DXCHANGE"])
        else:
            label_num = int(patient_info.iloc[-1]["DXCURREN"])
        #print(patient_info)
        #print(label_num)
        if label_num == 1 or label_num == 7 or label_num == 9:
            label = "NC"
            NC_age += age
            if gender == 'Male':
                NC_male += 1
            else:
                NC_female += 1
            NC += 1
        elif label_num == 2 or label_num == 4 or label_num == 8:
            label = "MCI"
            MCI_age += age
            if gender == 'Male':
                MCI_male += 1
            else:
                MCI_female += 1
            MCI += 1
        #label_num == 3 or label_num == 5 or label_num == 6:
        else: 
            label = "AD"
            AD_age += age
            if gender == 'Male':
                AD_male += 1
            else:
                AD_female += 1
            AD += 1
    print("NC")
    print(NC)
    print("Age")
    print(NC_age)
    print("Female")
    print(NC_female)
    print("MALE")
    print("MCI")
    print(MCI)
    print("Age")
    print(MCI_age)
    print("Female")
    print(MCI_female)
    print("MALE")
    print(MCI_male)
    print("AD")
    print(AD)
    print("Age")
    print(AD_age)
    print("Female")
    print(AD_female)
    print("MALE")
    print(AD_male)
    '''
            

    #Code to get Demographic Information for fMRI data
    '''
    unique = []
    SMC = 0
    EMCI = 0
    LMCI = 0
    CN = 0
    male_SMC = 0
    female_SMC = 0
    female_EMCI = 0
    male_EMCI = 0
    female_LMCI = 0
    male_LMCI = 0
    female_CN = 0
    male_CN = 0
    CN_age = 0
    LMCI_age = 0
    EMCI_age = 0
    SMC_age = 0 

    df_info = pd.read_csv("/gpfs/gibbs/pi/gerstein/sk2776/data/images/ADNI_2_fMRI_label_3_24_2023.csv")
    print(len(os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/all')))
    print(len(os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/Resting_State_fMRI')))
    print(len(os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/MoCoSeries')))
    for item in os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/all'):
        fmri_id = item[0:10]
        label = df_info[df_info["Subject"] == fmri_id]['Group'].values[0]
        age = int(df_info[df_info["Subject"] == fmri_id]['Age'].values[0])
        gender = df_info[df_info["Subject"] == fmri_id]['Sex'].values[0]
        if fmri_id not in unique:
            unique.append(fmri_id)
            if label == "AD":
                SMC += 1
                SMC_age += age
                if gender == 'F':
                    female_SMC += 1
                else:
                    male_SMC += 1
            elif label == "EMCI":
                EMCI += 1
                EMCI_age += age
                if gender == 'F':
                    female_EMCI += 1
                else:
                    male_EMCI += 1
            elif label == "LMCI":
                LMCI += 1
                LMCI_age += age
                if gender == 'M':
                    male_LMCI += 1
                else:
                    female_LMCI += 1
            else:
                CN += 1
                CN_age += age
                if gender == 'M':
                    male_CN += 1
                else:
                    female_CN += 1
    for item in os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/Resting_State_fMRI'):
        fmri_id = item[0:10]
        label = df_info[df_info["Subject"] == fmri_id]['Group'].values[0]
        #print(label)
        age = int(df_info[df_info["Subject"] == fmri_id]['Age'].values[0])
        gender = df_info[df_info["Subject"] == fmri_id]['Sex'].values[0]
        if fmri_id not in unique:
            print(fmri_id)
            unique.append(fmri_id)
            if label == "AD":
                print("HERE")
                SMC += 1
                SMC_age += age
                if gender == 'F':
                    female_SMC += 1
                else:
                    male_SMC += 1
            elif label == "EMCI":
                EMCI += 1
                EMCI_age += age
                if gender == 'F':
                    female_EMCI += 1
                else:
                    male_EMCI += 1
            elif label == "LMCI":
                LMCI += 1
                LMCI_age += age
                if gender == 'M':
                    male_LMCI += 1
                else:
                    female_LMCI += 1
            else:
                CN += 1
                CN_age += age
                if gender == 'M':
                    male_CN += 1
                else:
                    female_CN += 1
        for item in os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/MoCoSeries'):
            fmri_id = item[0:10]
            label = df_info[df_info["Subject"] == fmri_id]['Group'].values[0]
            age = int(df_info[df_info["Subject"] == fmri_id]['Age'].values[0])
            gender = df_info[df_info["Subject"] == fmri_id]['Sex'].values[0]
            if fmri_id not in unique:
                unique.append(fmri_id)
                if label == "AD":
                    SMC += 1
                    SMC_age += age
                    if gender == 'F':
                        female_SMC += 1
                    else:
                        male_SMC += 1
                elif label == "EMCI":
                    EMCI += 1
                    EMCI_age += age
                    if gender == 'F':
                        female_EMCI += 1
                    else:
                        male_EMCI += 1
                elif label == "LMCI":
                    LMCI += 1
                    LMCI_age += age
                    if gender == 'M':
                        male_LMCI += 1
                    else:
                        female_LMCI += 1
                else:
                    CN += 1
                    CN_age += age
                    if gender == 'M':
                        male_CN += 1
                    else:
                        female_CN += 1
    print("Male SMC")
    print(male_SMC)
    print("FEMALE SMC")
    print(female_SMC)
    print("SMC")
    print(SMC)
    print("Female EMCI")
    print(female_EMCI)
    print("Male EMCI")
    print(male_EMCI)
    print("EMC")
    print(EMCI)
    print("CN")
    print(CN)
    print("FEMALE CN")
    print(female_CN)
    print("Male CN")
    print(male_CN)
    print("LMCI")
    print(LMCI)
    print("FEMALE_LMCI")
    print(female_LMCI)
    print("MALE LMCI")
    print(male_LMCI)
    print("CN AGE")
    print(CN_age)
    print("LMCI Age")
    print(LMCI_age)
    print("EMCI Age")
    print(EMCI_age)
    print("SMC_age")
    print(SMC_age)
    print(len(unique))
    '''
            
    '''
    self.dir = '/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/Resting_State_fMRI'
        self.common = self.common[1]
        elif config['type'] == 'MoCo':
            self.dir = '/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/MoCoSeries'
    
    
    gene_dir = "/gpfs/gibbs/pi/gerstein/sk2776/data/genetic"
    gene_list= os.listdir(gene_dir)
    gene_label = pd.read_csv("/gpfs/gibbs/pi/gerstein/sk2776/data/DXSUM_PDXCONV_ADNIALL.csv")
    gene_label = gene_label[['PTID', 'DXCURREN']]
    all_gene_data = pd.DataFrame()
    files = random.choices(os.listdir(gene_dir), k=10)
    full = []
    for file in files:
        full.append(os.path.join('/gpfs/gibbs/pi/gerstein/sk2776/data/genetic/', file))
    #Step 1 combine all csvs into one 
    for name in full:
        print(name)
        gene_file = pd.read_csv(name)
        gene_file = gene_file.dropna()
        gene_file = gene_file.loc[int(gene_file['Chr']) == 19]
        gene_file['B Allele Freq'] = gene_file['B Allele Freq'].apply(lambda x: int(x * 2))
        gene_file = gene_file.pivot_table('B Allele Freq', ['Sample ID'], 'SNP Name')
        if all_gene_data.shape[0] == 0:
            all_gene_data = gene_file
        else:
            all_gene_data =  pd.concat([all_gene_data, gene_file])
        #Step 2removes ones where there is are no shared SNps
        all_gene_data = all_gene_data.dropna(axis='columns')
        #print(all_gene_data.shape)xs

    #Step 3 merge with labels
    gene_label = gene_label.dropna()
    gene_label = gene_label.groupby('PTID').apply(lambda x: x.iloc[ -1]).reset_index(drop=True)
    
    gene_label = gene_label[['PTID', 'DXCURREN']]
    
    #print(gene_label)
    gene_label['DXCURREN'].astype('Int64')
    print(gene_label.head)


    all_gene_data = pd.merge(all_gene_data, gene_label, left_on='Sample ID', right_on='PTID', how='left')

    #Step 4 drops rows where label is missing 
    all_gene_data = all_gene_data.dropna()
    #print(all_gene_data.head)

    #This is just separating into features and classes
    cols= len(all_gene_data.columns)
    X = all_gene_data.iloc[:, 0: (cols - 3)].values
    Y = all_gene_data.iloc[:, cols - 1].values
    #print(X)
    #print(Y)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    test = SelectKBest(chi2, k=500)
    X_kbest = test.fit_transform(X, Y)
    #fit = test.fit(X, Y)
    cols_idxs = test.get_support(indices=True)
    features= all_gene_data.iloc[:,cols_idxs]
    
    features = list(features.columns)
    print(features)
    filename = 'features.pk'
    with open(filename, 'wb') as fi :
        pickle.dump(features, fi)
    print(features)
    sys.exit(0)


    #Code to find overlap between fMRI and other samples
    '''
    '''
    count = 0
    common = [[], [], []]
    print(len(os.listdir("/gpfs/gibbs/pi/gerstein/sk2776/data/genetic")))
    for file in os.listdir("/gpfs/gibbs/pi/gerstein/sk2776/data/genetic"):
        part_id = file.split(".")[0]
        for name in os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/all'):
            fmri_id = name[0:10]
            if fmri_id == part_id:
                count += 1
                common[0].append(name)
        for name in os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/Resting_State_fMRI'):
            fmri_id = name[0:10]
            if fmri_id == part_id:
                count += 1
                common[1].append(name)
        for name in os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/MoCoSeries'):
            fmri_id = name[0:10]
            if fmri_id == part_id:
                count += 1
                common[2].append(name)
        filename = 'common.pk'
        print(common)
        
        with open(filename, 'wb') as fi :
            pickle.dump(common, fi)

    '''
    '''
    gene = []
    max = 0
    for name in os.listdir("/gpfs/gibbs/pi/gerstein/sk2776/data/genetic"):
        gene_file = pd.read_csv("/gpfs/gibbs/pi/gerstein/sk2776/data/genetic/" + name)
        subset = gene_file[["SNP Name", "Theta"]]
        size = subset.shape[0]
        for i in range(size):
            if abs(gene_file.iloc[i]['Log R Ratio']) > 6:
                gene.append(gene_file.iloc[i]['Theta'])
        print(len(gene))
        if len(gene) > max:
            max = len(gene)
    print(max)
    '''
    '''
    gene_dir = "/gpfs/gibbs/pi/gerstein/sk2776/data/genetic"
    final = pd.DataFrame()
    for file in os.listdir(gene_dir):
        gene_file = pd.read_csv("/gpfs/gibbs/pi/gerstein/sk2776/data/genetic/" + file)
        pivot_df = gene_file.pivot_table(index = "Sample ID", columns = ['SNP Name', , , values = ['Theta', 'Log R Ratio'])
        final = pd.concat([pivot_df, final], axis=0)
        print(final.head())
    final.to_csv('/gpfs/gibbs/pi/gerstein/sk2776/data/genetic/all.csv')  
    '''


    #self.train_data = fmri_gene_data(self.config)
    #self.train_dataloader = DataLoader(self.train_data, batch_size=self.config['batch_size'], shuffle=False, drop_last=True)
    #gets the folder name of every individual
    
    #010_S_4345 - only 1 image
    #018_S_4399 - 2nd image only has around 500 slices 
    #012_S_4188 - the 2nd 2 images only have 
    #012_S_4643 - only 1 slice

    #dir = '/gpfs/gibbs/pi/gerstein/sk2776/data/images/ADNI/002_S_0295/Resting_State_fMRI/2011-06-02_07_56_36.0/I238623'
    #dir_to_save = '/gpfs/gibbs/pi/gerstein/sk2776/data/images/ADNI'
    #dicom2nifti.convert_directory(dir, dir_to_save)


    '''
    dir = '/gpfs/gibbs/pi/gerstein/sk2776/data/images/ADNI/'
    dir1 = '/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/'
    #os.mkdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/ADNI/final/all')
    #os.mkdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/ADNI/final/MoCoSeries')
    for individual in os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/ADNI'):
        #print("Here")
        person_dir = individual
        for typeImage in os.listdir(dir + individual):
            person_dir +=  '/' + typeImage 
            for date in os.listdir(dir + individual + '/' + typeImage):
                #print(dir + person_dir)
                person_dir += '/' + date
                for imageNum in os.listdir(dir + individual +'/' + typeImage + '/' + date):
                    person_dir =  individual +'/' + typeImage + '/' + date + '/' + imageNum
                    person_save = individual + '_' + typeImage + '_' + date + '_' + imageNum
                    new_dir_path = ''
                    if typeImage == 'MoCoSeries':
                        new_dir_path = dir1 + 'MoCoSeries/' + person_save
                    elif typeImage == 'Resting_State_fMRI':
                        new_dir_path = dir1 + 'Resting_State_fMRI/' + person_save
                    else:
                        new_dir_path = dir1 + 'all/' + person_save
                    if typeImage == 'Resting_State_fMRI':
                        print(len(os.listdir(dir + person_dir)))
                        if len(os.listdir(dir + person_dir)) == 6720:
                            os.mkdir(new_dir_path)
                            dicom2nifti.convert_directory(dir + person_dir, new_dir_path)
                    elif typeImage == 'MoCoSeries':
                        print(len(os.listdir(dir + person_dir)))
                        if len(os.listdir(dir + person_dir)) == 105:
                            os.mkdir(new_dir_path)
                            dicom2nifti.convert_directory(dir + person_dir, new_dir_path)
                    else:
                        if len(os.listdir(dir + person_dir)) == 1:
                            os.mkdir(new_dir_path)
                            dicom2nifti.convert_directory(dir + person_dir, new_dir_path)

                    #dicom2nifti.convert_generic.dicom_to_nifti(dir + person_dir, dir + '/final' + person_save)

    '''
    '''
    df_info = pd.read_csv("/gpfs/gibbs/pi/gerstein/sk2776/data/images/ADNI_2_fMRI_label_3_24_2023.csv")
    dir = '/gpfs/gibbs/pi/gerstein/sk2776/data/images/ADNI/final'
    id = os.listdir(dir)[0]
    print(id)
    item_name = os.listdir(dir + '/' + id)[0]
    path_to_file = dir + '/' + id + '/' + item_name
    item = nb.load(path_to_file)
    image_num = id.split("_")[8]
    label = df_info[df_info["Image Data ID"] == image_num]['Group'].values[0]
    person_id = df_info[df_info["Image Data ID"] == image_num]['Subject'].values[0]
    print(person_id)

                    
    img = nb.load('/gpfs/gibbs/pi/gerstein/sk2776/data/images/ADNI/full_brain/501_resting_state_fmri.nii.gz')
    img_data = img.get_fdata()


    /gpfs/gibbs/pi/gerstein/sk2776/data/images/ADNI/002_S_0295_I23862.nii
        (want)
        each patient (pid_date_img_id.*)
            -> 900 people (900~1800 3D scans)
        at least in same dimension
            case 1: 5k slices , m1*n1 for each slice
            case 2: 6k slices , m2*n2 for each slice
            -------process-----
            case: 5k slices (preferrably), m*n for each slice
            (final form) -> 5k*m*n (3D scan)

            resting state fMRI: 6720 slices
            
            
        (now)
        each patient has a folder
            pid/date/img_id/2d_slices
            002_S_0295/Resting_State_fMRI/2011-06-02_07_56_36.0/I238623/ADNI_002_S_0295_MR_Resting_State_fMRI_br_raw_20110602125224332_4803_S110474_I238623.dcm'
            (optional for future) (filtering, dim reduction) -> fewer features
    /gpfs/gibbs/pi/gerstein/xz584/fmgene
    '''

    '''
    code to verify that dimensions are the same 
    count = 0
    for filename in os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/ADNI/'):
        for filename2 in os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/ADNI/' + filename):
            if filename2 == 'Resting_State_fMRI':
                count += 1
                number = len([file for file in os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/ADNI/' + filename + '/' + 'Resting_State_fMRI')])
                if number != 6720:
                    for filename3 in os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/ADNI/' + filename + '/' + 'Resting_State_fMRI'):
                        for filename4 in os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/ADNI/' + filename + '/' + 'Resting_State_fMRI' + '/' + filename3):
                            number2 = len([file for file in os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/ADNI/' + filename + '/' + 'Resting_State_fMRI' + '/' + filename3 + '/' + filename4)])
                            if number2 != 6720:
                                print(filename)
                                print(number)
    print(count)
    '''
    #073_S_4443_MoCoSeries_2012-03-12_09_18_22.0_I289900
    #    141_S_4232_MoCoSeries_2011-09-24_15_09_48.0_I258054
    '''
    for folder in os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/MoCoSeries'):
        for image in os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/MoCoSeries/' + folder):
            item = nb.load('/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/MoCoSeries/' + folder + '/' + image)
            image_dim = item.shape
            if image_dim[0] != 64:
                print(folder)
                break
            elif image_dim[1] != 64:
                print(folder)
            elif image_dim[2] != 24:
                print(folder)
            elif image_dim[3] != 105:
                print("Not 105")
                print(image_dim[3])
                print(folder)
    '''
    '''
    count = 0
    total = 0

    #128_S_2130_Perfusion_Weighted_2012-01-18_15_22_13.0_I279210
    #019_S_4293_Resting_State_fMRI_2012-05-09_11_58_25.0_I302671 
    #010_S_4345_Resting_State_fMRI_2012-01-24_12_11_28.0_I282008 - only has one slice
    for folder in os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/all'):
        name = folder.split("_")
        for image in os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/all/' + folder):
            total += 1
            item = nb.load('/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/all/' + folder + '/' + image)
            image_dim = item.shape
            if image_dim[0] != 64:
                #all.append(folder)
                #print(image_dim)
               # print(folder)
                count += 1
            elif image_dim[1] != 64:
                #all.append(folder)
                #print(image_dim)
                count += 1
                #print(folder)
            elif image_dim[2] != 48:
                #print("Here")
                name = folder.split("_")
                if (name[3] == "Resting"):
                    print(image_dim)
                    print(name)
                    print(folder)
                count += 1
                #all.append(folder)
            elif name[3] == 'Resting' and image_dim[3] != 140:
                print(image_dim)
                print(folder)

            else:
                continue
    print(count/total)
               # print(item.get_fdata())

    
    for image in os.listdir('/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/all/' + '031_S_4029_Resting_State_fMRI_2011-08-08_11_24_50.0_I249328'):
        item = nb.load('/gpfs/gibbs/pi/gerstein/sk2776/data/images/Final/all/' + '031_S_4029_Resting_State_fMRI_2011-08-08_11_24_50.0_I249328' + '/' + image)
        print(item.shape)
      
    #031_S_4029_Resting_State_fMRI_2011-08-08_11_24_50.0_I249328
    '''

    '''
    target_shape = np.array((64,64,24))
    new_resolution = [2,]*3
    new_affine = np.zeros((4,4))    
    new_affine[:3,:3] = np.diag(new_resolution)
    new_affine[:3,3] = target_shape*new_resolution/2.*-1
    new_affine[3,3] = 1.
    downsampled_and_cropped_nii = resample_img(orig_nii, target_affine=new_affine, target_shape=target_shape, interpolation='nearest')
    '''

    '''
    gene_file = pd.read_csv("/gpfs/gibbs/pi/gerstein/sk2776/data/genetic/021_S_0647.csv")
    subset = gene_file[["SNP Name", "Theta"]]
    genetic_data = []
    size = subset.shape[0]
    '''
    #print(size)
    #print(subset)
    #for i in range(size):
        #genetic_data.append(gene_file.iloc[i]['Theta'])
    #print(genetic_data)
            #print(genetic_data)

        #for index1, row1 in subset:
            #genetic_data[index][index1] = self.gene_file.iloc[index1]['Theta'].values[0]
            #print(genetic_data)

    #ds = dicom.dcmread('/gpfs/gibbs/pi/gerstein/sk2776/data/images/ADNI/123_S_4526/MoCoSeries/2012-02-09_07_35_17.0/I283596/ADNI_123_S_4526_MR_MoCoSeries_br_raw_20120209093442675_10_S140262_I283596.dcm')
    #print(ds)
    #print(ds.pixel_array)
    #plt.imshow(ds.pixel_array)
    #plt.imshow(ds.pixel_array)

   # 2.16.124.113543.6006.99.0413083514818964728


    #I238623


    #Steps for gene data
    #1. read csv file for each person - store
    #2. build a matrix where we use SNP index number and theta value
