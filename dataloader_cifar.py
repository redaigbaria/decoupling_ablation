from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

def make_weights_for_balanced_classes(inputs, nclasses):                        
    count = [0] * nclasses                                                      
    for item in inputs:                                                         
        count[item] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(inputs)                                              
    for idx, val in enumerate(inputs):                                          
        weight[idx] = weight_per_class[val]                                  
    return weight

class cifar_dataset(Dataset):
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', clean_pred=[], clean_probability=[], relabel_indicator=[], label_preds=[]):

        self.r = r # noise ratio
        self.transform = transform
        self.mode = mode
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise

        if self.mode=='test':
            if dataset=='cifar10':
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']
        else:
            train_data=[]
            train_label=[]
            if dataset=='cifar10':
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset=='cifar100':
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            if os.path.exists(noise_file):
                print(f"loading noise file {noise_file}")
                noise_label = json.load(open(noise_file,"r"))
            else:    #inject noise
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.r*50000)
                noise_idx = idx[:num_noise]
                for i in range(50000):
                    if i in noise_idx:
                        if noise_mode=='sym':
                            if dataset=='cifar10':
                                noiselabel = random.randint(0,9)
                            elif dataset=='cifar100':
                                noiselabel = random.randint(0,99)
                            noise_label.append(noiselabel)
                        elif noise_mode=='asym':
                            noiselabel = self.transition[train_label[i]]
                            noise_label.append(noiselabel)
                    else:
                        noise_label.append(train_label[i])
                print("save noisy labels to %s ..."%noise_file)
                json.dump(noise_label,open(noise_file,"w"))
            noise_label = np.array(noise_label)
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:
                labeled_indicator = (clean_pred | relabel_indicator)
                if self.mode == "labeled":
                    true_relabel = (1 - clean_pred) & relabel_indicator
                    pred_idx = labeled_indicator.nonzero()[0]
                    relabel_idx = true_relabel.nonzero()[0]
                    
                    import matplotlib.pyplot as plt
                    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
                    plt.clf()
                    print('Confusion matrix before re-labeling')
                    clean = noise_label == np.array(train_label)
                    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(clean, clean_pred)).plot()
                    plt.show()

                    # re-labeling
                    noise_label[relabel_idx] = label_preds[relabel_idx]
                    clean = (np.array(noise_label)==np.array(train_label))
                    plt.clf()
                    print('Confusion matrix after re-labeling')
                    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(clean, labeled_indicator)).plot()
                    plt.show()
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(clean_probability,clean)
                    auc,_,_ = auc_meter.value()
                    print('Numer of labeled samples:%d   AUC:%.3f\n'%(labeled_indicator.sum(),auc))


                elif self.mode == "unlabeled":
                    pred_idx = (1 - labeled_indicator).nonzero()[0]

                self.noise_label = [noise_label[i] for i in pred_idx]
                self.train_data = train_data[pred_idx]
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))

    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            return img1, target
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
        
class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])
        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
    def run(self,mode,clean_pred=[],clean_probability=[], relabel_indicator=[], label_preds=[], CBS=False):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

        elif mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="labeled", noise_file=self.noise_file, clean_pred=clean_pred, clean_probability=clean_probability, relabel_indicator=relabel_indicator, label_preds=label_preds)
            if CBS:
                print('using CBS')
                # TODO: change when you correct Stage 2 re-labeling
                weights = make_weights_for_balanced_classes(labeled_dataset.noise_label, 10 if self.dataset == 'cifar10' else 100)
                cb_sampler = torch.utils.data.WeightedRandomSampler(weights, len(labeled_dataset), replacement=True)
                labeled_trainloader = DataLoader(
                    dataset=labeled_dataset,
                    batch_size=self.batch_size,
                    sampler= cb_sampler,
                    drop_last=True,
                    num_workers=self.num_workers)
            else:
                labeled_trainloader = DataLoader(
                    dataset=labeled_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=self.num_workers)

            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled", noise_file=self.noise_file, clean_pred=clean_pred, relabel_indicator=relabel_indicator)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader

