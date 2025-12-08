import logging
logging.basicConfig(level=logging.ERROR)
import torch
import torchvision
from torch.autograd import Variable
from torch import nn
from torch import optim
from torchvision import datasets
import torchvision.transforms as transforms
#from sequential_mnist import loadSequentialMNIST
from dual_pathway import ConvLMU2

import random
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
import numpy as np
import argparse
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from spikingjelly.activation_based import functional
from scipy.signal import cont2discrete
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
    #IndRNNModel
def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample
import warnings
warnings.filterwarnings('ignore')
class shdDataset(Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.samples = np.loadtxt(self.data_paths+'.txt').astype('int')
        self.transform = transform

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):  
        inputindex = self.samples[index, 0]
        label = self.samples[index, 1]
        path = self.data_paths + '/'+str(inputindex.item()) + '.npy'
        input = torch.from_numpy(np.load(path))

        input = input.reshape(-1, 140, 5)
        input =  input.sum(dim=-1).squeeze()
        label = torch.from_numpy(np.asarray(label))
        #if self.transform:
            
        #    x = self.transform(x)
        return input.float(), label
        
        

def train(model, loader, optimizer, criterion, batchSize):
    """ A single training epoch on the dataset """

    epoch_loss = 0
    y_pred = []
    y_true = []

    model.train()
    for batch_idx, (batch, labels) in enumerate(loader):
        batch = batch.to(DEVICE)
        labels = labels.long().to(DEVICE)

        optimizer.zero_grad()  # Clear previous gradients

        output = model(batch)  # Forward pass
        loss = criterion(output, labels)

        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        functional.reset_net(model)
        preds = output.argmax(dim=1)
        y_pred += preds.tolist()
        y_true += labels.tolist()
        epoch_loss += loss.item()

    return epoch_loss / len(loader), accuracy_score(y_true, y_pred)

def validate(model, loader, criterion, batchSize):
    """ A single validation epoch on the psMNIST data """

    epoch_loss = 0
    y_pred = []
    y_true = []
    
    model.eval()
    with torch.no_grad():
        #for batch, labels in tqdm(loader):
        for batch_idx, (batch, labels) in enumerate(loader):
            #torch.cuda.empty_cache()

            batch = batch.to(DEVICE)
            labels = labels.long().to(DEVICE)

            output = model(batch)
            loss = criterion(output, labels)
            
            preds  = output.argmax(dim = 1)
            y_pred += preds.tolist()
            y_true += labels.tolist()
            epoch_loss += loss.item()
            functional.reset_net(model)            
    # Loss
    avg_epoch_loss = epoch_loss / len(loader)

    # Accuracy
    epoch_acc = accuracy_score(y_true, y_pred)

    return avg_epoch_loss, epoch_acc

def countParameters(model):
    """ Counts and prints the number of trainable and non-trainable parameters of a model """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"The model has {trainable:,} trainable parameters and {frozen:,} frozen parameters")
    

def setSeed(seed):
    """ Set all seeds to ensure reproducibility """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    
if __name__ == '__main__':
    # Connect to GPU
    if torch.cuda.is_available():
        DEVICE = "cuda"
        # Clear cache if non-empty
        #torch.cuda.empty_cache()
        # See which GPU has been allotted 
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        DEVICE = "cpu"

    SEED = 42
    #SEED = 3407
    setSeed(SEED)
    batchSize = 256 
    N_epochs = 200

    parser = argparse.ArgumentParser(description='provide arguments')

    # simulation parameters
    parser.add_argument('--delay_size',  type=int, default=40,help='defulat GPU number')
    parser.add_argument('--batchsize', type =int,default=128)
    parser.add_argument('-d', type =int,default=10)
    parser.add_argument('-t', type =int,default=250)
    args = parser.parse_args()
    batchSize = args.batchsize
    
    trainingSet = shdDataset('data/train')      
    dl_train = DataLoader(dataset=trainingSet, batch_size=256, shuffle=True, num_workers=8) 
    testSet = shdDataset('data/test')   
    dl_test = DataLoader(dataset=testSet, batch_size=256, shuffle=False, num_workers=8)     
    valSet = shdDataset('data/valid')    
    dl_val = DataLoader(dataset=valSet, batch_size=256, shuffle=False, num_workers=8)          

    dd= args.d
    TT = 250
    t = args.t
    model = ConvLMU2(dd, TT, t).to(DEVICE)
    optimizer = optim.Adam(params = model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=80, gamma=.5) # LIF
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(DEVICE)


    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_acc = 0

    for epoch in range(200):
   
        print("Epoch:", epoch)

        train_loss, train_acc = train(model, dl_train, optimizer, criterion, batchSize)
        val_loss, val_acc = validate(model, dl_val, criterion, batchSize)
        scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print("Train Loss:  | Train Acc:",train_loss,train_acc)
        print("Val. Loss:  |  Val. Acc: ",val_loss,val_acc )
        if val_acc > best_acc:
            best_acc =  val_acc
            test_loss, test_acc = validate(model, dl_test, criterion, batchSize)
            best_real_acc = test_acc
            print('acc',best_acc, best_real_acc)
            torch.save(model.state_dict(),  'ssc_last.pkl')



