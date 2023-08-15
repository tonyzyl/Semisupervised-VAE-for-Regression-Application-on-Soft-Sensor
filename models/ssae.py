from tabnanny import verbose
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from .scheduler import CosineAnnealingWarmupRestarts
import copy
import pickle

class AE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)
        self.y_layer = nn.Linear(latent_dim, 1, bias=True)

        nn.init.xavier_uniform_(self.encoder.weight.data)
        nn.init.xavier_uniform_(self.decoder.weight.data)
        nn.init.xavier_normal_(self.y_layer.weight.data)

    def forward(self, X, decode=True):
        # Return (latent/decode output, Y estimate by dense)
        H = torch.tanh(self.encoder(X))
        if decode:
            return torch.tanh(self.decoder(H)), self.y_layer(H)
        else:
            return H, self.y_layer(H)

class AE_ReLU(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)
        self.y_layer = nn.Linear(latent_dim, 1, bias=True)

    def forward(self, X):
        # Return (latent/decode output, Y estimate by dense)
        H = nn.functional.relu(self.encoder(X))
        return H, torch.sigmoid(self.y_layer(H))

class SSAE(nn.Module):
    def __init__(self,AE_list):
        super().__init__()
        self.num_AE = len(AE_list)
        self.SAE_list=[]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for i in range(1, self.num_AE+1):
            if i != self.num_AE:
                self.SAE_list.append(AE(AE_list[i-1], AE_list[i]).to(self.device))
            else:
                self.SAE_list.append(AE(AE_list[-1], AE_list[-1]).to(self.device))
    
    def wgtFromList(self, wts_list):
        for i in range(self.num_AE):
            self.SAE_list[i].load_state_dict(wts_list[i])
    
    def forward(self, x, layer_idx, preTrain = False):
        # preTrain: previous layers' parameters are frozen 
        # preTrain -> Return (input, AE_output, y estimate)
        # !preTrain -> Return last layer's (latent, estimate)
        output = x
        if preTrain:
            if layer_idx == 0:
                inputs = output
                output, y_estimate = self.SAE_list[layer_idx](output, decode=True)                
                return inputs, output, y_estimate

            else:
                for i in range(layer_idx):
                    for param in self.SAE_list[i].parameters():
                        param.requires_grad = False
                    output,_ = self.SAE_list[i](output, decode = False)
                inputs = output
                output, y_estimate = self.SAE_list[layer_idx](output, decode=True)                
                return inputs, output, y_estimate
        else:
            for i in range(self.num_AE-1):
                for param in self.SAE_list[i].parameters():
                    param.requires_grad = True
                output, _ = self.SAE_list[i](output, decode = False)
            return self.SAE_list[-1](output, decode = False)

def loss_func(input_encode, output_latent, y_estimate,  y_label, unlabel_idx, tradeoff_param):
    N_total = unlabel_idx.size(dim=0)
    N_u = torch.sum(unlabel_idx)
    N_l = N_total - N_u
    label_idx = torch.ones(unlabel_idx.size(), dtype=int, device=unlabel_idx.device)-unlabel_idx
    if N_u != 0:
        unlabel_term = (N_total/(2*N_u))*nn.functional.mse_loss(unlabel_idx*input_encode, unlabel_idx*output_latent)
    else:
        unlabel_term = 0
    if N_l != 0:
        label_term = (N_total/(2*N_l))*(nn.functional.mse_loss(label_idx*input_encode, label_idx*output_latent)+
                    tradeoff_param*nn.functional.mse_loss(label_idx*y_estimate, label_idx*y_label))
    else:
        label_term = 0
    return unlabel_term+label_term

class SSAE_Trainer:
    def __init__(self, *, dataloader_dict: dict, model, pretrain_lr: float, pretrain_epoch: int, finetune_lr: float, finetune_epoch: int, tradeoff_param: int, scheduler=None, verbose=None):
        self.dataloader_dict = dataloader_dict
        self.model = model
        self.pretrain_lr = pretrain_lr
        self.finetune_lr = finetune_lr
        self.criterion = nn.MSELoss()
        self.pretrain_epoch = pretrain_epoch
        self.finetune_epoch = finetune_epoch
        self.tradeoff_param =tradeoff_param
        self.scheduler = scheduler
        self.verbose = verbose
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        print(f"Using {self.device} device")
        print("PyTorch Version: ",torch.__version__)
        
        self.model.to(self.device)
    
    def _do_pretrain_epoch(self, phase: str, layer_idx: int, optimizer: object):
        sum_loss = 0.
        if phase == 'train':
            self.model.SAE_list[layer_idx].train()
        else:
            self.model.SAE_list[layer_idx].eval()
        
        for inputs, labels, unlabel_idx in self.dataloader_dict[phase]:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            unlabel_idx = unlabel_idx.to(self.device)
            optimizer.zero_grad()
            input_encode,output_latent,y_estimate = self.model(inputs,layer_idx,preTrain=True)
            loss = loss_func(input_encode, output_latent, y_estimate, labels, unlabel_idx, self.tradeoff_param)

            if phase == 'train':
                loss.backward()
                optimizer.step()
            sum_loss+= loss.item()
        return sum_loss/len(self.dataloader_dict[phase].sampler)
    
    def pretrain(self):
        for layer_idx in range(self.model.num_AE):
            print('Pre-training on the {} layer'.format(layer_idx))
            # Pretraining of each layer
            optimizer = torch.optim.Adam(self.model.SAE_list[layer_idx].parameters(), lr = self.pretrain_lr)
            best_layer_wts = copy.deepcopy(self.model.SAE_list[layer_idx].state_dict())
            best_loss = np.inf

            for epoch in range(self.pretrain_epoch):
                train_loss = self._do_pretrain_epoch("train", layer_idx=layer_idx, optimizer=optimizer)
                with torch.no_grad():
                    val_loss = self._do_pretrain_epoch("val", layer_idx=layer_idx, optimizer=optimizer)
                    #scheduler.step(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_layer_wts = copy.deepcopy(self.model.SAE_list[layer_idx].state_dict())  
                    #torch.save(self.model.state_dict(), "new_best_pretrain.pth")
                if epoch % 9 == 0 and self.verbose:
                    print(f"Epoch {epoch+1}/{self.pretrain_epoch}, train_loss: {train_loss:>7f}, val_exp_loss: {val_loss:>7f}")
            self.model.SAE_list[layer_idx].load_state_dict(best_layer_wts)
            print(f'Best loss: {best_loss:>7f}')
        print('Pretrain end\n')

    def _do_finetune_epoch(self, phase: str, optimizer: object):
        sum_loss = 0.
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        for tmp_inputs, tmp_labels, unlabel_idx in self.dataloader_dict[phase]:
            unlabel_idx = unlabel_idx.to(self.device)
            if unlabel_idx.size(dim=0) == torch.sum(unlabel_idx):
                # All data in the batch is unlabelled, bypass
                continue
            label_idx = torch.ones(unlabel_idx.size(), dtype=int, device=self.device)-unlabel_idx
            inputs = torch.index_select(tmp_inputs, 0, torch.argwhere(label_idx)[:,0]).to(self.device)
            labels = torch.index_select(tmp_labels, 0, torch.argwhere(label_idx)[:,0]).to(self.device)
            optimizer.zero_grad()
            _, y_estimate = self.model(inputs,self.model.num_AE-1,preTrain=False)
            loss = self.criterion(y_estimate, labels)
            if phase == 'train':
                loss.backward()
                optimizer.step()
            sum_loss += loss.item()
        return sum_loss/len(self.dataloader_dict[phase])
    
    def finetune(self):
        print('Start Finetuning\n')
        param_list=[]
        best_wts_list=[]
        best_loss = np.inf
        for tmp_sub_model in self.model.SAE_list:
            best_wts_list.append(tmp_sub_model.state_dict())
            for tmp_param in tmp_sub_model.parameters():
                param_list.append(tmp_param)
        optimizer = torch.optim.Adam(param_list, lr=self.finetune_lr)
        if self.scheduler is not None:
            print('Using CosineAnnealingWarmupRestarts lr.')
            self.scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                            first_cycle_steps=250,
                                            cycle_mult=1.0,
                                            max_lr=self.finetune_lr,
                                            min_lr=0.0001,
                                            warmup_steps=20,
                                            gamma=0.5)

        for epoch in range(self.finetune_epoch):
            train_loss = self._do_finetune_epoch("train", optimizer=optimizer)
            if self.scheduler is not None:
                self.scheduler.step()
                if epoch % 10 == 0 and self.verbose:
                    print('last lr: ',self.scheduler.get_lr())
            with torch.no_grad():
                val_loss = self._do_finetune_epoch("val", optimizer=optimizer)
            if val_loss < best_loss:
                best_loss = val_loss
                for AE_idx in range(self.model.num_AE):
                    best_wts_list[AE_idx] = copy.deepcopy(self.model.SAE_list[AE_idx].state_dict())
            if epoch % 10 == 0 and self.verbose:
                print(f"Epoch {epoch+1}/{self.finetune_epoch}, train_loss: {train_loss:>7f}, val_exp_loss: {val_loss:>7f}")
        self.model.wgtFromList(best_wts_list)
        with open("best_wts_list", "wb") as fp:   #Pickling
            pickle.dump(best_wts_list, fp)
    
    def train(self):
        self.pretrain()
        self.finetune()

class MyDataset(Dataset):
    # Initialization
    def __init__(self, data, label, unlabel_idx, mode='2D'):
        self.data, self.label, self.mode, self.unlabel_idx= data, label, mode, unlabel_idx

    # Get item
    def __getitem__(self, index):
        if self.mode == '2D':
            return self.data[index, :], self.label[index, :], self.unlabel_idx[index, :]
        elif self.mode == '3D':
            return self.data[:, index, :], self.label[:, index, :], self.unlabel_idx[:, index, :]

    # Get length
    def __len__(self):
        if self.mode == '2D':
            return self.data.shape[0]
        elif self.mode == '3D':
            return self.data.shape[1]

    def getNumpyLabel(self):
        return self.label.cpu().detach().numpy()
    def getTensorTrain(self):
        return self.data