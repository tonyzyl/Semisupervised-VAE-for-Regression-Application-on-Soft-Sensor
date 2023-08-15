import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from .scheduler import CosineAnnealingWarmupRestarts
import copy

class FC(nn.Module):
    def __init__(self, dims_list, act=None, output_act=None):
        super(FC, self).__init__()

        neurons = dims_list[:-1]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(linear_layers)
        self.act_func = act
        self.lastLayer = nn.Linear(neurons[-1], dims_list[-1])

        self.output_act = output_act

        # weights initialization could have a large effect on final result
        self._weight_initialization(self.hidden, self.act_func)
        self._weight_initialization(self.lastLayer, self.output_act)
    
    def _weight_initialization(self,layers, act_func):
        for m in layers.modules():
            if isinstance(m, nn.Linear):
                if type(act_func) == type(nn.Tanh()):
                    nn.init.xavier_uniform_(m.weight.data)
                elif type(act_func) == type(nn.ReLU()):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif type(act_func) == type(nn.LeakyReLU()):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='leaky_relu')
                else:
                    nn.init.xavier_normal_(m.weight.data)
                    #nn.init.trunc_normal_(m.weight.data, a=-1.,b=1.)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        for layer in self.hidden:
            if self.act_func is not None:
                x = self.act_func(layer(x))
            else:
                x = layer(x)
        if self.output_act is not None:
            return self.output_act(self.lastLayer(x))
        else:
            return self.lastLayer(x)


class FCN(nn.Module):
    def __init__(self, nn_dim: list) -> None:
        super(FCN, self).__init__()
        
        self.fcn = FC(nn_dim, act=nn.Tanh(), output_act=None)

    def forward(self, x):
        pred = self.fcn(x)
        return pred

class FCN_Trainer():
    def __init__(self, *, dataloader_dict: dict, model, epoch: int,lr: float, scheduler=None, verbose=True):
        self.dataloader_dict = dataloader_dict
        self.model = model
        self.lr = lr
        self.epoch = epoch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr) 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        self.scheduler = scheduler
        if self.scheduler is not None:
            print('Using CosineAnnealingWarmupRestarts lr.')
            self.scheduler = CosineAnnealingWarmupRestarts(self.optimizer,
                                          first_cycle_steps=300,
                                          cycle_mult=1.0,
                                          max_lr=self.lr,
                                          min_lr=0.0001,
                                          warmup_steps=60,
                                          gamma=0.5)
        self.verbose = verbose
        print(f"Using {self.device} device")
        print("PyTorch Version: ",torch.__version__)
        self.model.to(self.device)

    def loss_func(self, orig_inputs_x, labels, unlabel_idx):
        label_idx = torch.ones(unlabel_idx.size(), dtype=int, device=unlabel_idx.device)-unlabel_idx
        orig_inputs_x = torch.index_select(orig_inputs_x, 0, torch.argwhere(label_idx)[:,0])
        labels = torch.index_select(labels, 0, torch.argwhere(label_idx)[:,0])
        pred = self.model(orig_inputs_x)
        return F.mse_loss(pred, labels)
    
    def _do_epoch(self, phase: str):
        sum_loss = 0.
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        for inputs, labels, unlabel_idx in self.dataloader_dict[phase]:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            unlabel_idx = unlabel_idx.to(self.device)
            self.optimizer.zero_grad()
            loss= self.loss_func(inputs, labels, unlabel_idx)

            if phase == 'train':
                loss.backward()
                self.optimizer.step()
            sum_loss+= loss.item()
        return sum_loss/len(self.dataloader_dict[phase].sampler)

    def train(self):
        best_layer_wts = copy.deepcopy(self.model.state_dict())
        best_loss = np.inf

        for self._cur_epoch in range(self.epoch):
            train_loss = self._do_epoch("train")
            if self.scheduler is not None:
                self.scheduler.step()
                if self._cur_epoch % 10 == 0 and self.verbose:
                    print('last lr: ',self.scheduler.get_lr())
            with torch.no_grad():
                val_loss = self._do_epoch("val")
            if val_loss < best_loss:
                best_loss = val_loss
                best_label = val_loss
                best_layer_wts = copy.deepcopy(self.model.state_dict())  
                best_epoch = self._cur_epoch
                #torch.save(self.model.state_dict(), "model.pth")
            if self._cur_epoch % 10 == 0 and self.verbose:
                print(f"Epoch {self._cur_epoch+1}/{self.epoch}, train total: {train_loss:>5f}, val total: {val_loss:>5f}.")
        self.model.load_state_dict(best_layer_wts)
        print(f'Best loss: {best_loss:>7f}, best label loss: {best_label:>7f}, @ epoch: {best_epoch}')
        return best_loss

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