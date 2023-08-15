import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from .scheduler import CosineAnnealingWarmupRestarts
import copy

class UnitNormLinear(nn.Linear):
    def unit_norm_(self,w):
        with torch.no_grad():
            norm =torch.linalg.norm(w, dim=0, ord=2)
            return nn.Parameter(w/(1e-10+norm)) 

    def forward(self, input):
        self.weight = self.unit_norm_(self.weight)
        return F.linear(input, self.weight, self.bias)
        
class UnitNormFC(nn.Module):
    def __init__(self, dims_list, act=None, output_act=None):
        super(UnitNormFC, self).__init__()

        neurons = dims_list[:-1]
        linear_layers = [UnitNormLinear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(linear_layers)
        self.act_func = act
        self.lastLayer = UnitNormLinear(neurons[-1], dims_list[-1])

        self.output_act = output_act

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


class SSVAE(nn.Module):
    def __init__(self, share_dim: list, encode_dim: list, regressor_dim: list, latentgen_dim:list) -> None:
        super(SSVAE, self).__init__()
        if share_dim[-1] != encode_dim[0]:
            raise ValueError('The numer of neurons at the end of shared encoder, {}, does not match start of latent encoder, {}.'.format(share_dim[-1], encode_dim[0]))
        if encode_dim[-1] != latentgen_dim[-1]:
            raise ValueError('The numer of neurons at the end of latent encoder, {}, does not match end of latent generator, {}.'.format(encode_dim[-1], latentgen_dim[-1]))
        if regressor_dim[-1] !=1:
            raise ValueError('The numer of neurons at the end of regressor, {}, does not equal to 1'.format(regressor_dim[-1]))
        if latentgen_dim[0] !=2:
            raise ValueError('The numer of neurons at the start of latent generator, {}, does not equal to 2.'.format(latentgen_dim[0]))
        self.share_dim = share_dim
        self.encode_dim = encode_dim
        self.regressor_dim = regressor_dim
        self.latentgen_dim = latentgen_dim

        self.share_encoder = FC(self.share_dim, act=nn.Tanh(), output_act=nn.Tanh())
        self.mean_encoder = FC(self.encode_dim, act=nn.ReLU(), output_act=None)
        self.logvar_encoder = FC(self.encode_dim, act=nn.ReLU(), output_act=None)
        self.y_mean_regressor = FC(self.regressor_dim, act=nn.Tanh(), output_act=None)
        self.y_logvar_regressor = FC(self.regressor_dim, act=nn.ReLU(), output_act=None)
        self.trend_regressor = FC(self.regressor_dim, act=nn.Tanh(), output_act=nn.Tanh())
        self.latent_generator = FC(self.latentgen_dim, act=nn.ReLU(), output_act=None)
        self.decoder = FC([*list(reversed(encode_dim)), *list(reversed(share_dim[:-1]))], act=nn.ReLU(), output_act=None)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        x = self.share_encoder(x)
        pred = self.y_mean_regressor(x)
        return pred

class SSVAE_Trainer():
    def __init__(self, *, dataloader_dict: dict, model, epoch: int,lr: float, scheduler=None, verbose=True, betaSchedule=None):
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
        self.betaSchedule = betaSchedule
        if self.betaSchedule is not None:
            print('Using monotonic beta schedule, cycle:{}, ratio:{}'.format(*self.betaSchedule))
            self.betaSchedule = self.frange_cycle_linear(0.,1.,self.epoch, self.betaSchedule[0], self.betaSchedule[1])
        self.verbose = verbose
        print(f"Using {self.device} device")
        print("PyTorch Version: ",torch.__version__)
        self.model.to(self.device)

    def frange_cycle_linear(self, start, stop, n_epoch, n_cycle=4, ratio=0.5):
        L = np.ones(n_epoch)
        period = n_epoch/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule

        for c in range(n_cycle):

            v , i = start , 0
            while v <= stop and (int(i+c*period) < n_epoch):
                L[int(i+c*period)] = v
                v += step
                i += 1
        return L 

    def loss_func(self, orig_inputs_x, next_orig_inputs, labels, unlabel_idx):
        label_idx = torch.ones(unlabel_idx.size(), dtype=int, device=unlabel_idx.device)-unlabel_idx

        inputs_x = self.model.share_encoder(orig_inputs_x)
        y_mean = self.model.y_mean_regressor(inputs_x)
        y_logvar = self.model.y_logvar_regressor(inputs_x)
        y_resample = self.model.reparameterize(y_mean, y_logvar)
        z_mean = self.model.mean_encoder(inputs_x)
        z_logvar = self.model.logvar_encoder(inputs_x)
        z_resample = self.model.reparameterize(z_mean, z_logvar)
        dy_mean = self.model.trend_regressor(inputs_x)
        decode_x = self.model.decoder(z_resample)
        latent_gen_input = torch.cat((y_resample, dy_mean), dim=1)
        z_gen = self.model.latent_generator(latent_gen_input)
        z_gen_resample = self.model.reparameterize(z_gen, torch.ones_like(z_gen, device=z_gen.device))
        z_gen_resample = z_gen_resample.detach()
        decode_x_gen = self.model.decoder(z_gen_resample)

        with torch.no_grad():
            next_inputs = self.model.share_encoder(next_orig_inputs)
            next_z_logvar = self.model.logvar_encoder(next_inputs)
            next_z_mean = self.model.mean_encoder(next_inputs)
            next_ydy_resample = self.model.reparameterize((y_mean+dy_mean),y_logvar)
            next_dy_mean = self.model.trend_regressor(next_inputs)
            next_z_gen = self.model.latent_generator(torch.cat((next_ydy_resample, next_dy_mean), dim=1))

        reconstruct_loss = F.mse_loss(orig_inputs_x, decode_x)
        reconstruct_loss += F.mse_loss(orig_inputs_x, decode_x_gen)

        kld_loss = torch.mean(-0.5*(1 + z_logvar - (z_mean-z_gen).pow(2) - z_logvar.exp()).sum(dim=1))
        kld_loss += torch.mean(-0.5*(1 + next_z_logvar-(next_z_mean-next_z_gen).pow(2) - next_z_logvar.exp()).sum(dim=1))

        unlabel_y_logvar = torch.index_select(y_logvar, 0, torch.argwhere(unlabel_idx)[:,0])
        pred = torch.index_select(y_mean, 0, torch.argwhere(label_idx)[:,0])
        labelled_y_logvar = torch.index_select(y_logvar, 0, torch.argwhere(label_idx)[:,0])
        labelled_labels = torch.index_select(labels, 0, torch.argwhere(label_idx)[:,0])
        unlabel_entropy = torch.mean(0.5+0.5*torch.log(2*np.pi*unlabel_y_logvar.exp()))
        labelled_loss = torch.mean(-0.5+0.5*(torch.div((pred-labelled_labels).pow(2), labelled_y_logvar.exp())+labelled_y_logvar))
        label_loss = 0
        if not torch.isnan(labelled_loss).item() and torch.isnan(unlabel_entropy).item():
            label_loss += labelled_loss
        elif torch.isnan(labelled_loss).item() and not torch.isnan(unlabel_entropy).item():
            label_loss += unlabel_entropy
        else:
            label_loss += unlabel_entropy+labelled_loss
        if self.betaSchedule is not None:
            #beta = min(self.betaSchedule[0],self._cur_epoch/self.betaSchedule[1])
            #kld_loss = kld_loss*beta
            kld_loss = kld_loss*self.betaSchedule[self._cur_epoch]
        loss = reconstruct_loss + kld_loss + label_loss
        return loss, reconstruct_loss, kld_loss, label_loss
    
    def _do_epoch(self, phase: str):
        avg_loss = 0.
        avg_reconstruct_loss = 0.
        avg_kld_loss = 0.
        avg_label_loss = 0.
        sample_num = len(self.dataloader_dict[phase].sampler)
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        for inputs, next_inputs, labels, unlabel_idx in self.dataloader_dict[phase]:
            inputs = inputs.to(self.device)
            next_inputs = next_inputs.to(self.device)
            labels = labels.to(self.device)
            unlabel_idx = unlabel_idx.to(self.device)
            self.optimizer.zero_grad()
            loss, reconstruct_loss, kld_loss, label_loss = self.loss_func(inputs, next_inputs, labels, unlabel_idx)

            if phase == 'train':
                loss.backward()
                self.optimizer.step()
            avg_loss += loss.item() * inputs.size(0)
            avg_reconstruct_loss += reconstruct_loss.item() * inputs.size(0)
            avg_kld_loss += kld_loss.item() * inputs.size(0)
            avg_label_loss += label_loss.item() * inputs.size(0)
        return avg_loss/sample_num, avg_reconstruct_loss/sample_num, avg_kld_loss/sample_num, avg_label_loss/sample_num

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
            if val_loss[0] < best_loss:
                best_loss = val_loss[0]
                best_label = val_loss[3]
                best_layer_wts = copy.deepcopy(self.model.state_dict())  
                best_epoch = self._cur_epoch
                #torch.save(self.model.state_dict(), "model.pth")
            if self._cur_epoch % 10 == 0 and self.verbose:
                print(f"Epoch {self._cur_epoch+1}/{self.epoch}, train total: {train_loss[0]:>5f},reconstruct: {train_loss[1]:>5f},kld: {train_loss[2]:>5f},label: {train_loss[3]:>5f}, \n val total: {val_loss[0]:>5f},reconstruct: {val_loss[1]:>5f},kld: {val_loss[2]:>5f},label: {val_loss[3]:>5f}")
        self.model.load_state_dict(best_layer_wts)
        print(f'Best loss: {best_loss:>7f}, best label loss: {best_label:>7f}, @ epoch: {best_epoch}')
        return best_loss

class MyDataset(Dataset):
    # Initialization
    def __init__(self, data, next_data, label, unlabel_idx, mode='2D'):
        self.data, self.next_data, self.label, self.mode, self.unlabel_idx= data, next_data, label, mode, unlabel_idx

    # Get item
    def __getitem__(self, index):
        if self.mode == '2D':
            return self.data[index, :], self.next_data[index, :], self.label[index, :], self.unlabel_idx[index, :]
        elif self.mode == '3D':
            return self.data[:, index, :], self.next_data[:, index, :], self.label[:, index, :], self.unlabel_idx[:, index, :]

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