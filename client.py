from typing import Dict
import flwr as fl
from flwr.common import NDArrays, Scalar, ndarrays_to_parameters
from collections import OrderedDict
#from model.unet import train, validate
from training_evaluation import train, validate
import torch
from torch import nn
import torch.optim as optim

import utilities.losses as losses

from hydra.utils import instantiate

class LiverClient(fl.client.NumPyClient):
    def __init__(self,
                 trainloader,
                 valloader,
                 cfg,
                 device) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = valloader

        self.model = instantiate(cfg.model)       
        self.cfg = cfg
        self.device = device
    
    def set_parameters(self, parameters):
        
        parameters = [torch.Tensor(v)  if not(len(v.shape) == 0) else torch.empty(()) for v in parameters]
        params_dict =  zip(self.model.state_dict().keys(), parameters)
         
        state_dict = OrderedDict({k: torch.Tensor(v) for k,v in params_dict})
        self.model.load_state_dict(state_dict,strict =True)
        
    def get_parameters(self, config:Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]


    def fit(self, parameters, config):
        #copy parameters sent by the server into client's local model
        self.set_parameters(parameters)
        lr = self.cfg.config_fit['lr']
        momentum = self.cfg.config_fit['momentum']
        epochs = self.cfg.local_epochs
        optimizer_name = self.cfg.config_fit['optimizer']
        loss = self.cfg.config_fit['loss']
        
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr) 
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, momentum=momentum)
        
        # define loss function (criterion)
        if loss == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = losses.BCEDiceLoss()
        
        #do local training
        for epoch in range(self.cfg.local_epochs):
            train(self.model,self.trainloader,self.valloader,optimizer,criterion,self.device, epoch)
        
        return self.get_parameters({}), len(self.trainloader), {}
    
    def evaluate(self,parameters:NDArrays, criterion, config: Dict[str, Scalar]=None):
        # define loss function (criterion)
        if self.cfg.config_fit['loss'] == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = losses.BCEDiceLoss()
        self.set_parameters(parameters)
        loss, log = validate(self.model, self.valloader,criterion, self.device)
        
        return float(loss), len(self.valloader), log

def generate_client_fn(trainloader, valloaders, cfg, device):
    def client_fn(cid:str):
        return LiverClient(trainloader=trainloader[int(cid)], valloader=valloaders[int(cid)], cfg=cfg, device=device).to_client()
    return client_fn
