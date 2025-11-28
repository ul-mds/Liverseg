from collections import OrderedDict
from omegaconf import DictConfig
#from model.unet import test
from training_evaluation import train, test
import torch
import utilities.losses as losses

from hydra.utils import instantiate

def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round:int):
        return {'lr': config['lr'], 'momentum': config['momentum']}
    
    return fit_config_fn

def get_evaluate_fn(cfg, testloader, device):
    def evaluate_fn(server_round: int, parameters, config):
        model = instantiate(cfg.model)
        
        parameters = [torch.Tensor(v) if not(len(v.shape) == 0) else torch.empty(()) for v in parameters]
        
        params_dict =  zip(model.state_dict().keys(), parameters)
        
        state_dict = OrderedDict({k: v  for k,v in params_dict })
        
        if cfg.config_fit.loss == 'BCEWithLogitsLoss':
            criterion = torch.nn.BCEWithLogitsLoss().to(device)
        else:
            criterion = losses.BCEDiceLoss().to(device)
        
        model.load_state_dict(state_dict,strict =True)
        loss, log = test(model, testloader, criterion, device=device)
        
        return loss, log

    return evaluate_fn

def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    print(metrics)
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": int(sum(accuracies)) / int(sum(examples))}
