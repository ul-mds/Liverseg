import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import flwr as fl
import pickle
import torch

from pathlib import Path 

from server import get_evaluate_fn, get_on_fit_config, weighted_average

from omegaconf import DictConfig, OmegaConf
from datasets.livseg import prepare_dataset,prepare_multiple_datasets
from client import generate_client_fn


@hydra.main(config_path="conf",config_name="base", version_base=None)
def main(cfg:DictConfig):

    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    ## 2. Prepare your dataset
    #trainloaders, validationloaders, testloader = prepare_dataset(cfg.dataset)
    #trainloaders, validationloaders, testloaders = prepare_multiple_datasets(cfg.dataset) 
    trainloaders, testloaders = prepare_multiple_datasets(cfg.dataset)
    print(len(trainloaders), len(trainloaders[0].dataset))
    #print(len(trainloaders), len(trainloaders[1].dataset))
    #print(len(trainloaders), len(trainloaders[2].dataset))

    #print(len(validationloaders), len(validationloaders[0].dataset))
    #print(len(validationloaders), len(validationloaders[1].dataset))
    #print(len(validationloaders), len(validationloaders[2].dataset))

    print(len(testloaders), len(testloaders[0].dataset))
    print(len(testloaders), len(testloaders[1].dataset))
    print(len(testloaders), len(testloaders[2].dataset))
    #print(len(testloaders), len(testloaders[3].dataset))

    
    
    ## 3. Set the available GPU/CPU device (GPU has the priority)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    #device = 'cpu'
    ## 4. Define your clients
    client_fn = generate_client_fn(trainloaders, testloaders[:-1], cfg, device) 
           
    ## 5. Define the federated strategy
    strategy = instantiate(cfg.strategy,
                           evaluate_fn=get_evaluate_fn(cfg, testloaders, device),
                           evaluate_metrics_aggregation_fn=weighted_average
                           )
    
    
    ## 6. Start Simulation
    history = fl.simulation.start_simulation(client_fn=client_fn,
                                             num_clients=cfg.num_clients,
                                             config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
                                             strategy=strategy,
                                             client_resources={"num_cpus": 3, "num_gpus": 1.0},
                                             )
    
    ## 7. Save your results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / 'results.pkl'

    results = {'history':history}
    with open(str(results_path), 'wb') as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
