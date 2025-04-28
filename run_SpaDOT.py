import warnings
warnings.filterwarnings("ignore")

import os, argparse
import json
import logging
import random
import scanpy as sc
import numpy as np

import torch
from torch.backends import cudnn

# --- load my package

from SpaDOT import SpaDOT
import utils as utils

# import train_utils
# import data_utils
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
seed = 1993
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
cudnn.deterministic = True
cudnn.benchmark = False



if __name__ == '__main__':
    args = parse_args()
    simulation_data_dir = data_dir+os.sep+str(args.seed)+os.sep+str(args.DEG_x)+'_'+str(args.DEG_y)
    result_dir = result_dir+os.sep+str(args.seed)+os.sep+str(args.DEG_x)+'_'+str(args.DEG_y)
    os.makedirs(result_dir, exist_ok=True)
    # load data
    adata = sc.read_h5ad(simulation_data_dir+os.sep+'adata.h5ad')
    random.seed(1234)
    sampled_idx = random.sample(range(adata.shape[0]), 3000*3)
    adata = adata[sampled_idx].copy()
    adata.obs.rename(columns={'x_coor': 'pixel_x', 'y_coor': 'pixel_y', 'label': 'annotation'}, inplace=True)
    adata.obs['timepoint_numeric'] = adata.obs['timepoint'].astype('category').cat.codes
    adata.obsm['spatial'] = adata.obs[['pixel_x', 'pixel_y']]

    # spaDOT analysis
    timepoints = list(set(adata.obs['timepoint']))
    timepoints.sort()
    # load spaDOT args 
    spaDOT_args = train_utils.parse_spaDOT_args()
    spaDOT_args.timepoints = timepoints
    spaDOT_args.kernel_scale = [0.1] * len(timepoints) # set kernel scale as 0.1
    spaDOT_args.result_dir = result_dir
    with open(spaDOT_args.result_dir+os.sep+'args.json', 'w') as f:
        json.dump(spaDOT_args.__dict__, f, indent=4)
    spaDOT_args.device = device

    # --- preprocess data
    adata = data_utils.preprocess_adata(spaDOT_args, adata, get_SVG=False)
    initial_inducing_points_dict, kernel_scale_dict, N_train_dict, dataloaders_dict, adj_dict, dataset_dict = train_utils.prepare_dataloader(adata, spaDOT_args)

    def trainable(config):
        config_result_dir = result_dir+os.sep+f"lambda1_{config['lambda1']}_lambda2_{config['lambda2']}_beta1_{config['beta1']}_beta2_{config['beta2']}_omiga1_{config['omiga1']}_omiga2_{config['omiga2']}_omiga3_{config['omiga3']}"
        os.makedirs(config_result_dir, exist_ok=True)
        spaDOT_args.result_dir = config_result_dir

        # --- set up latent records
        spaDOT_args.latent_record = True
        latent_records_dir = spaDOT_args.result_dir+os.sep+'latent_records'
        os.makedirs(latent_records_dir, exist_ok=True)
        spaDOT_args.latent_records_dir = latent_records_dir

        # set parameters
        spaDOT_args.lambda1, spaDOT_args.lambda2 = config['lambda1'], config['lambda2']
        spaDOT_args.beta1, spaDOT_args.beta2 = config['beta1'], config['beta2']
        spaDOT_args.omiga1, spaDOT_args.omiga2, spaDOT_args.omiga3 = config['omiga1'], config['omiga2'], config['omiga3']
        # --- set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler(spaDOT_args.result_dir+os.sep+'training.log')],
                            force=True)
        print(logging.getLogger().handlers)
        logging.info(spaDOT_args.__dict__)

        start_time = time.time()
        spaDOT_args.latent_record = False # do not evaluate latent space
        # --- initialize model
        model = spaDOT(input_dim=adata.n_vars, 
                    SVGP_z_dim=int(spaDOT_args.z_dim/2), GNN_z_dim=int(spaDOT_args.z_dim/2),
                    encoder_layers=spaDOT_args.encoder_layers, decoder_layers=spaDOT_args.decoder_layers,
                    fixed_inducing_points=spaDOT_args.fix_inducing_points, initial_inducing_points=initial_inducing_points_dict, 
                    fixed_gp_params=spaDOT_args.fixed_gp_params, kernel_type=spaDOT_args.kernel_type,
                    kernel_scale=kernel_scale_dict, N_train_dict=N_train_dict, 
                    dtype=torch.float64, device=spaDOT_args.device)
        logging.info(str(model))
        model.dataloaders_dict = dataloaders_dict
        model.adj_dict = adj_dict
        model.dataset_dict = dataset_dict
        
        # --- train function
        model = train_utils.train_spaDOT(model, spaDOT_args, logging, adata)
        end_time = time.time()
        with open(spaDOT_args.result_dir+os.sep+'spaDOT_running_time.txt', 'w') as f:
            f.write(str(end_time-start_time))
            f.write('\n')
        # --- evaluation
        # latent_adata = train_utils.do_eval(model, spaDOT_args, logging, adata,
        #             prefix='final_', plot=True)
        # latent_adata.write_h5ad(spaDOT_args.result_dir+os.sep+'final_latent.h5ad')
        # mean_ARI = train_utils.test_spaDOT(spaDOT_args, latent_adata)
        # logging.info("config: ", config)
        # logging.info(f"mean ARI: {mean_ARI}")  

    
    config = {'lambda1': lambda1, 'lambda2': lambda2,
              'beta1': beta1, 'beta2': beta2,
        'omiga1': omiga1, 'omiga2': omiga2, 'omiga3': omiga3}
    trainable(config)

