import os
import anndata
import torch
from SpaDOT.utils import _train_utils, _utils

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch") # this is for suppressing UserWarning from torch.tensor
 
def train(args):
    # --- load data ---
    print("Loading data...")
    data_dir = os.path.abspath(args.data)
    if not args.output_dir:
        args.output_dir = os.path.dirname(data_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    adata = anndata.read_h5ad(data_dir)
    model_config = _utils.load_model_config(args)
    # add adata related parameters to model_config
    model_config['input_dim'] = adata.n_vars
    tps = adata.obs['timepoint'].unique()
    tps.sort()
    model_config['timepoints'] = tps
    # add device and dtype to model_config
    model_config['device'] = torch.device(args.device)
    model_config['dtype'] = torch.float64

    _utils.set_seed(model_config['seed'])
    # obtain inducing points, graph adjacency matrix, and dataloaders
    print("Preparing data...")
    dataloader_dict = _train_utils.prepare_dataloader(adata, model_config)
    _utils._save_inducing_points(args, dataloader_dict['inducing_points']) # save inducing points

    # train model
    print("Training model...")
    SpaDOT_model, loss_df = _train_utils.train_SpaDOT(dataloader_dict, model_config)
    loss_df.T.to_csv(args.output_dir + os.sep + 'loss.csv')
    if args.save_model:
        torch.save(SpaDOT_model.state_dict(), args.output_dir+os.sep+'SpaDOT_model.pth')
        print("Model saved to %s" % (args.output_dir))
    # obtain the latent representation
    latent_adata = _train_utils.get_latent(SpaDOT_model, model_config, adata, dataloader_dict)
    latent_adata.write_h5ad(args.output_dir+os.sep+args.prefix+'latent.h5ad')

if __name__ == "__main__":
    # data_dir = "./examples"
    data_dir = "/net/mulan/home/wenjinma/projects/SpaDOT/examples_Python_version"
    # create arguments for testing
    class Args:
        data = os.path.join(data_dir, "preprocessed_ChickenHeart.h5ad")
        prefix = ""
        config = None
        save_model = True
        device = 'cuda:0'
    args = Args()
    print(args)
    # create output directory if not exists
    if 'output_dir' not in args.__dict__:
        args.output_dir = os.path.dirname(args.data)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # train SpaDOT model
    train(args)