import os
import anndata
from utils import _analyze_utils

def analyze(args):
    latent = anndata.read_h5ad(args.data)
    if args.n_clusters is None:
        latent = _analyze_utils.Adaptive_Clustering(latent)
    else:
        latent = _analyze_utils.KMeans_Clustering(latent, args.n_clusters)
    latent.obs['pixel_x'] = latent.obsm['spatial'][:, 0]
    latent.obs['pixel_y'] = latent.obsm['spatial'][:, 1]
    # draw domains
    _analyze_utils.plot_domains(args, latent)

    # perform optimal transport analysis
    # _analyze_utils.OT_analysis(args, latent)
    # plot OT results
    # _analyze_utils.plot_OT(args)



if __name__ == "__main__":
    data_dir = "./examples"
    # create arguments for testing
    class Args:
        data = os.path.join(data_dir, "latent.h5ad")
        prefix = ""
        n_clusters = [5, 7, 7, 6]
    args = Args()
    # create output directory if not exists
    if 'output_dir' not in args.__dict__:
        args.output_dir = os.path.dirname(args.data)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # analyze latent representations
    analyze(args)