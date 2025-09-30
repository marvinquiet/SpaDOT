import os
import anndata
from SpaDOT.utils import _analyze_utils # for debugging

def analyze(args):
    data_dir = os.path.abspath(args.data)
    if not args.output_dir:
        args.output_dir = os.path.dirname(data_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.n_clusters is None and args.prefix == "":
        args.prefix = 'adaptive_'

    latent = anndata.read_h5ad(data_dir)
    if args.n_clusters is None:
        latent = _analyze_utils.Adaptive_clustering(args, latent)
    else:
        latent = _analyze_utils.KMeans_Clustering(latent, args.n_clusters)
    latent.obs['pixel_x'] = latent.obsm['spatial'][:, 0]
    latent.obs['pixel_y'] = latent.obsm['spatial'][:, 1]
    # draw domains
    _analyze_utils.plot_domains(args, latent)
    # perform optimal transport analysis
    _analyze_utils.OT_analysis(args, latent)
    # plot OT results
    _analyze_utils.plot_OT(args, latent)


if __name__ == "__main__":
    # data_dir = "./examples"
    data_dir = "/net/mulan/home/wenjinma/projects/SpaDOT/examples_Python_version"
    # create arguments for testing
    class Args:
        data = os.path.join(data_dir, "latent.h5ad")
        prefix = ""
        n_clusters = [5, 7, 7, 6]
        # n_clusters = None
    args = Args()
    # create output directory if not exists
    if 'output_dir' not in args.__dict__:
        args.output_dir = os.path.dirname(args.data)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # analyze latent representations
    analyze(args)