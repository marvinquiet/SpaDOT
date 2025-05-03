import pandas as pd
import scanpy as sc                                                                                                                                                                    
import anndata

# --- D4 data
D4_adata = sc.read_10x_h5('GSM4502482_chicken_heart_spatial_RNAseq_D4_filtered_feature_bc_matrix.h5')
D4_adata.obs['barcode'] = D4_adata.obs_names
D4_adata.obs_names = 'D4-A1_'+D4_adata.obs['barcode']
D4_coords = pd.read_csv('chicken_heart_spatial_RNAseq_D4_tissue_positions_list.csv', header=None, index_col=0)
D4_coords.columns = ['in_out', 'x', 'y', 'pixel_x', 'pixel_y']
D4_coords.index = 'D4-A1_'+D4_coords.index
D4_adata.obs = pd.merge(D4_adata.obs, D4_coords, how='left', left_index=True, right_index=True)
D4_adata.obs['orig.ident'] = 'D4'
# --- D7 data
D7_adata = sc.read_10x_h5('GSM4502483_chicken_heart_spatial_RNAseq_D7_filtered_feature_bc_matrix.h5')
D7_adata.obs['barcode'] = D7_adata.obs_names
D7_adata.obs_names = 'D7-B1_'+D7_adata.obs['barcode']
D7_pos = pd.read_csv('chicken_heart_spatial_RNAseq_D7_tissue_positions_list.csv', header=None, index_col=0)
D7_pos.columns = ['in_out', 'x', 'y', 'pixel_x', 'pixel_y']
D7_pos.index = 'D7-B1_'+D7_pos.index
D7_adata.obs = pd.merge(D7_adata.obs, D7_pos, how='left', left_index=True, right_index=True)
D7_adata.obs['orig.ident'] = 'D7'
# --- D10 data
D10_adata = sc.read_10x_h5('GSM4502484_chicken_heart_spatial_RNAseq_D10_filtered_feature_bc_matrix.h5')
D10_adata.obs['barcode'] = D10_adata.obs_names
D10_adata.obs_names = 'D10-C1_'+D10_adata.obs['barcode']
D10_pos = pd.read_csv('chicken_heart_spatial_RNAseq_D10_tissue_positions_list.csv', header=None, index_col=0)
D10_pos.columns = ['in_out', 'x', 'y', 'pixel_x', 'pixel_y']
D10_pos.index = 'D10-C1_'+D10_pos.index
D10_adata.obs = pd.merge(D10_adata.obs, D10_pos, how='left', left_index=True, right_index=True)
D10_adata.obs['orig.ident'] = 'D10'
# --- D14 data
D14_adata = sc.read_10x_h5('GSM4502485_chicken_heart_spatial_RNAseq_D14_filtered_feature_bc_matrix.h5')
D14_adata.obs['barcode'] = D14_adata.obs_names
D14_adata.obs_names = 'D14-D1_'+D14_adata.obs['barcode']
D14_pos = pd.read_csv('chicken_heart_spatial_RNAseq_D14_tissue_positions_list.csv', header=None, index_col=0)
D14_pos.columns = ['in_out', 'x', 'y', 'pixel_x', 'pixel_y']
D14_pos.index = 'D14-D1_'+D14_pos.index
D14_adata.obs = pd.merge(D14_adata.obs, D14_pos, how='left', left_index=True, right_index=True)
D14_adata.obs['orig.ident'] = 'D14'

# merge all data
# --- remove all duplicated genes
D4_adata = D4_adata[:, ~D4_adata.var_names.duplicated()]
D7_adata = D7_adata[:, ~D7_adata.var_names.duplicated()]
D10_adata = D10_adata[:, ~D10_adata.var_names.duplicated()]
D14_adata = D14_adata[:, ~D14_adata.var_names.duplicated()]
adata_list = [D4_adata, D7_adata, D10_adata, D14_adata]
adata = anndata.concat(adata_list, axis=0)
adata.var['gene'] = adata.var_names
# --- add timepoints correspondence
timepoint_dict = {'D4': 0, 'D7': 1, 'D10': 2, 'D14': 3}
adata.obs['timepoint'] = [timepoint_dict[_] for _ in adata.obs['orig.ident']]
adata.X = adata.X.astype(int)
adata.obs.index.name = None
# --- edit spatial coordinates to rotate the image, but keep timepoint 0 unchanged
adata.obs['pixel_x_bak'] = adata.obs['pixel_x']
adata.obs['pixel_y_bak'] = adata.obs['pixel_y']
adata.obs['pixel_x'] = adata.obs.apply(
    lambda row: row['pixel_y_bak'] if row['timepoint'] in [1, 2, 3] else -row['pixel_y_bak'], axis=1
)
adata.obs['pixel_y'] = adata.obs.apply(
    lambda row: row['pixel_x_bak'] if row['timepoint'] in [1, 2, 3] else -row['pixel_x_bak'], axis=1
)
adata.obsm['spatial'] = adata.obs[['pixel_x', 'pixel_y']].values
adata.write_h5ad('ChickenHeart.h5ad')

# draw the spatial coordinates to confirm
import matplotlib.pyplot as plt
def plot_spatial_coordinates_per_timepoint(adata):
    """
    Plot spatial coordinates for each timepoint to verify rotation.
    """
    timepoints = adata.obs['timepoint'].unique()
    for tp in timepoints:
        tp_data = adata[adata.obs['timepoint'] == tp]
        plt.figure(figsize=(6, 6))
        plt.scatter(tp_data.obs['pixel_x'], tp_data.obs['pixel_y'], s=1, alpha=0.7)
        plt.title(f"Timepoint {tp}")
        plt.xlabel("Pixel X")
        plt.ylabel("Pixel Y")
        plt.savefig(f"spatial_coordinates_{tp}.png")

# Call the function to plot
plot_spatial_coordinates_per_timepoint(adata)
