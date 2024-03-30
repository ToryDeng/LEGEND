import numpy as np
from scipy.stats import nbinom
from scipy.stats import nbinom
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import scanpy as sc
import SpaGCN as spg
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix

import STAGATE

def get_data(sample_id):
        assert sample_id == '151676', "please choose the 151676 for testing"
        path_file = f"F:/151676_10xvisium.h5ad"
        adata =  sc.read_h5ad(path_file)
        return adata

def size_factor_normalization(adata):
        adata_X_dense = adata.X.todense()
        adata_X_dense = np.log(adata_X_dense)
        adata_X_dense[~np.isfinite(adata_X_dense)] = 0
        col_means = np.mean(adata_X_dense, axis=0)
        adata_X_dense = adata_X_dense - col_means
        size_factor = np.exp(np.median(adata_X_dense, axis=0))
        adata_X_dense = adata.X.todense() / size_factor #The original `adata.X` should be used as the dividend
        adata.X = csr_matrix(adata_X_dense)
        return adata

def data_process(adata):
        adata = adata[~adata.obs['layer_guess_reordered_short'].isna()]
        adata = adata[~adata.obs['discard'].astype(bool), :]
        adata.obs['cluster']  = adata.obs['layer_guess_reordered_short']
        adata.var_names_make_unique()
        spg.prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
        spg.prefilter_specialgenes(adata)
        adata = size_factor_normalization(adata)
        all_genes = adata.var.index.values
        adata.obs['array_x']=np.ceil((adata.obs['array_col']-
                                  adata.obs['array_col'].min())/2).astype(int)
        adata.obs['array_y']=adata.obs['array_row']-adata.obs['array_row'].min()
        all_gene_exp_matrices = {}
        shape = (adata.obs['array_y'].max()+1, adata.obs['array_x'].max()+1)
        return adata

#1Extract the mu corresponding to top t%, along with the respective var and indices.
def extract_top_t_percent(mu, var,t):
    mu_rank = np.argsort(mu)[::-1] #Obtain the indices after sorting in descending order.
    top_t_percent_count = int(t *0.01* len(mu))
    top_mu = mu[mu_rank[:top_t_percent_count]]
    top_var = var[mu_rank[:top_t_percent_count]]
    top_mu_indices = mu_rank[:top_t_percent_count]
    return top_mu,top_var,top_mu_indices

#2Extract the mean values (mu) and estimate the corresponding s.
def estimate_s(top_mu, top_var,top_mu_indices):
    #2.1 Perform a quadratic regression with the independent variable as mu and the dependent variable as var.
    poly = PolynomialFeatures(degree=2)
    mu_poly = poly.fit_transform(top_mu.reshape(-1, 1))
    model = LinearRegression().fit(mu_poly, top_var)
    print(model.coef_)
    alpha2=model.coef_[2]
    print("alpha2={}".format(alpha2))
    random_mu_indices = np.random.choice(top_mu_indices) # Randomly sample mu corresponding to the selected indices.
    random_mu=top_mu[np.where(top_mu_indices == random_mu_indices)[0]]
    print(random_mu)
    s=1/alpha2
    print(s)
    p = s/(random_mu+s)
    return random_mu_indices,random_mu,s,p


#Data preprocessing
adata = get_data('151676')
adata=data_process(adata)
print(adata.X)

adata.obs['Ground Truth']  = adata.obs['layer_guess_reordered_short']

adata_X_L1 = adata[adata.obs['Ground Truth'] == 'L1'].X
adata_X_L2 = adata[adata.obs['Ground Truth'] == 'L2'].X
adata_X_L3 = adata[adata.obs['Ground Truth'] == 'L3'].X
adata_X_L4 = adata[adata.obs['Ground Truth'] == 'L4'].X
adata_X_L5 = adata[adata.obs['Ground Truth'] == 'L5'].X
adata_X_L6 = adata[adata.obs['Ground Truth'] == 'L6'].X
adata_X_WM = adata[adata.obs['Ground Truth'] == 'WM'].X

#Layer1 t=65
dense_matrix = adata_X_L1.todense()
mu = np.mean(dense_matrix, axis=1)
mu = np.asarray(mu).flatten()
var = np.var(dense_matrix, axis=1)
var = np.asarray(var).flatten()
assert mu.shape == var.shape
t=65
#simulation
top_mu, top_var,top_mu_indices=extract_top_t_percent(mu, var,t)
random_mu_indices,random_mu,s,p=estimate_s(top_mu,top_var,top_mu_indices)
simulated_data = nbinom.rvs(s,p=p, size=adata_X_L1.shape[0]) #The size is determined by the number of spots in that area.
print("Simulated negative binomial distribution:", simulated_data)
#assignment
sorted_simulated_data = np.sort(simulated_data)
col_indices_plot_gene = np.argsort(mu)
rearranged_simulated_data = sorted_simulated_data.copy()
for i, col_index in enumerate(col_indices_plot_gene):
    rearranged_simulated_data[col_index] = sorted_simulated_data[i]
gene_list = adata.var_names
adata[adata.obs['Ground Truth'] == 'L1',gene_list[0]].X =rearranged_simulated_data

#Layer2,t=65
dense_matrix = adata_X_L2.todense()
mu = np.mean(dense_matrix, axis=1)
mu = np.asarray(mu).flatten()
var = np.var(dense_matrix, axis=1)
var = np.asarray(var).flatten()
assert mu.shape == var.shape
t=65
#simulation
top_mu, top_var,top_mu_indices=extract_top_t_percent(mu, var,t)
random_mu_indices,random_mu,s,p=estimate_s(top_mu,top_var,top_mu_indices)
simulated_data = nbinom.rvs(s,p=p, size=adata_X_L2.shape[0])
print("Simulated negative binomial distribution:", simulated_data)
#assignment
sorted_simulated_data = np.sort(simulated_data)
col_indices_plot_gene = np.argsort(mu)
rearranged_simulated_data = sorted_simulated_data.copy()
for i, col_index in enumerate(col_indices_plot_gene):
    rearranged_simulated_data[col_index] = sorted_simulated_data[i]
gene_list = adata.var_names
adata[adata.obs['Ground Truth'] == 'L2',gene_list[0]].X =rearranged_simulated_data

#Layer3,t=25
dense_matrix = adata_X_L3.todense()
mu = np.mean(dense_matrix, axis=1)
mu = np.asarray(mu).flatten()
var = np.var(dense_matrix, axis=1)
var = np.asarray(var).flatten()
assert mu.shape == var.shape
t=25
#simulation
top_mu, top_var,top_mu_indices=extract_top_t_percent(mu, var,t)
random_mu_indices,random_mu,s,p=estimate_s(top_mu,top_var,top_mu_indices)
simulated_data = nbinom.rvs(s,p=p, size=adata_X_L3.shape[0])
print("Simulated negative binomial distribution:", simulated_data)
#assignment
sorted_simulated_data = np.sort(simulated_data)
col_indices_plot_gene = np.argsort(mu)
rearranged_simulated_data = sorted_simulated_data.copy()
for i, col_index in enumerate(col_indices_plot_gene):
    rearranged_simulated_data[col_index] = sorted_simulated_data[i]
gene_list = adata.var_names
adata[adata.obs['Ground Truth'] == 'L3',gene_list[0]].X =rearranged_simulated_data

#Layer4,t=65
dense_matrix = adata_X_L4.todense()
mu = np.mean(dense_matrix, axis=1)
mu = np.asarray(mu).flatten()
var = np.var(dense_matrix, axis=1)
var = np.asarray(var).flatten()
assert mu.shape == var.shape
t=65
#simulation
top_mu, top_var,top_mu_indices=extract_top_t_percent(mu, var,t)
random_mu_indices,random_mu,s,p=estimate_s(top_mu,top_var,top_mu_indices)
simulated_data = nbinom.rvs(s,p=p, size=adata_X_L4.shape[0])
print("Simulated negative binomial distribution:", simulated_data)
#assignment
sorted_simulated_data = np.sort(simulated_data)
col_indices_plot_gene = np.argsort(mu)
rearranged_simulated_data = sorted_simulated_data.copy()
for i, col_index in enumerate(col_indices_plot_gene):
    rearranged_simulated_data[col_index] = sorted_simulated_data[i]
gene_list = adata.var_names
adata[adata.obs['Ground Truth'] == 'L4',gene_list[0]].X =rearranged_simulated_data

#Layer5,t=65
dense_matrix = adata_X_L5.todense()
mu = np.mean(dense_matrix, axis=1)
mu = np.asarray(mu).flatten()
var = np.var(dense_matrix, axis=1)
var = np.asarray(var).flatten()
assert mu.shape == var.shape
t=65
#simulation
top_mu, top_var,top_mu_indices=extract_top_t_percent(mu, var,t)
random_mu_indices,random_mu,s,p=estimate_s(top_mu,top_var,top_mu_indices)
simulated_data = nbinom.rvs(s,p=p, size=adata_X_L5.shape[0])
print("Simulated negative binomial distribution:", simulated_data)
#assignment
sorted_simulated_data = np.sort(simulated_data)
col_indices_plot_gene = np.argsort(mu)
rearranged_simulated_data = sorted_simulated_data.copy()
for i, col_index in enumerate(col_indices_plot_gene):
    rearranged_simulated_data[col_index] = sorted_simulated_data[i]
gene_list = adata.var_names
adata[adata.obs['Ground Truth'] == 'L5',gene_list[0]].X =rearranged_simulated_data

#Layer6,t=65
dense_matrix = adata_X_L6.todense()
mu = np.mean(dense_matrix, axis=1)
mu = np.asarray(mu).flatten()
var = np.var(dense_matrix, axis=1)
var = np.asarray(var).flatten()
assert mu.shape == var.shape
t=65
#simulation
top_mu, top_var,top_mu_indices=extract_top_t_percent(mu, var,t)
random_mu_indices,random_mu,s,p=estimate_s(top_mu,top_var,top_mu_indices)
simulated_data = nbinom.rvs(s,p=p, size=adata_X_L6.shape[0])
print("Simulated negative binomial distribution:", simulated_data)
#assignment
sorted_simulated_data = np.sort(simulated_data)
col_indices_plot_gene = np.argsort(mu)
rearranged_simulated_data = sorted_simulated_data.copy()
for i, col_index in enumerate(col_indices_plot_gene):
    rearranged_simulated_data[col_index] = sorted_simulated_data[i]
gene_list = adata.var_names
adata[adata.obs['Ground Truth'] == 'L6',gene_list[0]].X =rearranged_simulated_data

#WM,t=5
dense_matrix = adata_X_WM.todense()
mu = np.mean(dense_matrix, axis=1)
mu = np.asarray(mu).flatten()
var = np.var(dense_matrix, axis=1)
var = np.asarray(var).flatten()
assert mu.shape == var.shape
t=5
#simulation
top_mu, top_var,top_mu_indices=extract_top_t_percent(mu, var,t)
random_mu_indices,random_mu,s,p=estimate_s(top_mu,top_var,top_mu_indices)
simulated_data = nbinom.rvs(s,p=p, size=adata_X_WM.shape[0])
print("Simulated negative binomial distribution:", simulated_data)
#assignment
sorted_simulated_data = np.sort(simulated_data)
col_indices_plot_gene = np.argsort(mu)
rearranged_simulated_data = sorted_simulated_data.copy()
for i, col_index in enumerate(col_indices_plot_gene):
    rearranged_simulated_data[col_index] = sorted_simulated_data[i]
gene_list = adata.var_names
adata[adata.obs['Ground Truth'] == 'WM',gene_list[0]].X =rearranged_simulated_data

###denoise
STAGATE.Cal_Spatial_Net(adata, rad_cutoff=150)
STAGATE.Stats_Spatial_Net(adata)
adata = STAGATE.train_STAGATE(adata, alpha=0, save_reconstrction=True)
#image before denoising
sc.pl.spatial(adata, img_key="hires", color= gene_list[0],title=None,show=False, vmax='p99')
ax = plt.gca()
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('')
plt.savefig(f"real_.png", bbox_inches='tight')
plt.show()
#The comparison of images before and after denoising, from left to right, includes the following: simulated data, simulated data denoised using STAGATE, and a real gene image after denoising.
plot_gene = gene_list[5001]
fig, axs = plt.subplots(1, 3, figsize=(10, 3), gridspec_kw={'width_ratios': [1, 1, 1,]})
sc.pl.spatial(adata, img_key="hires", color=gene_list[0], show=False, ax=axs[0], title='simulated_result', vmax='p99')
sc.pl.spatial(adata, img_key="hires", color=gene_list[0], show=False, ax=axs[1], title='STAGATE_simulated_result', layer='STAGATE_ReX', vmax='p99')
sc.pl.spatial(adata, img_key="hires", color=plot_gene, show=False, ax=axs[2], title='RAW_'+plot_gene, vmax='p99',layer='STAGATE_ReX')
for ax in axs:
    ax.title.set_size(9)
plt.subplots_adjust(wspace=0.5)
plt.show()

