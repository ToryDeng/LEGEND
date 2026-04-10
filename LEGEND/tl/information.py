# -*- coding: utf-8 -*-
# @Time : 2023/1/6 2:19
# @Author : Tory Deng
# @File : information.py
# @Software: PyCharm
import os
from itertools import combinations
from multiprocessing.pool import Pool
from typing import Tuple

import anndata as ad
import igraph as ig
import numpy as np
from loguru import logger
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing._data import scale

expr_mtx, clusters, seed = None, None, 0


def add_small_noise(X: np.ndarray, random_state: int):
    X = scale(X, with_mean=False, copy=False)
    means = np.maximum(1, np.mean(np.abs(X), axis=0))
    rng = np.random.default_rng(random_state)
    X += (1e-10 * means * rng.standard_normal(size=X.shape))
    return X


def _compute_relevance(i: int):
    global expr_mtx, clusters, seed
    g1 = expr_mtx[:, i].reshape(-1, 1)
    # return _compute_mi(g1, clusters, x_discrete=False, y_discrete=True)
    return mutual_info_classif(g1, clusters, discrete_features=False, random_state=seed)[0]


def find_relevant_genes(adata: ad.AnnData, top_pct: int, max_workers: int = os.cpu_count() - 1, random_state: int = 0):
    """
    Compute relevance of each gene to the pseudo cell types (stored in `adata.var['relevance']`),
    and then inplace subsets genes with the highest relevance as relevant genes.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix of shape (n_obs, n_vars).
        Rows correspond to cells and columns to genes.
    top_pct : int
        The percentage of relevant genes. This parameter should be between 0 and 100.
    max_workers : int
        The maximum value of workers which can be used during feature selection.
    random_state : int
        Change to use different initial states for the optimization.
    """
    logger.info("Finding relevant genes...")
    if top_pct != 100:
        global expr_mtx, clusters, seed
        expr_mtx, clusters, seed = adata.X.copy(), adata.obs['cluster'], random_state
        with Pool(processes=max_workers) as pool:
            relevance = np.array(pool.map(_compute_relevance, range(adata.n_vars)))
        logger.debug("Gene relevance computed!")
        rlv_th = np.percentile(relevance, 100 - top_pct)
        logger.opt(colors=True).debug(f"Relevance threshold: <yellow>{rlv_th}</yellow>")
        is_relevant = relevance >= rlv_th
    else:
        relevance = np.ones(shape=(adata.n_vars,))
        is_relevant = relevance.copy().astype(bool)
    adata._inplace_subset_var(is_relevant)
    adata.var['relevance'] = relevance[is_relevant]
    logger.opt(colors=True).info(
        f"<yellow>{is_relevant.sum()}</yellow> (<yellow>{top_pct}%</yellow>) genes are marked as relevant genes."
    )


def _compute_redundancy(gene_index_pair: Tuple[int, int]):
    global expr_mtx, clusters, seed
    g1, g2 = expr_mtx[gene_index_pair[0], :].reshape(-1, 1), expr_mtx[gene_index_pair[1], :]
    # return _compute_mi(g1, g2, x_discrete=False, y_discrete=False)
    return mutual_info_regression(g1, g2, discrete_features=False, random_state=seed)[0]


def compute_gene_redundancy(adata: ad.AnnData, max_workers: int = os.cpu_count() - 1, random_state: int = 0):
    """
    Compute redundancy of all relevant gene pairs and store them as a symmetric matrix in `adata.varp['redundancy']`.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix of shape (n_obs, n_vars).
        Rows correspond to cells and columns to genes.
    max_workers : int
        The maximum value of workers which can be used during feature selection.
    random_state : int
        Change to use different initial states for the optimization.
    """
    logger.info("Computing gene redundancy...")
    global expr_mtx, clusters, seed
    expr_mtx, clusters, seed = adata.varm['X_pca'].copy(), None, random_state
    with Pool(processes=max_workers) as pool:  # upper triangular matrix
        adata.varp['redundancy'] = squareform(pool.map(_compute_redundancy, combinations(range(adata.n_vars), 2)))
    logger.info(f"Gene redundancy computed.")


def _compute_complementarity(gene_index_pair: Tuple[int, int]):
    global expr_mtx, clusters, seed
    i, j = gene_index_pair[0], gene_index_pair[1]

    cmi = 0
    for clus in np.unique(clusters):
        clus_mask = clusters == clus
        g1, g2 = expr_mtx[clus_mask, i].reshape(-1, 1), expr_mtx[clus_mask, j]
        # rlv = _compute_mi(g1, g2, x_discrete=False, y_discrete=False)
        rlv = mutual_info_regression(g1, g2, discrete_features=False, n_neighbors=3, random_state=seed)[0]
        cmi += rlv * clus_mask.sum() / expr_mtx.shape[0]
    return cmi


def compute_gene_complementarity(adata: ad.AnnData, max_workers: int = os.cpu_count() - 1, random_state: int = 0):
    """
    Compute complementarity of any two genes that are connected in the MST.
    Store the pairwise gene complementarity in `adata.uns['mst_edges_complm']` as an ndarray of shape (n_edges, ).
    Store the edges of MST in `adata.uns['mst_edges']` as an ndarray of shape (n_edges, 2).

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix of shape (n_obs, n_vars).
        Rows correspond to cells and columns to genes.
    max_workers : int
        The maximum value of workers which can be used during feature selection.
    random_state : int
        Change to use different initial states for the optimization.
    """
    logger.info("Computing gene complementarity...")
    global expr_mtx, clusters, seed
    expr_mtx, clusters, seed = adata.X.copy(), adata.obs['cluster'], random_state
    # compute complementarity in MST
    with Pool(processes=max_workers) as pool:
        edge_complm = np.array(pool.map(_compute_complementarity, adata.uns['MST'].get_edgelist()))
    logger.info(f"Gene complementarity computed.")
    return edge_complm


def build_MST(adjacency: np.ndarray):
    logger.debug("Building MST...")
    G = ig.Graph.Weighted_Adjacency(adjacency, mode="undirected", attr='neg_redundancy')
    MST = G.spanning_tree(weights=G.es["neg_redundancy"])
    logger.debug("MST built.")
    return MST

