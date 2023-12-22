# -*- coding: utf-8 -*-
# @Time : 2022/6/3 16:27
# @Author : Tory Deng
# @File : _model.py
# @Software: PyCharm
import os
from typing import Literal, Optional, Tuple, Union

import anndata as ad
import numpy as np
from loguru import logger
from scipy.sparse import issparse

import LEGEND.pp as pp
import LEGEND.tl as tl

from ._utils import set_logger
from ._validation import check_all_genes_selected, check_args


def GeneClust(
    adata: ad.AnnData,
    image: np.ndarray = None,
    n_var_clusters: int = None,
    n_obs_clusters: int = None,
    n_components: int = 10,
    relevant_gene_pct: int = 20,
    post_hoc_filtering: bool = True,
    version: Literal["fast", "ps"] = "fast",
    modality: Literal["sc", "st"] = "sc",
    shape: Literal["hexagon", "square"] = "hexagon",
    alpha: float = 0.3,
    return_info: bool = False,
    subset: bool = False,
    max_workers: int = os.cpu_count() - 1,
    log_path: Optional[Union[os.PathLike, str]] = None,
    verbosity: Literal[0, 1, 2] = 1,
    random_state: int = 0,
) -> Optional[Union[Tuple[ad.AnnData, np.ndarray], np.ndarray]]:
    """
    This function is the common interface for *GeneClust-fast* and *GeneClust-ps*.

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
        We reconmmend to pass raw counts which are stored in `adata.X`. However, if `adata.X` doesn't contain raw counts,
        GeneClust will assume the data in `adata.X` have been normalized and directly use them.
    image : ndarray
        The image of tissue section.
    n_var_clusters : int
        The number of clusters in gene clustering. Only valid in GeneClust-fast.
    n_obs_clusters : int
        The number of clusters in cell/spots clustering used to find high-confidence cells/spots. Only valid in GeneClust-ps.
    n_components : int, default=10
        The number of principal components used along with the first component. Only valid in GeneClust-ps.
    relevant_gene_pct: int, default=20
        The percentage of relevant genes. This parameter should be between 0 and 100. Only valid in GeneClust-ps.
    post_hoc_filtering : bool, default=True
        Whether to find outliers in singleton gene clusters (in GeneClust-fast) or low-density genes (in GeneClust-ps)
        after gene clustering.
    version : Literal['fast', 'ps'], default='fast'
        Choose the version of GeneClust.
    modality : Literal['sc', 'st'], default='sc'
        Type of the dataset. 'sc' for scRNA-seq data, 'st' for spatially resolved transcriptomics (SRT) data.
    shape : Literal['hexagon', 'square'], default='hexagon'
        The shape of spot neighbors. 'hexagon' for Visium data, 'square' for ST data.
    alpha: float, default=0.3
        Spots of which the distances to spatial cluster centers <= alpha * radius are chosen as highly confident spots.
    return_info: bool, default=False
        If `False`, only return names of selected genes.
        Otherwise, return an `AnnData` object which contains intermediate results generated during feature selection.
    subset: bool, default=False
        If `True`, inplace subset to selected genes otherwise merely return the names of selected genes
        (and intermediate results recorded in an `AnnData` object, depending on the value of `return_info`).
    max_workers : int, default=os.cpu_count() - 1
        The maximum value of workers which can be used during feature selection. Default is the number of CPUs - 1.
    log_path: Union[os.PathLike, str], default=None
        Path to the log file
    verbosity : Literal[0, 1, 2], default=1
        The verbosity level.
        If 0, only prints warnings and errors.
        If 1, prints info-level messages, warnings and errors.
        If 2, prints debug-level messages, info-level messages, warnings and errors.
    random_state : int, default=0
        Change to use different initial states for the optimization.

    Returns
    -------
    Depending on `subset` and `return_info`, returns names of selected genes (and intermediate results),
    or inplace subsets to selected genes and returns `None`.

    copied_adata : AnnData
        Stores intermediate results generated during feature selection.
        The normalized counts are stored in `copied_adata.layers['pearson_norm']`.
        The cell-level principal components are stored in `copied_adata.varm['X_pca']`.
        The gene cluster labels are in `copied_adata.var['cluster']`.
        For GeneClust-fast, the closeness of genes to their cluster centers are in `copied_adata.var['closeness']`.
        For GeneClust-ps, the gene-level principal components are in `copied_adata.obsm['X_pca']`.
        The high-confidence cell cluster labels are in `copied_adata.obs['cluster']`.
        Low-confidence cell clusters are filtered.
        Genes relevance values are in `copied_adata.var['relevance']`. Irrelevant genes are filtered.
        Gene redundancy values are in `copied_adata.varp['redundancy']`.
        MST of relevant genes is in `copied_adata.uns['MST']`.
        Gene outlier scores are in `copied_adata.var['outlier_score']`.
        Representative genes are indicated by `copied_adata.var['representative']`.
    selected_genes : ndarray
        Names of selected genes.

    Examples
    -------
    >>> from LEGEND import GeneClust, load_PBMC3k
    >>>
    >>>
    >>> adata = load_PBMC3k()
    >>> selected_genes_fast = GeneClust(adata, version='fast', n_var_clusters=200)
    >>> selected_genes_ps = GeneClust(adata, version='ps', n_obs_clusters=7)
    """
    # set log level and log path
    set_logger(verbosity, log_path)

    # check arguments
    do_norm = check_args(
        adata,
        image,
        version,
        n_var_clusters,
        n_obs_clusters,
        n_components,
        relevant_gene_pct,
        post_hoc_filtering,
        modality,
        shape,
        alpha,
        return_info,
        subset,
        max_workers,
        random_state,
    )

    # feature selection starts
    logger.opt(colors=True).info(
        f"Performing <magenta>GeneClust-{version}</magenta> "
        f"on <magenta>{'scRNA-seq' if modality == 'sc' else 'SRT'}</magenta> data "
        f"with <yellow>{max_workers}</yellow> workers."
    )
    copied_adata = adata.copy()
    copied_adata.X = adata.X.toarray() if issparse(adata.X) else adata.X

    # preprocessing
    if do_norm:
        pp.normalize(copied_adata, modality)
    pp.reduce_dim(copied_adata, version, random_state)
    # gene clustering
    tl.cluster_genes(
        copied_adata,
        image,
        version,
        modality,
        shape,
        alpha,
        n_var_clusters,
        n_obs_clusters,
        n_components,
        relevant_gene_pct,
        max_workers,
        random_state,
    )
    # select features from gene clusters
    selected_genes = tl.select_from_clusters(
        copied_adata, version, modality, 20, post_hoc_filtering, random_state
    )
    check_all_genes_selected(copied_adata, selected_genes)

    if subset:
        adata._inplace_subset_var(selected_genes)
        logger.opt(colors=True).info(
            f"<magenta>GeneClust-{version}</magenta> finished."
        )
        return None

    logger.opt(colors=True).info(f"<magenta>GeneClust-{version}</magenta> finished.")
    if return_info:
        return copied_adata, selected_genes
    else:
        return selected_genes


def integrate(
    adata_rna: ad.AnnData,
    adata_st: ad.AnnData,
    rna_weight: float = 0.5,
    rel_pct: int = 20,
    post_hoc_filtering: bool = True,
    return_info: bool = False,
    max_workers: int = os.cpu_count() - 1,
    log_path: Optional[Union[os.PathLike, str]] = None,
    verbosity: Literal[0, 1, 2] = 1,
    random_state: int = 0,
):
    """
    Integrate information from multimodal data to identify co-expressed genes.

    Parameters
    ----------
    adata_rna : AnnData
        Stores intermediate results generated during feature selection on sc/snRNA-seq data.
    adata_st : AnnData
        Stores intermediate results generated during feature selection on SRT data.
    rna_weight : float
        Weight of sc/snRNA-seq information.
    rel_pct : float
        Percent of relevant genes which should be selected from each gene cluster.
    post_hoc_filtering : bool, default=True
        Whether to find outliers in singleton gene clusters after gene clustering.
    return_info: bool, default=False
        If `False`, only return names of selected genes.
        Otherwise, return an `AnnData` object which contains intermediate results generated during co-expressed genes detection.
    max_workers : int, default=os.cpu_count() - 1
        The maximum value of workers which can be used during feature selection. Default is the number of CPUs - 1.
    log_path: Union[os.PathLike, str], default=None
        Path to the log file
    verbosity : Literal[0, 1, 2], default=1
        The verbosity level.
        If 0, only prints warnings and errors.
        If 1, prints info-level messages, warnings and errors.
        If 2, prints debug-level messages, info-level messages, warnings and errors.
    random_state : int, default=0
        Change to use different initial states for the optimization.

    Returns
    -------
    If `return_info=True`, returns names of selected genes and intermediate results.
    Otherwise only returns an AnnData object which stores intermediate results generated during co-expressed genes detection.
    """
    # set log level and log path
    set_logger(verbosity, log_path)

    common_genes = np.intersect1d(adata_rna.var_names, adata_st.var_names)
    logger.opt(colors=True).info(
        f"Detected <yellow>{common_genes.shape[0]}</yellow> genes shared by "
        f"<magenta>SRT</magenta> and <magenta>scRNA-seq</magenta>."
    )

    adata_rna = adata_rna[:, common_genes].copy()
    adata_st = adata_st[:, common_genes].copy()

    pseudo_adata = ad.AnnData(np.zeros((1, common_genes.shape[0])), dtype=float)
    pseudo_adata.var_names = common_genes

    comb_redundancy = (
        rna_weight * adata_rna.varp["redundancy"]
        + (1 - rna_weight) * adata_st.varp["redundancy"]
    )
    comb_relevance = (
        rna_weight * adata_rna.var["relevance"]
        + (1 - rna_weight) * adata_st.var["relevance"]
    )
    comb_MST = tl.information.build_MST(-comb_redundancy)
    adata_st.uns["MST"], adata_rna.uns["MST"] = comb_MST, comb_MST
    logger.opt(colors=True).info(
        f"Start to compute complementarity on <magenta>SRT</magenta> data..."
    )
    st_complm = tl.information.compute_gene_complementarity(
        adata_st, max_workers, random_state
    )
    logger.opt(colors=True).info(
        f"Start to compute complementarity on <magenta>scRNA-seq</magenta> data..."
    )
    rna_complm = tl.information.compute_gene_complementarity(
        adata_rna, max_workers, random_state
    )
    comb_MST.es["complm"] = rna_weight * st_complm + (1 - rna_weight) * rna_complm

    pseudo_adata.uns["MST"] = comb_MST
    pseudo_adata.var["relevance"] = comb_relevance
    pseudo_adata.var["relevance_rna"] = adata_rna.var["relevance"]
    pseudo_adata.var["relevance_st"] = adata_st.var["relevance"]

    tl.cluster.generate_gene_clusters(pseudo_adata)
    selected_genes = tl.select_from_clusters(
        pseudo_adata, "ps", "st", rel_pct, post_hoc_filtering, random_state
    )
    check_all_genes_selected(pseudo_adata, selected_genes)

    if return_info:
        return pseudo_adata, selected_genes
    else:
        return selected_genes
