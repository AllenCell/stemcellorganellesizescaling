#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
import datetime
from scipy.stats import gaussian_kde
from tqdm import tqdm
from matplotlib import cm
import os, platform

# Third party
# from stemcellorganellesizescaling.analyses.data_prep import outlier_removal, initial_parsing
# from stemcellorganellesizescaling.analyses.compute_stats import compensate
# importlib.reload(sys.modules["stemcellorganellesizescaling.analyses.compute_stats"])
# from stemcellorganellesizescaling.analyses.compute_stats import compensate, pairwisestats
# Relative

print("Libraries loaded succesfully")
###############################################################################

log = logging.getLogger(__name__)

###############################################################################

#%% Directories
if platform.system() == "Windows":
    data_root = Path("E:/DA/Data/scoss/Data/")
    pic_root = Path("E:/DA/Data/scoss/Pics/")
elif platform.system() == "Linux":
    data_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Data/")
    pic_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Pics/")
dirs = []
dirs.append(data_root)
dirs.append(pic_root)

# %% Start

# def outlier_removal(
#     dirs: list, dataset: Path, dataset_clean: Path,
# ):
"""
Removes outliers, generates diagnostic plots, saves cleaned feature data table

Parameters
----------
dirs: list
    Lists data and plotting dir
dataset: Path
    Path to CSV file with cell by feature data table
dataset_clean: Path
    Path to cleaned CSV file
"""

# Resolve directories
data_root = dirs[0]
pic_root = dirs[1]

dataset = "SizeScaling_20201006.csv"
dataset_clean = "SizeScaling_20201006_clean.csv"

# Load dataset
cells = pd.read_csv(data_root / dataset)

# Remove outliers
# %% Parameters, updated directories
save_flag = 1  # save plot (1) or show on screen (0)
pic_root = pic_root / "outlier_removal"
pic_root.mkdir(exist_ok=True)
data_root_extra = data_root / "outlier_removal"
data_root_extra.mkdir(exist_ok=True)

# %% Yep

# Load dataset
# cells = pd.read_csv(data_root / dataset)

####### Remove outliers ########

# %% Remove cells that lack a Structure Volume value
cells_ao = cells[['CellId', 'structure_name']].copy()
cells_ao['Outlier annotation'] = 'Keep'
print(cells.shape)
CellIds_remove = cells.loc[cells["Structure volume"].isnull(), 'CellId'].squeeze().to_numpy()
cells_ao.loc[cells_ao['CellId'].isin(CellIds_remove),'Outlier annotation'] = 'Missing structure volume'
cells = cells.drop(cells.index[cells['CellId'].isin(CellIds_remove)])
print(f'Removing {len(CellIds_remove)} cells that lack a Structure Volume measurement value')
print(cells.shape)

# %%
print("FIX LINE BELOW")
# cells["Piece std"] = cells["Piece std"].replace(np.nan, 0)

# %% Feature set for cell and nuclear features
cellnuc_metrics = ['Cell surface area', 'Cell volume', 'Cell height',
                     'Nuclear surface area', 'Nuclear volume', 'Nucleus height',
                     'Cytoplasmic volume']
cellnuc_abbs = ['Cell area', 'Cell vol', 'Cell height', 'Nuc area', 'Nuc vol', 'Nuc height', 'Cyto vol']
struct_metrics =   ['Structure volume']

# %% All metrics including height
L = len(cellnuc_metrics)
pairs = np.zeros((int(L*(L-1)/2),2)).astype(np.int)
i = 0
for f1 in np.arange(L):
    for f2 in np.arange(L):
        if f2>f1:
            pairs[i,:] = [f1, f2]
            i += 1

# # %% The typical six scatter plots
# xvec = [1, 1, 6, 1, 4, 6]
# yvec = [4, 6, 4, 0, 3, 3]
# pairs = np.stack((xvec, yvec)).T
#
# # %% Just one
# xvec = [1]
# yvec = [4]
# pairs = np.stack((xvec, yvec)).T


# %% Parameters
nbins = 100
N = 10000
fac = 1000
Rounds = 5

# %% For all pairs compute densities
# dens_th = 1e-40
remove_cells = cells['CellId'].to_frame().copy()
for i, xy_pair in tqdm(enumerate(pairs),'Enumerate pairs of metrics'):

    metricX = cellnuc_metrics[xy_pair[0]]
    metricY = cellnuc_metrics[xy_pair[1]]
    print(f"{metricX} vs {metricY}")

    # data
    x = cells[metricX].to_numpy()/fac
    y = cells[metricY].to_numpy()/fac
    # x = cells[metricX].sample(1000,random_state = 1117).to_numpy() / fac
    # y = cells[metricY].sample(1000,random_state = 1117).to_numpy() / fac

    # sampling on x and y
    xii, yii = np.mgrid[
        x.min() : x.max() : nbins * 1j, y.min() : y.max() : nbins * 1j
    ]
    xi = xii[:, 0]

    # density estimate, repeat because of probabilistic nature of density estimate used here
    for r in np.arange(Rounds):
        remove_cells[f"{metricX} vs {metricY}_{r}"] = np.nan
        print(f"Round {r+1} of {Rounds}")
        rs = int(r)
        xS, yS = resample(
            x, y, replace=False, n_samples=np.amin([N, len(x)]), random_state=rs
        )
        k = gaussian_kde(np.vstack([xS, yS]))
        zii = k(np.vstack([xii.flatten(), yii.flatten()]))
        cell_dens = k(np.vstack([x.flatten(), y.flatten()]))
        cell_dens = cell_dens / np.sum(cell_dens)
        remove_cells.loc[remove_cells.index[np.arange(len(cell_dens))], f"{metricX} vs {metricY}_{r}"] = cell_dens

remove_cells.to_csv(data_root_extra / 'cell_nucleus.csv')

# # %% Summarize across repeats
# remove_cells_summary = cells['CellId'].to_frame().copy()
# for i, xy_pair in enumerate(pairs):
#
#     metricX = cellnuc_metrics[xy_pair[0]]
#     metricY = cellnuc_metrics[xy_pair[1]]
#     filter_col = [col for col in remove_cells if col.startswith(f"{metricX} vs {metricY}")]
#     x = remove_cells[filter_col].to_numpy()
#     pos = np.argwhere(np.any(x < 1e-20, axis=1))
#     y = x[pos,:].squeeze()
#
#     fig, axs = plt.subplots(1, 2, figsize=(10, 8))
#     xr = np.log(x.flatten())
#     xr = np.delete(xr,np.argwhere(np.isinf(xr)))
#     axs[0].hist(xr,bins=100)
#     axs[0].set_title(f"Histogram of cell probabilities (log scale)")
#     axs[0].set_yscale('log')
#     im = axs[1].imshow(np.log(y),aspect='auto')
#     plt.colorbar(im)
#     axs[1].set_title(f"Heatmap with low probability cells (log scale)")
#
#     if save_flag:
#         plot_save_path = pic_root / f"{metricX} vs {metricY}_cellswithlowprobs.png"
#         plt.savefig(plot_save_path, format="png", dpi=1000)
#         plt.close()
#     else:
#         plt.show()
#
#     remove_cells_summary[f"{metricX} vs {metricY}"] = np.median(x,axis=1)
#
# # %% Identify cells to be removed
# cell_dens_th_array = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-15, 1e-20, 1e-30, 1e-40]
# filenames = []
# for j, cell_dens_th in enumerate(cell_dens_th_array):
#     CellIds_remove_dict = {}
#     CellIds_remove = np.empty(0,dtype=int)
#     for i, xy_pair in enumerate(pairs):
#         metricX = cellnuc_metrics[xy_pair[0]]
#         metricY = cellnuc_metrics[xy_pair[1]]
#         CellIds_remove = np.union1d(CellIds_remove, np.argwhere(remove_cells_summary[f"{metricX} vs {metricY}"].to_numpy() < cell_dens_th))
#         len(CellIds_remove)
#         CellIds_remove_dict[f"{metricX} vs {metricY}"] = CellIds_remove
#
#     oplot(cellnuc_metrics, cellnuc_abbs, pairs, cells, True, pic_root, f"OutlierFun_{j}", 2, CellIds_remove_dict)
#     filenames.append(pic_root / f"OutlierFun_{j}")
#
# # %%
# import imageio
# images = []
# for filename in filenames:
#     print(f"{filename}.png")
#     images.append(imageio.imread(f"{filename}.png"))
# imageio.mimsave(pic_root / 'Outlier.gif', images)
#
# # %%
# oplot(cellnuc_metrics, cellnuc_abbs, pairs, cells, False, pic_root, f"rainbow", 2, remove_cells_summary)
#
# #%% Plot and remove outliers
# plotname = 'CellNucleus'
# oplot(cellnuc_metrics, cellnuc_abbs, pairs, cells, True, pic_root, f"{plotname}_org_fine", 0.5, [])
# oplot(cellnuc_metrics, cellnuc_abbs, pairs, cells, True, pic_root, f"{plotname}_org_thick", 2, [])
# oplot(cellnuc_metrics, cellnuc_abbs, pairs, cells, True, pic_root, f"{plotname}_outliers", 2, CellIds_remove_dict)
# print(cells.shape)
# CellIds_remove_total = np.empty(0,dtype=int)
# for i, xy_pair in enumerate(pairs):
#     metricX = cellnuc_metrics[xy_pair[0]]
#     metricY = cellnuc_metrics[xy_pair[1]]
#     CellIds_remove_total = np.union1d(CellIds_remove_total, CellIds_remove_dict[f"{metricX}_{metricY}"])
# cells_ao.loc[cells_ao['CellId'].isin(CellIds_remove_total), 'Outlier annotation'] = 'Abnormal cell or nuclear metric'
# cells = cells.drop(cells.index[cells['CellId'].isin(CellIds_remove)])
# print('Removing cells due to abnormal cell or nuclear metric')
# print(cells.shape)
# oplot(cellnuc_metrics, cellnuc_abbs, pairs, cells, True, pic_root, f"{plotname}_clean_thick", 2, [])
# oplot(cellnuc_metrics, cellnuc_abbs, pairs, cells, True, pic_root, f"{plotname}_clean_fine", 2, [])
#
# # %% Feature set for structures
# # %% Select metrics
# selected_metrics = [
#     "Cell volume",
#     "Cell surface area",
#     "Nuclear volume",
#     "Nuclear surface area",
# ]
# selected_metrics_abb = ["Cell Vol", "Cell Area", "Nuc Vol", "Nuc Area"]
# selected_structures = [
#     "LMNB1",
#     "ST6GAL1",
#     "TOMM20",
#     "SEC61B",
#     "ATP2A2",
#     "LAMP1",
#     "RAB5A",
#     "SLC25A17",
#     "TUBA1B",
#     "TJP1",
#     "NUP153",
#     "FBL",
#     "NPM1",
#     "SON",
# ]
# selected_structures_org = [
#     "Nuclear envelope",
#     "Golgi",
#     "Mitochondria",
#     "ER",
#     "ER",
#     "Lysosome",
#     "Endosomes",
#     "Peroxisomes",
#     "Microtubules",
#     "Tight junctions",
#     "NPC",
#     "Nucleolus F",
#     "Nucleolus G",
#     "SON",
# ]
# selected_structures_cat = [
#     "Major organelle",
#     "Major organelle",
#     "Major organelle",
#     "Major organelle",
#     "Major organelle",
#     "Somes",
#     "Somes",
#     "Somes",
#     "Cytoplasmic structure",
#     "Cell-to-cell contact",
#     "Nuclear",
#     "Nuclear",
#     "Nuclear",
#     "Nuclear",
# ]
# structure_metric = "Structure volume"
#
# # %% Parameters
# nbins = 100
# N = 1000
# fac = 1000
#
# # %% For all pairs compute densities and identify cells to be removed
# dens_th = 1e-15
# remove_cells = {}
# for xm, metric in tqdm(enumerate(selected_metrics), "Iterating metrics"):
#     for ys, struct in tqdm(enumerate(selected_structures), "and structures"):
#
#         # data
#         x = cells.loc[cells["structure_name"] == struct, [metric]].squeeze().to_numpy()/fac
#         y = cells.loc[
#             cells["structure_name"] == struct, [structure_metric]
#         ].squeeze().to_numpy()/fac
#
#         # density estimate
#         remove_cells[f"{metric}_{struct}"] = np.empty(0, dtype=np.int)
#         lrc = -1
#         while lrc != len(remove_cells[f"{metric}_{struct}"]):
#             rs = int(datetime.datetime.utcnow().timestamp())
#             xS, yS = resample(
#                 x, y, replace=False, n_samples=np.amin([N, len(x)]), random_state=rs
#             )
#             k = gaussian_kde(np.vstack([xS, yS]))
#             zii = k(np.vstack([xii.flatten(), yii.flatten()]))
#             cell_dens = k(np.vstack([x.flatten(), y.flatten()]))
#             cell_dens = cell_dens / np.sum(cell_dens)
#             lrc = len(remove_cells[f"{metric}_{struct}"])
#             remove_cells[f"{metric}_{struct}"] = np.union1d(remove_cells[f"{metric}_{struct}"],
#                                                               np.argwhere(cell_dens < dens_th))
#             print(len(remove_cells[f"{metric}_{struct}"]))
#
# # %%
#
# CellIds_remove = np.empty(0,dtype=int)
# for xm, metric in tqdm(enumerate(selected_metrics), "Iterating metrics"):
#     for ys, struct in tqdm(enumerate(selected_structures), "and structures"):
#         CellIds_remove = np.union1d(CellIds_remove, remove_cells[f"{metric}_{struct}"])
#
# # %%
#
#
#
#
#
#
#
#
#
#
#
#
# for i, xy_pair in enumerate(pairs):
#
#     metricX = cellnuc_metrics[xy_pair[0]]
#     metricY = cellnuc_metrics[xy_pair[1]]
#     print(f"{metricX} vs {metricY}")
#
#     # data
#     x = cells[metricX].to_numpy()/fac
#     y = cells[metricY].to_numpy()/fac
#
#     # sampling on x and y
#     xii, yii = np.mgrid[
#         x.min() : x.max() : nbins * 1j, y.min() : y.max() : nbins * 1j
#     ]
#     xi = xii[:, 0]
#
#     # density estimate
#     for round in np.arange(Rounds):
#         rs = int(datetime.datetime.utcnow().timestamp())
#         xS, yS = resample(
#             x, y, replace=False, n_samples=np.amin([N, len(x)]), random_state=rs
#         )
#         # xS, yS = resample(x, y, replace=False, n_samples=len(x), random_state=rs)
#         k = gaussian_kde(np.vstack([xS, yS]))
#         zii = k(np.vstack([xii.flatten(), yii.flatten()]))
#         cell_dens = k(np.vstack([x.flatten(), y.flatten()]))
#         cell_dens = cell_dens / np.sum(cell_dens)
#         remove_cells = np.union1d(remove_cells, np.argwhere(cell_dens < dens_th))
#         print(len(remove_cells))
#
#
# Q = {}  # 'Structure volume'
# # structure_metric = 'Number of pieces'
#
# for xm, metric in tqdm(enumerate(selected_metrics), "Iterating metrics"):
#     for ys, struct in tqdm(enumerate(selected_structures), "and structures"):
#
#         # data
#         x = cells.loc[cells["structure_name"] == struct, [metric]].squeeze()
#         y = cells.loc[
#             cells["structure_name"] == struct, [structure_metric]
#         ].squeeze()
#         x = x.to_numpy()
#         y = y.to_numpy()
#         x = x / fac
#         y = y / fac
#
#         # density estimate
#         for round in np.arange(Rounds):
#             rs = int(datetime.datetime.utcnow().timestamp())
#             xS, yS = resample(
#                 x, y, replace=False, n_samples=np.amin([N, len(x)]), random_state=rs
#             )
#             k = gaussian_kde(np.vstack([xS, yS]))
#             zii = k(np.vstack([xii.flatten(), yii.flatten()]))
#             cell_dens = k(np.vstack([x.flatten(), y.flatten()]))
#             cell_dens = cell_dens / np.sum(cell_dens)
#             # make into cumulative sum
#             zii = zii / np.sum(zii)
#             ix = np.argsort(zii)
#             zii = zii[ix]
#             zii = np.cumsum(zii)
#             jx = np.argsort(ix)
#             zii = zii[jx]
#             zii = zii.reshape(xii.shape)
#             Q[f"{metric}_{struct}_dens_x_{round}"] = xii
#             Q[f"{metric}_{struct}_dens_y_{round}"] = yii
#             Q[f"{metric}_{struct}_dens_z_{round}"] = zii
#             Q[f"{metric}_{struct}_dens_c_{round}"] = cell_dens
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# #%%
#
# oplot(cellnuc_metrics, cellnuc_abbs, pairs, cells, False, pic_root, f"rainbow", 2, remove_cells_summary)
#
#
# #%%
# def oplot(cellnuc_metrics, cellnuc_abbs, pairs, cells, save_flag, pic_root, name, markersize, remove_cells):
#
#     #%% Selecting number of pairs
#     no_of_pairs, _ = pairs.shape
#     nrows = np.floor(np.sqrt(2 / 3 *no_of_pairs))
#     if nrows == 0:
#         nrows = 1
#     ncols = np.floor(nrows*3/2)
#     while nrows*ncols<no_of_pairs:
#         ncols += 1
#
#     #%% Plotting parameters
#     fac = 1000
#     ms = markersize
#     fs2 = np.round(np.interp(nrows*ncols,[6, 21, 50],[25, 12, 8]))
#     fs = np.round(fs2*2/3)
#     lw2 = 1.5
#     nbins = 100
#     plt.rcParams.update({"font.size": fs})
#
#     #%% Plotting flags
#     # W = 500
#
#     #%% Time for a flexible scatterplot
#     w1 = 0.001
#     w2 = 0.01
#     w3 = 0.001
#     h1 = 0.001
#     h2 = 0.01
#     h3 = 0.001
#     xp = 0.1
#     yp = 0.1
#     xx = (1-w1-((ncols-1)*w2)-w3)/ncols
#     yy = (1-h1-((nrows-1)*h2)-h3)/nrows
#     xw = xx*xp
#     xx = xx*(1-xp)
#     yw = yy*yp
#     yy = yy*(1-yp)
#
#     fig = plt.figure(figsize=(16, 9))
#
#     for i, xy_pair in enumerate(pairs):
#
#         print(i)
#
#         metricX = cellnuc_metrics[xy_pair[0]]
#         metricY = cellnuc_metrics[xy_pair[1]]
#         abbX = cellnuc_abbs[xy_pair[0]]
#         abbY = cellnuc_abbs[xy_pair[1]]
#
#         # data
#         x = cells[metricX].to_numpy()/fac
#         y = cells[metricY].to_numpy()/fac
#
#         # select subplot
#         row = nrows-np.ceil((i+1)/ncols)+1
#         row = row.astype(np.int64)
#         col = (i+1) % ncols
#         if col == 0: col = ncols
#         col = col.astype(np.int64)
#         print(f"{i}_{row}_{col}")
#
#         # Main scatterplot
#         ax = fig.add_axes([w1 + ((col - 1) * (xw + xx + w2)) + xw, h1 + ((row - 1) * (yw + yy + h2)) + yw, xx, yy])
#         # ax.plot(x, y, 'b.', markersize=ms)
#
#         # cr = remove_cells[f"{metricX} vs {metricY}"].astype(np.int)
#         # if len(cr)>0:
#         #     ax.plot(x[cr], y[cr], 'r.', markersize=2*ms)
#         cr = remove_cells[f"{metricX} vs {metricY}"].to_numpy()
#         cr = -np.log10(cr)
#         max_cr = 10
#         cr[np.argwhere(cr > max_cr)] = max_cr
#         cr[np.argwhere(np.isinf(cr))] = max_cr
#
#         ax.scatter(x, y, c=cr, s=ms,cmap=plt.cm.plasma)
#
#         xticks = ax.get_xticks()
#         yticks = ax.get_yticks()
#         xlim = ax.get_xlim()
#         ylim = ax.get_ylim()
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.grid()
#
#         ax.text(xlim[1], ylim[1], f"n= {len(x)}", fontsize=fs, verticalalignment='top', horizontalalignment='right', color=[.75, .75, .75, .75])
#         ax.set_facecolor('black')
#
#         # Bottom histogram
#         ax = fig.add_axes([w1 + ((col - 1) * (xw + xx + w2)) + xw, h1 + ((row - 1) * (yw + yy + h2)), xx, yw])
#         ax.hist(x, bins = nbins, color = [.5,.5,.5,.5])
#         ylimBH = ax.get_ylim()
#         ax.set_xticks(xticks)
#         ax.set_yticks([])
#         ax.set_yticklabels([])
#         ax.set_xticklabels([])
#         ax.set_xlim(left=xlim[0], right=xlim[1])
#         ax.grid()
#         ax.invert_yaxis()
#         for n, val in enumerate(xticks):
#             if val>=xlim[0] and val<=xlim[1]:
#                 if int(val)==val:
#                     val = int(val)
#                 else:
#                     val = np.round(val,2)
#                 ax.text(val, ylimBH[0], f"{val}", fontsize=fs, horizontalalignment='center', verticalalignment='bottom', color=[.75, .75, .75, .75])
#
#         ax.text(np.mean(xlim), ylimBH[1], f"{abbX}", fontsize=fs2, horizontalalignment='center', verticalalignment='bottom')
#         ax.axis('off')
#
#         # Side histogram
#         ax = fig.add_axes([w1 + ((col - 1) * (xw + xx + w2)), h1 + ((row - 1) * (yw + yy + h2))+yw, xw, yy])
#         ax.hist(y, bins=nbins, color=[.5,.5,.5,.5], orientation='horizontal')
#         xlimSH = ax.get_xlim()
#         ax.set_yticks(yticks)
#         ax.set_xticks([])
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_ylim(bottom=ylim[0], top=ylim[1])
#         ax.grid()
#         ax.invert_xaxis()
#         for n, val in enumerate(yticks):
#             if val >= ylim[0] and val <= ylim[1]:
#                 if int(val) == val:
#                     val = int(val)
#                 else:
#                     val = np.round(val,2)
#                 ax.text(xlimSH[0], val, f"{val}", fontsize=fs, horizontalalignment='left', verticalalignment='center', color=[.75, .75, .75, .75])
#
#         ax.text(xlimSH[1], np.mean(ylim), f"{abbY}", fontsize=fs2, horizontalalignment='left', verticalalignment='center',rotation=90)
#         ax.axis('off')
#
#     if save_flag:
#         plot_save_path = pic_root / f"{name}.png"
#         plt.savefig(plot_save_path, format="png", dpi=1000)
#         plt.close()
#     else:
#         plt.show()
#
# # %%
#
#
#
#
# # %%
#
#
#
#
# selected_metricsX = [
#     "Cell volume",
#     "Cell volume",
#     "Cytoplasmic volume",
#     "Cell volume",
#     "Nuclear volume",
#     "Cytoplasmic volume",
# ]
#
# selected_metricsX_abb = [
#     "Cell Vol",
#     "Cell Vol",
#     "Cyt vol",
#     "Cell Vol",
#     "Nuc Vol",
#     "Cyt Vol",
# ]
#
# selected_metricsY = [
#     "Nuclear volume",
#     "Cytoplasmic volume",
#     "Nuclear volume",
#     "Cell surface area",
#     "Nuclear surface area",
#     "Nuclear surface area",
# ]
#
# selected_metricsY_abb = [
#     "Nuc Vol",
#     "Cyt Vol",
#     "Nuc Vol",
#     "Cell Area",
#     "Nuc Area",
#     "Nuc Area",
# ]
#
# # %% Plotting parameters
# fac = 1000
# ms = 0.5
# ms2 = 3
# fs2 = 16
# lw2 = 1.5
# nbins = 100
# plt.rcParams.update({"font.size": 12})
#
# # %% Time for a flexible scatterplot
# nrows = 2
# ncols = 3
# w1 = 0.07
# w2 = 0.07
# w3 = 0.01
# h1 = 0.07
# h2 = 0.12
# h3 = 0.07
# xw = 0.03
# yw = 0.03
# xx = (1 - w1 - ((ncols - 1) * w2) - w3 - (ncols * xw)) / ncols
# yy = (1 - h1 - ((nrows - 1) * h2) - h3 - (nrows * yw)) / nrows
#
# fig = plt.figure(figsize=(16, 9))
#
# i = 0
# for xi, pack in enumerate(
#     zip(
#         selected_metricsX,
#         selected_metricsY,
#         selected_metricsX_abb,
#         selected_metricsY_abb,
#     )
# ):
#     metric1 = pack[0]
#     metric2 = pack[1]
#     label1 = pack[2]
#     label2 = pack[3]
#
#     # data
#     x = cells[metric1]
#     y = cells[metric2]
#     x = x / fac
#     y = y / fac
#
#     # select subplot
#     i = i + 1
#     row = nrows - np.ceil(i / ncols) + 1
#     row = row.astype(np.int64)
#     col = i % ncols
#     if col == 0:
#         col = ncols
#     print(f"{i}_{row}_{col}")
#
#     # Main scatterplot
#     ax = fig.add_axes(
#         [
#             w1 + ((col - 1) * (xw + xx + w2)) + xw,
#             h1 + ((row - 1) * (yw + yy + h2)) + yw,
#             xx,
#             yy,
#         ]
#     )
#     ax.plot(x, y, "b.", markersize=ms)
#     xticks = ax.get_xticks()
#     yticks = ax.get_yticks()
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.grid()
#     ax.set_title(f"{label1} vs {label2} (n= {len(x)})", fontsize=fs2)
#
#     # Bottom histogram
#     ax = fig.add_axes(
#         [
#             w1 + ((col - 1) * (xw + xx + w2)) + xw,
#             h1 + ((row - 1) * (yw + yy + h2)),
#             xx,
#             yw,
#         ]
#     )
#     ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 1])
#     ax.set_xticks(xticks)
#     ax.set_yticks([])
#     ax.set_yticklabels([])
#     ax.set_xlim(left=xlim[0], right=xlim[1])
#     ax.grid()
#     ax.set_xlabel(label1, fontsize=fs2)
#     ax.invert_yaxis()
#
#     # Side histogram
#     ax = fig.add_axes(
#         [
#             w1 + ((col - 1) * (xw + xx + w2)),
#             h1 + ((row - 1) * (yw + yy + h2)) + yw,
#             xw,
#             yy,
#         ]
#     )
#     ax.hist(y, bins=nbins, color=[0.5, 0.5, 0.5, 1], orientation="horizontal")
#     ax.set_yticks(yticks)
#     ax.set_xticks([])
#     ax.set_xticklabels([])
#     ax.set_ylim(bottom=ylim[0], top=ylim[1])
#     ax.grid()
#     ax.set_ylabel(label2, fontsize=fs2)
#     ax.invert_xaxis()
#
# if save_flag:
#     plot_save_path = pic_root / "CellNucleus_org_fine.png"
#     plt.savefig(plot_save_path, format="png", dpi=1000)
#     plt.close()
# else:
#     plt.show()
#
# # %% Time for a flexible scatterplot
# ms = 2
# nrows = 2
# ncols = 3
# w1 = 0.07
# w2 = 0.07
# w3 = 0.01
# h1 = 0.07
# h2 = 0.12
# h3 = 0.07
# xw = 0.03
# yw = 0.03
# xx = (1 - w1 - ((ncols - 1) * w2) - w3 - (ncols * xw)) / ncols
# yy = (1 - h1 - ((nrows - 1) * h2) - h3 - (nrows * yw)) / nrows
#
# fig = plt.figure(figsize=(16, 9))
#
# i = 0
# for xi, pack in enumerate(
#     zip(
#         selected_metricsX,
#         selected_metricsY,
#         selected_metricsX_abb,
#         selected_metricsY_abb,
#     )
# ):
#     metric1 = pack[0]
#     metric2 = pack[1]
#     label1 = pack[2]
#     label2 = pack[3]
#
#     # data
#     x = cells[metric1]
#     y = cells[metric2]
#     x = x / fac
#     y = y / fac
#
#     # select subplot
#     i = i + 1
#     row = nrows - np.ceil(i / ncols) + 1
#     row = row.astype(np.int64)
#     col = i % ncols
#     if col == 0:
#         col = ncols
#     print(f"{i}_{row}_{col}")
#
#     # Main scatterplot
#     ax = fig.add_axes(
#         [
#             w1 + ((col - 1) * (xw + xx + w2)) + xw,
#             h1 + ((row - 1) * (yw + yy + h2)) + yw,
#             xx,
#             yy,
#         ]
#     )
#     ax.plot(x, y, "b.", markersize=ms)
#     xticks = ax.get_xticks()
#     yticks = ax.get_yticks()
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.grid()
#     ax.set_title(f"{label1} vs {label2} (n= {len(x)})", fontsize=fs2)
#
#     # Bottom histogram
#     ax = fig.add_axes(
#         [
#             w1 + ((col - 1) * (xw + xx + w2)) + xw,
#             h1 + ((row - 1) * (yw + yy + h2)),
#             xx,
#             yw,
#         ]
#     )
#     ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 1])
#     ax.set_xticks(xticks)
#     ax.set_yticks([])
#     ax.set_yticklabels([])
#     ax.set_xlim(left=xlim[0], right=xlim[1])
#     ax.grid()
#     ax.set_xlabel(label1, fontsize=fs2)
#     ax.invert_yaxis()
#
#     # Side histogram
#     ax = fig.add_axes(
#         [
#             w1 + ((col - 1) * (xw + xx + w2)),
#             h1 + ((row - 1) * (yw + yy + h2)) + yw,
#             xw,
#             yy,
#         ]
#     )
#     ax.hist(y, bins=nbins, color=[0.5, 0.5, 0.5, 1], orientation="horizontal")
#     ax.set_yticks(yticks)
#     ax.set_xticks([])
#     ax.set_xticklabels([])
#     ax.set_ylim(bottom=ylim[0], top=ylim[1])
#     ax.grid()
#     ax.set_ylabel(label2, fontsize=fs2)
#     ax.invert_xaxis()
#
# if save_flag:
#     plot_save_path = pic_root / "CellNucleus_org_thick.png"
#     plt.savefig(plot_save_path, format="png", dpi=1000)
#     plt.close()
# else:
#     plt.show()
#
# # %% Parameters
# nbins = 100
# N = 5000
# Rounds = 5
#
# # %% Identify pairs, compute stuff and put into dicts
# selected_metrics = [
#     "Cell volume",
#     "Cell surface area",
#     "Nuclear volume",
#     "Nuclear surface area",
# ]
# Q = {}
#
# counter = 0
# for xi, pack in enumerate(
#     zip(
#         selected_metricsX,
#         selected_metricsY,
#         selected_metricsX_abb,
#         selected_metricsY_abb,
#     )
# ):
#     metric1 = pack[0]
#     metric2 = pack[1]
#     label1 = pack[2]
#     label2 = pack[3]
#
#     print(counter)
#     counter = counter + 1
#
#     # data
#     x = cells[metric1]
#     y = cells[metric2]
#     x = x.to_numpy()
#     y = y.to_numpy()
#     x = x / fac
#     y = y / fac
#
#     # sampling on x and y
#     xii, yii = np.mgrid[
#         x.min() : x.max() : nbins * 1j, y.min() : y.max() : nbins * 1j
#     ]
#     xi = xii[:, 0]
#
#     # density estimate
#     for round in np.arange(Rounds):
#         rs = int(datetime.datetime.utcnow().timestamp())
#         xS, yS = resample(
#             x, y, replace=False, n_samples=np.amin([N, len(x)]), random_state=rs
#         )
#         # xS, yS = resample(x, y, replace=False, n_samples=len(x), random_state=rs)
#         k = gaussian_kde(np.vstack([xS, yS]))
#         zii = k(np.vstack([xii.flatten(), yii.flatten()]))
#         cell_dens = k(np.vstack([x.flatten(), y.flatten()]))
#         cell_dens = cell_dens / np.sum(cell_dens)
#         # make into cumulative sum
#         zii = zii / np.sum(zii)
#         ix = np.argsort(zii)
#         zii = zii[ix]
#         zii = np.cumsum(zii)
#         jx = np.argsort(ix)
#         zii = zii[jx]
#         zii = zii.reshape(xii.shape)
#         Q[f"{metric1}_{metric2}_dens_x_{round}"] = xii
#         Q[f"{metric1}_{metric2}_dens_y_{round}"] = yii
#         Q[f"{metric1}_{metric2}_dens_z_{round}"] = zii
#         Q[f"{metric1}_{metric2}_dens_c_{round}"] = cell_dens
#
# # %%
# nrows = 2
# ncols = 3
# w1 = 0.07
# w2 = 0.07
# w3 = 0.01
# h1 = 0.07
# h2 = 0.12
# h3 = 0.07
# xw = 0.03
# yw = 0.03
# xx = (1 - w1 - ((ncols - 1) * w2) - w3 - (ncols * xw)) / ncols
# yy = (1 - h1 - ((nrows - 1) * h2) - h3 - (nrows * yw)) / nrows
#
# fig = plt.figure(figsize=(16, 9))
#
# dens_th = 1e-40
# remove_cells = []
#
# i = 0
# for xi, pack in enumerate(
#     zip(
#         selected_metricsX,
#         selected_metricsY,
#         selected_metricsX_abb,
#         selected_metricsY_abb,
#     )
# ):
#     metric1 = pack[0]
#     metric2 = pack[1]
#     label1 = pack[2]
#     label2 = pack[3]
#
#     # data
#     x = cells[metric1]
#     y = cells[metric2]
#     x = x / fac
#     y = y / fac
#     x = x.to_numpy()
#     y = y.to_numpy()
#
#     # select subplot
#     i = i + 1
#     row = nrows - np.ceil(i / ncols) + 1
#     row = row.astype(np.int64)
#     col = i % ncols
#     if col == 0:
#         col = ncols
#     print(f"{i}_{row}_{col}")
#
#     # Main scatterplot
#     ax = fig.add_axes(
#         [
#             w1 + ((col - 1) * (xw + xx + w2)) + xw,
#             h1 + ((row - 1) * (yw + yy + h2)) + yw,
#             xx,
#             yy,
#         ]
#     )
#     ax.plot(x, y, "b.", markersize=ms)
#     pos = []
#     for round in np.arange(Rounds):
#         cii = Q[f"{metric1}_{metric2}_dens_c_{round}"]
#         pos = np.union1d(pos, np.argwhere(cii < dens_th))
#         print(len(pos))
#     print(len(pos))
#     pos = pos.astype(int)
#     remove_cells = np.union1d(remove_cells, pos)
#     ax.plot(x[pos], y[pos], "r.", markersize=ms2)
#     xticks = ax.get_xticks()
#     yticks = ax.get_yticks()
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.grid()
#     ax.set_title(f"{label1} vs {label2} (n= {len(x)})", fontsize=fs2)
#
#     # Bottom histogram
#     ax = fig.add_axes(
#         [
#             w1 + ((col - 1) * (xw + xx + w2)) + xw,
#             h1 + ((row - 1) * (yw + yy + h2)),
#             xx,
#             yw,
#         ]
#     )
#     ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 1])
#     ax.set_xticks(xticks)
#     ax.set_yticks([])
#     ax.set_yticklabels([])
#     ax.set_xlim(left=xlim[0], right=xlim[1])
#     ax.grid()
#     ax.set_xlabel(label1, fontsize=fs2)
#     ax.invert_yaxis()
#
#     # Side histogram
#     ax = fig.add_axes(
#         [
#             w1 + ((col - 1) * (xw + xx + w2)),
#             h1 + ((row - 1) * (yw + yy + h2)) + yw,
#             xw,
#             yy,
#         ]
#     )
#     ax.hist(y, bins=nbins, color=[0.5, 0.5, 0.5, 1], orientation="horizontal")
#     ax.set_yticks(yticks)
#     ax.set_xticks([])
#     ax.set_xticklabels([])
#     ax.set_ylim(bottom=ylim[0], top=ylim[1])
#     ax.grid()
#     ax.set_ylabel(label2, fontsize=fs2)
#     ax.invert_xaxis()
#
# if save_flag:
#     plot_save_path = pic_root / "CellNucleus_outliers.png"
#     plt.savefig(plot_save_path, format="png", dpi=1000)
#     plt.close()
# else:
#     plt.show()
#
# print(len(remove_cells))
#
# # %%  Drop them
# cells = cells.drop(cells.index[remove_cells.astype(int)])
#
# # %% Time for a flexible scatterplot
# nrows = 2
# ncols = 3
# w1 = 0.07
# w2 = 0.07
# w3 = 0.01
# h1 = 0.07
# h2 = 0.12
# h3 = 0.07
# xw = 0.03
# yw = 0.03
# xx = (1 - w1 - ((ncols - 1) * w2) - w3 - (ncols * xw)) / ncols
# yy = (1 - h1 - ((nrows - 1) * h2) - h3 - (nrows * yw)) / nrows
#
# fig = plt.figure(figsize=(16, 9))
#
# i = 0
# for xi, pack in enumerate(
#     zip(
#         selected_metricsX,
#         selected_metricsY,
#         selected_metricsX_abb,
#         selected_metricsY_abb,
#     )
# ):
#     metric1 = pack[0]
#     metric2 = pack[1]
#     label1 = pack[2]
#     label2 = pack[3]
#
#     # data
#     x = cells[metric1]
#     y = cells[metric2]
#     x = x / fac
#     y = y / fac
#
#     # select subplot
#     i = i + 1
#     row = nrows - np.ceil(i / ncols) + 1
#     row = row.astype(np.int64)
#     col = i % ncols
#     if col == 0:
#         col = ncols
#     print(f"{i}_{row}_{col}")
#
#     # Main scatterplot
#     ax = fig.add_axes(
#         [
#             w1 + ((col - 1) * (xw + xx + w2)) + xw,
#             h1 + ((row - 1) * (yw + yy + h2)) + yw,
#             xx,
#             yy,
#         ]
#     )
#     ax.plot(x, y, "b.", markersize=ms)
#     xticks = ax.get_xticks()
#     yticks = ax.get_yticks()
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.grid()
#     ax.set_title(f"{label1} vs {label2} (n= {len(x)})", fontsize=fs2)
#
#     # Bottom histogram
#     ax = fig.add_axes(
#         [
#             w1 + ((col - 1) * (xw + xx + w2)) + xw,
#             h1 + ((row - 1) * (yw + yy + h2)),
#             xx,
#             yw,
#         ]
#     )
#     ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 1])
#     ax.set_xticks(xticks)
#     ax.set_yticks([])
#     ax.set_yticklabels([])
#     ax.set_xlim(left=xlim[0], right=xlim[1])
#     ax.grid()
#     ax.set_xlabel(label1, fontsize=fs2)
#     ax.invert_yaxis()
#
#     # Side histogram
#     ax = fig.add_axes(
#         [
#             w1 + ((col - 1) * (xw + xx + w2)),
#             h1 + ((row - 1) * (yw + yy + h2)) + yw,
#             xw,
#             yy,
#         ]
#     )
#     ax.hist(y, bins=nbins, color=[0.5, 0.5, 0.5, 1], orientation="horizontal")
#     ax.set_yticks(yticks)
#     ax.set_xticks([])
#     ax.set_xticklabels([])
#     ax.set_ylim(bottom=ylim[0], top=ylim[1])
#     ax.grid()
#     ax.set_ylabel(label2, fontsize=fs2)
#     ax.invert_xaxis()
#
# if save_flag:
#     plot_save_path = pic_root / "CellNucleus_clean_thick.png"
#     plt.savefig(plot_save_path, format="png", dpi=1000)
#     plt.close()
# else:
#     plt.show()
#
# # %% Time for a flexible scatterplot
# ms = 0.5
# nrows = 2
# ncols = 3
# w1 = 0.07
# w2 = 0.07
# w3 = 0.01
# h1 = 0.07
# h2 = 0.12
# h3 = 0.07
# xw = 0.03
# yw = 0.03
# xx = (1 - w1 - ((ncols - 1) * w2) - w3 - (ncols * xw)) / ncols
# yy = (1 - h1 - ((nrows - 1) * h2) - h3 - (nrows * yw)) / nrows
#
# fig = plt.figure(figsize=(16, 9))
#
# i = 0
# for xi, pack in enumerate(
#     zip(
#         selected_metricsX,
#         selected_metricsY,
#         selected_metricsX_abb,
#         selected_metricsY_abb,
#     )
# ):
#     metric1 = pack[0]
#     metric2 = pack[1]
#     label1 = pack[2]
#     label2 = pack[3]
#
#     # data
#     x = cells[metric1]
#     y = cells[metric2]
#     x = x / fac
#     y = y / fac
#
#     # select subplot
#     i = i + 1
#     row = nrows - np.ceil(i / ncols) + 1
#     row = row.astype(np.int64)
#     col = i % ncols
#     if col == 0:
#         col = ncols
#     print(f"{i}_{row}_{col}")
#
#     # Main scatterplot
#     ax = fig.add_axes(
#         [
#             w1 + ((col - 1) * (xw + xx + w2)) + xw,
#             h1 + ((row - 1) * (yw + yy + h2)) + yw,
#             xx,
#             yy,
#         ]
#     )
#     ax.plot(x, y, "b.", markersize=ms)
#     xticks = ax.get_xticks()
#     yticks = ax.get_yticks()
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.grid()
#     ax.set_title(f"{label1} vs {label2} (n= {len(x)})", fontsize=fs2)
#
#     # Bottom histogram
#     ax = fig.add_axes(
#         [
#             w1 + ((col - 1) * (xw + xx + w2)) + xw,
#             h1 + ((row - 1) * (yw + yy + h2)),
#             xx,
#             yw,
#         ]
#     )
#     ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 1])
#     ax.set_xticks(xticks)
#     ax.set_yticks([])
#     ax.set_yticklabels([])
#     ax.set_xlim(left=xlim[0], right=xlim[1])
#     ax.grid()
#     ax.set_xlabel(label1, fontsize=fs2)
#     ax.invert_yaxis()
#
#     # Side histogram
#     ax = fig.add_axes(
#         [
#             w1 + ((col - 1) * (xw + xx + w2)),
#             h1 + ((row - 1) * (yw + yy + h2)) + yw,
#             xw,
#             yy,
#         ]
#     )
#     ax.hist(y, bins=nbins, color=[0.5, 0.5, 0.5, 1], orientation="horizontal")
#     ax.set_yticks(yticks)
#     ax.set_xticks([])
#     ax.set_xticklabels([])
#     ax.set_ylim(bottom=ylim[0], top=ylim[1])
#     ax.grid()
#     ax.set_ylabel(label2, fontsize=fs2)
#     ax.invert_xaxis()
#
# if save_flag:
#     plot_save_path = pic_root / "CellNucleus_clean_fine.png"
#     plt.savefig(plot_save_path, format="png", dpi=1000)
#     plt.close()
# else:
#     plt.show()
#
# # %% Now to structures
#
# # %% Select metrics
# selected_metrics = [
#     "Cell volume",
#     "Cell surface area",
#     "Nuclear volume",
#     "Nuclear surface area",
# ]
# selected_metrics_abb = ["Cell Vol", "Cell Area", "Nuc Vol", "Nuc Area"]
# selected_structures = [
#     "LMNB1",
#     "ST6GAL1",
#     "TOMM20",
#     "SEC61B",
#     "LAMP1",
#     "RAB5A",
#     "SLC25A17",
#     "TUBA1B",
#     "TJP1",
#     "NUP153",
#     "FBL",
#     "NPM1",
# ]
# selected_structures_org = [
#     "Nuclear envelope",
#     "Golgi",
#     "Mitochondria",
#     "ER",
#     "Lysosome",
#     "Endosomes",
#     "Peroxisomes",
#     "Microtubules",
#     "Tight junctions",
#     "NPC",
#     "Nucleolus F",
#     "Nucleolus G",
# ]
# selected_structures_cat = [
#     "Major organelle",
#     "Major organelle",
#     "Major organelle",
#     "Major organelle",
#     "Somes",
#     "Somes",
#     "Somes",
#     "Cytoplasmic structure",
#     "Cell-to-cell contact",
#     "Nuclear",
#     "Nuclear",
#     "Nuclear",
# ]
# structure_metric = "Structure volume"
#
# # %% Plotting parameters
# fac = 1000
# ms = 0.5
# ms2 = 3
# ms3 = 10
# fs2 = 12
# fs3 = 17
# lw2 = 1.5
# lw3 = 3
# lw4 = 2.5
# nbins = 100
# plt.rcParams.update({"font.size": 5})
#
# categories = np.unique(selected_structures_cat)
# # colors = np.linspace(0, 1, len(categories))
# colors = cm.get_cmap("viridis", len(categories))
# colordict = dict(zip(categories, colors.colors))
#
# # %% Initial scatterplot
# nrows = len(selected_metrics)
# ncols = len(selected_structures)
# w1 = 0.027
# w2 = 0.01
# w3 = 0.002
# h1 = 0.07
# h2 = 0.08
# h3 = 0.07
# xw = 0
# yw = 0
# xx = (1 - w1 - ((ncols - 1) * w2) - w3 - (ncols * xw)) / ncols
# yy = (1 - h1 - ((nrows - 1) * h2) - h3 - (nrows * yw)) / nrows
#
# fig = plt.figure(figsize=(16, 9))
#
# i = 0
#
# for yi, metric in enumerate(selected_metrics):
#     for xi, struct in enumerate(selected_structures):
#
#         x = cells.loc[cells["structure_name"] == struct, [metric]].squeeze()
#         y = cells.loc[
#             cells["structure_name"] == struct, [structure_metric]
#         ].squeeze()
#         selcel = (cells["structure_name"] == struct).to_numpy()
#         struct_pos = np.argwhere(selcel)
#         x = x.to_numpy()
#         y = y.to_numpy()
#         x = x / fac
#         y = y / fac
#
#         # select subplot
#         i = i + 1
#         row = nrows - np.ceil(i / ncols) + 1
#         row = row.astype(np.int64)
#         col = i % ncols
#         if col == 0:
#             col = ncols
#         print(f"{i}_{row}_{col}")
#
#         # Main scatterplot
#         ax = fig.add_axes(
#             [
#                 w1 + ((col - 1) * (xw + xx + w2)) + xw,
#                 h1 + ((row - 1) * (yw + yy + h2)) + yw,
#                 xx,
#                 yy,
#             ]
#         )
#         ax.plot(x, y, "b.", markersize=ms)
#         xticks = ax.get_xticks()
#         yticks = ax.get_yticks()
#         xlim = ax.get_xlim()
#         ylim = ax.get_ylim()
#         ax.grid()
#         if xw == 0:
#             if yi == 0:
#                 plt.text(
#                     np.mean(xlim),
#                     ylim[0] + 1.3 * (ylim[1] - ylim[0]),
#                     struct,
#                     fontsize=fs2,
#                     horizontalalignment="center",
#                 )
#                 plt.text(
#                     np.mean(xlim),
#                     ylim[0] + 1.2 * (ylim[1] - ylim[0]),
#                     selected_structures_org[xi],
#                     fontsize=fs2,
#                     horizontalalignment="center",
#                 )
#                 plt.text(
#                     np.mean(xlim),
#                     ylim[0] + 1.1 * (ylim[1] - ylim[0]),
#                     f"n= {len(x)}",
#                     fontsize=fs2,
#                     horizontalalignment="center",
#                 )
#             if xi == 0:
#                 plt.figtext(
#                     0.5,
#                     h1 + ((row - 1) * (yw + yy + h2)) - h2 / 2,
#                     metric,
#                     fontsize=fs3,
#                     horizontalalignment="center",
#                 )
#         else:
#             # ax.set_title(f"{selected_structures[xi]} vs {selected_metrics_abb[yi]} (n= {len(x)})", fontsize=fs2)
#             ax.set_title(f"n= {len(x)}", fontsize=fs2)
#             ax.set_xticklabels([])
#             ax.set_yticklabels([])
#
#         ax.spines["bottom"].set_color(colordict[selected_structures_cat[xi]])
#         ax.spines["bottom"].set_linewidth(lw3)
#         ax.spines["top"].set_color(colordict[selected_structures_cat[xi]])
#         ax.spines["top"].set_linewidth(lw3)
#         ax.spines["right"].set_color(colordict[selected_structures_cat[xi]])
#         ax.spines["right"].set_linewidth(lw3)
#         ax.spines["left"].set_color(colordict[selected_structures_cat[xi]])
#         ax.spines["left"].set_linewidth(lw3)
#
#         if xw != 0:
#             # Bottom histogram
#             ax = fig.add_axes(
#                 [
#                     w1 + ((col - 1) * (xw + xx + w2)) + xw,
#                     h1 + ((row - 1) * (yw + yy + h2)),
#                     xx,
#                     yw,
#                 ]
#             )
#             ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 1])
#             ax.set_xticks(xticks)
#             ax.set_yticks([])
#             ax.set_yticklabels([])
#             ax.set_xlim(left=xlim[0], right=xlim[1])
#             ax.grid()
#             # if yi==len(selected_metrics_abb):
#             ax.set_xlabel(selected_metrics_abb[yi], fontsize=fs2)
#             ax.invert_yaxis()
#
#             # Side histogram
#             ax = fig.add_axes(
#                 [
#                     w1 + ((col - 1) * (xw + xx + w2)),
#                     h1 + ((row - 1) * (yw + yy + h2)) + yw,
#                     xw,
#                     yy,
#                 ]
#             )
#             ax.hist(
#                 y, bins=nbins, color=[0.5, 0.5, 0.5, 1], orientation="horizontal"
#             )
#             ax.set_yticks(yticks)
#             ax.set_xticks([])
#             ax.set_xticklabels([])
#             ax.set_ylim(bottom=ylim[0], top=ylim[1])
#             ax.grid()
#             # if xi==0:
#             ax.set_ylabel(selected_structures[xi], fontsize=fs2)
#             ax.invert_xaxis()
#
# if save_flag:
#     plot_save_path = pic_root / "Structures_org_fine.png"
#     plt.savefig(plot_save_path, format="png", dpi=1000)
#     plt.close()
# else:
#     plt.show()
#
# # %% thick
# ms = 2
# nrows = len(selected_metrics)
# ncols = len(selected_structures)
# w1 = 0.027
# w2 = 0.01
# w3 = 0.002
# h1 = 0.07
# h2 = 0.08
# h3 = 0.07
# xw = 0
# yw = 0
# xx = (1 - w1 - ((ncols - 1) * w2) - w3 - (ncols * xw)) / ncols
# yy = (1 - h1 - ((nrows - 1) * h2) - h3 - (nrows * yw)) / nrows
#
# fig = plt.figure(figsize=(16, 9))
#
# i = 0
#
# for yi, metric in enumerate(selected_metrics):
#     for xi, struct in enumerate(selected_structures):
#
#         x = cells.loc[cells["structure_name"] == struct, [metric]].squeeze()
#         y = cells.loc[
#             cells["structure_name"] == struct, [structure_metric]
#         ].squeeze()
#         selcel = (cells["structure_name"] == struct).to_numpy()
#         struct_pos = np.argwhere(selcel)
#         x = x.to_numpy()
#         y = y.to_numpy()
#         x = x / fac
#         y = y / fac
#
#         # select subplot
#         i = i + 1
#         row = nrows - np.ceil(i / ncols) + 1
#         row = row.astype(np.int64)
#         col = i % ncols
#         if col == 0:
#             col = ncols
#         print(f"{i}_{row}_{col}")
#
#         # Main scatterplot
#         ax = fig.add_axes(
#             [
#                 w1 + ((col - 1) * (xw + xx + w2)) + xw,
#                 h1 + ((row - 1) * (yw + yy + h2)) + yw,
#                 xx,
#                 yy,
#             ]
#         )
#         ax.plot(x, y, "b.", markersize=ms)
#         xticks = ax.get_xticks()
#         yticks = ax.get_yticks()
#         xlim = ax.get_xlim()
#         ylim = ax.get_ylim()
#         ax.grid()
#         if xw == 0:
#             if yi == 0:
#                 plt.text(
#                     np.mean(xlim),
#                     ylim[0] + 1.3 * (ylim[1] - ylim[0]),
#                     struct,
#                     fontsize=fs2,
#                     horizontalalignment="center",
#                 )
#                 plt.text(
#                     np.mean(xlim),
#                     ylim[0] + 1.2 * (ylim[1] - ylim[0]),
#                     selected_structures_org[xi],
#                     fontsize=fs2,
#                     horizontalalignment="center",
#                 )
#                 plt.text(
#                     np.mean(xlim),
#                     ylim[0] + 1.1 * (ylim[1] - ylim[0]),
#                     f"n= {len(x)}",
#                     fontsize=fs2,
#                     horizontalalignment="center",
#                 )
#             if xi == 0:
#                 plt.figtext(
#                     0.5,
#                     h1 + ((row - 1) * (yw + yy + h2)) - h2 / 2,
#                     metric,
#                     fontsize=fs3,
#                     horizontalalignment="center",
#                 )
#         else:
#             # ax.set_title(f"{selected_structures[xi]} vs {selected_metrics_abb[yi]} (n= {len(x)})", fontsize=fs2)
#             ax.set_title(f"n= {len(x)}", fontsize=fs2)
#             ax.set_xticklabels([])
#             ax.set_yticklabels([])
#
#         ax.spines["bottom"].set_color(colordict[selected_structures_cat[xi]])
#         ax.spines["bottom"].set_linewidth(lw3)
#         ax.spines["top"].set_color(colordict[selected_structures_cat[xi]])
#         ax.spines["top"].set_linewidth(lw3)
#         ax.spines["right"].set_color(colordict[selected_structures_cat[xi]])
#         ax.spines["right"].set_linewidth(lw3)
#         ax.spines["left"].set_color(colordict[selected_structures_cat[xi]])
#         ax.spines["left"].set_linewidth(lw3)
#
#         if xw != 0:
#             # Bottom histogram
#             ax = fig.add_axes(
#                 [
#                     w1 + ((col - 1) * (xw + xx + w2)) + xw,
#                     h1 + ((row - 1) * (yw + yy + h2)),
#                     xx,
#                     yw,
#                 ]
#             )
#             ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 1])
#             ax.set_xticks(xticks)
#             ax.set_yticks([])
#             ax.set_yticklabels([])
#             ax.set_xlim(left=xlim[0], right=xlim[1])
#             ax.grid()
#             # if yi==len(selected_metrics_abb):
#             ax.set_xlabel(selected_metrics_abb[yi], fontsize=fs2)
#             ax.invert_yaxis()
#
#             # Side histogram
#             ax = fig.add_axes(
#                 [
#                     w1 + ((col - 1) * (xw + xx + w2)),
#                     h1 + ((row - 1) * (yw + yy + h2)) + yw,
#                     xw,
#                     yy,
#                 ]
#             )
#             ax.hist(
#                 y, bins=nbins, color=[0.5, 0.5, 0.5, 1], orientation="horizontal"
#             )
#             ax.set_yticks(yticks)
#             ax.set_xticks([])
#             ax.set_xticklabels([])
#             ax.set_ylim(bottom=ylim[0], top=ylim[1])
#             ax.grid()
#             # if xi==0:
#             ax.set_ylabel(selected_structures[xi], fontsize=fs2)
#             ax.invert_xaxis()
#
# if save_flag:
#     plot_save_path = pic_root / "Structures_org_thick.png"
#     plt.savefig(plot_save_path, format="png", dpi=1000)
#     plt.close()
# else:
#     plt.show()
#
# # %% Parameters
# nbins = 100
# N = 1000
# fac = 1000
# Rounds = 5
#
# # %% Identify pairs, compute stuff and put into dicts
#
# Q = {}  # 'Structure volume'
# # structure_metric = 'Number of pieces'
#
# for xm, metric in tqdm(enumerate(selected_metrics), "Iterating metrics"):
#     for ys, struct in tqdm(enumerate(selected_structures), "and structures"):
#
#         # data
#         x = cells.loc[cells["structure_name"] == struct, [metric]].squeeze()
#         y = cells.loc[
#             cells["structure_name"] == struct, [structure_metric]
#         ].squeeze()
#         x = x.to_numpy()
#         y = y.to_numpy()
#         x = x / fac
#         y = y / fac
#
#         # density estimate
#         for round in np.arange(Rounds):
#             rs = int(datetime.datetime.utcnow().timestamp())
#             xS, yS = resample(
#                 x, y, replace=False, n_samples=np.amin([N, len(x)]), random_state=rs
#             )
#             k = gaussian_kde(np.vstack([xS, yS]))
#             zii = k(np.vstack([xii.flatten(), yii.flatten()]))
#             cell_dens = k(np.vstack([x.flatten(), y.flatten()]))
#             cell_dens = cell_dens / np.sum(cell_dens)
#             # make into cumulative sum
#             zii = zii / np.sum(zii)
#             ix = np.argsort(zii)
#             zii = zii[ix]
#             zii = np.cumsum(zii)
#             jx = np.argsort(ix)
#             zii = zii[jx]
#             zii = zii.reshape(xii.shape)
#             Q[f"{metric}_{struct}_dens_x_{round}"] = xii
#             Q[f"{metric}_{struct}_dens_y_{round}"] = yii
#             Q[f"{metric}_{struct}_dens_z_{round}"] = zii
#             Q[f"{metric}_{struct}_dens_c_{round}"] = cell_dens
#
# # %% Initial scatterplot
# nrows = len(selected_metrics)
# ncols = len(selected_structures)
# w1 = 0.027
# w2 = 0.01
# w3 = 0.002
# h1 = 0.07
# h2 = 0.08
# h3 = 0.07
# xw = 0
# yw = 0
# xx = (1 - w1 - ((ncols - 1) * w2) - w3 - (ncols * xw)) / ncols
# yy = (1 - h1 - ((nrows - 1) * h2) - h3 - (nrows * yw)) / nrows
#
# fig = plt.figure(figsize=(16, 9))
#
# i = 0
#
# dens_th = 1e-15
# remove_cells = []
#
# for yi, metric in enumerate(selected_metrics):
#     for xi, struct in enumerate(selected_structures):
#
#         x = cells.loc[cells["structure_name"] == struct, [metric]].squeeze()
#         y = cells.loc[
#             cells["structure_name"] == struct, [structure_metric]
#         ].squeeze()
#         selcel = (cells["structure_name"] == struct).to_numpy()
#         struct_pos = np.argwhere(selcel)
#         x = x.to_numpy()
#         y = y.to_numpy()
#         x = x / fac
#         y = y / fac
#
#         # select subplot
#         i = i + 1
#         row = nrows - np.ceil(i / ncols) + 1
#         row = row.astype(np.int64)
#         col = i % ncols
#         if col == 0:
#             col = ncols
#         print(f"{i}_{row}_{col}")
#
#         # Main scatterplot
#         ax = fig.add_axes(
#             [
#                 w1 + ((col - 1) * (xw + xx + w2)) + xw,
#                 h1 + ((row - 1) * (yw + yy + h2)) + yw,
#                 xx,
#                 yy,
#             ]
#         )
#         ax.plot(x, y, "b.", markersize=ms)
#         pos = []
#         for round in np.arange(Rounds):
#             cii = Q[f"{metric}_{struct}_dens_c_{round}"]
#             pos = np.union1d(pos, np.argwhere(cii < dens_th))
#             print(len(pos))
#         print(len(pos))
#         pos = pos.astype(int)
#         remove_cells = np.union1d(remove_cells, struct_pos[pos])
#         ax.plot(x[pos], y[pos], "r.", markersize=ms2)
#         xticks = ax.get_xticks()
#         yticks = ax.get_yticks()
#         xlim = ax.get_xlim()
#         ylim = ax.get_ylim()
#         ax.grid()
#         if xw == 0:
#             if yi == 0:
#                 plt.text(
#                     np.mean(xlim),
#                     ylim[0] + 1.3 * (ylim[1] - ylim[0]),
#                     struct,
#                     fontsize=fs2,
#                     horizontalalignment="center",
#                 )
#                 plt.text(
#                     np.mean(xlim),
#                     ylim[0] + 1.2 * (ylim[1] - ylim[0]),
#                     selected_structures_org[xi],
#                     fontsize=fs2,
#                     horizontalalignment="center",
#                 )
#                 plt.text(
#                     np.mean(xlim),
#                     ylim[0] + 1.1 * (ylim[1] - ylim[0]),
#                     f"n= {len(x)}",
#                     fontsize=fs2,
#                     horizontalalignment="center",
#                 )
#             if xi == 0:
#                 plt.figtext(
#                     0.5,
#                     h1 + ((row - 1) * (yw + yy + h2)) - h2 / 2,
#                     metric,
#                     fontsize=fs3,
#                     horizontalalignment="center",
#                 )
#         else:
#             # ax.set_title(f"{selected_structures[xi]} vs {selected_metrics_abb[yi]} (n= {len(x)})", fontsize=fs2)
#             ax.set_title(f"n= {len(x)}", fontsize=fs2)
#             ax.set_xticklabels([])
#             ax.set_yticklabels([])
#
#         ax.spines["bottom"].set_color(colordict[selected_structures_cat[xi]])
#         ax.spines["bottom"].set_linewidth(lw3)
#         ax.spines["top"].set_color(colordict[selected_structures_cat[xi]])
#         ax.spines["top"].set_linewidth(lw3)
#         ax.spines["right"].set_color(colordict[selected_structures_cat[xi]])
#         ax.spines["right"].set_linewidth(lw3)
#         ax.spines["left"].set_color(colordict[selected_structures_cat[xi]])
#         ax.spines["left"].set_linewidth(lw3)
#
#         if xw != 0:
#             # Bottom histogram
#             ax = fig.add_axes(
#                 [
#                     w1 + ((col - 1) * (xw + xx + w2)) + xw,
#                     h1 + ((row - 1) * (yw + yy + h2)),
#                     xx,
#                     yw,
#                 ]
#             )
#             ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 1])
#             ax.set_xticks(xticks)
#             ax.set_yticks([])
#             ax.set_yticklabels([])
#             ax.set_xlim(left=xlim[0], right=xlim[1])
#             ax.grid()
#             # if yi==len(selected_metrics_abb):
#             ax.set_xlabel(selected_metrics_abb[yi], fontsize=fs2)
#             ax.invert_yaxis()
#
#             # Side histogram
#             ax = fig.add_axes(
#                 [
#                     w1 + ((col - 1) * (xw + xx + w2)),
#                     h1 + ((row - 1) * (yw + yy + h2)) + yw,
#                     xw,
#                     yy,
#                 ]
#             )
#             ax.hist(
#                 y, bins=nbins, color=[0.5, 0.5, 0.5, 1], orientation="horizontal"
#             )
#             ax.set_yticks(yticks)
#             ax.set_xticks([])
#             ax.set_xticklabels([])
#             ax.set_ylim(bottom=ylim[0], top=ylim[1])
#             ax.grid()
#             # if xi==0:
#             ax.set_ylabel(selected_structures[xi], fontsize=fs2)
#             ax.invert_xaxis()
#
# if save_flag:
#     plot_save_path = pic_root / "Structures_outliers.png"
#     plt.savefig(plot_save_path, format="png", dpi=1000)
#     plt.close()
# else:
#     plt.show()
#
# print(len(remove_cells))
#
# # %%  Drop them
# cells = cells.drop(cells.index[remove_cells.astype(int)])
#
# # %% Initial scatterplot
# ms = 2
# nrows = len(selected_metrics)
# ncols = len(selected_structures)
# w1 = 0.027
# w2 = 0.01
# w3 = 0.002
# h1 = 0.07
# h2 = 0.08
# h3 = 0.07
# xw = 0
# yw = 0
# xx = (1 - w1 - ((ncols - 1) * w2) - w3 - (ncols * xw)) / ncols
# yy = (1 - h1 - ((nrows - 1) * h2) - h3 - (nrows * yw)) / nrows
#
# fig = plt.figure(figsize=(16, 9))
#
# i = 0
#
# for yi, metric in enumerate(selected_metrics):
#     for xi, struct in enumerate(selected_structures):
#
#         x = cells.loc[cells["structure_name"] == struct, [metric]].squeeze()
#         y = cells.loc[
#             cells["structure_name"] == struct, [structure_metric]
#         ].squeeze()
#         selcel = (cells["structure_name"] == struct).to_numpy()
#         struct_pos = np.argwhere(selcel)
#         x = x.to_numpy()
#         y = y.to_numpy()
#         x = x / fac
#         y = y / fac
#
#         # select subplot
#         i = i + 1
#         row = nrows - np.ceil(i / ncols) + 1
#         row = row.astype(np.int64)
#         col = i % ncols
#         if col == 0:
#             col = ncols
#         print(f"{i}_{row}_{col}")
#
#         # Main scatterplot
#         ax = fig.add_axes(
#             [
#                 w1 + ((col - 1) * (xw + xx + w2)) + xw,
#                 h1 + ((row - 1) * (yw + yy + h2)) + yw,
#                 xx,
#                 yy,
#             ]
#         )
#         ax.plot(x, y, "b.", markersize=ms)
#         xticks = ax.get_xticks()
#         yticks = ax.get_yticks()
#         xlim = ax.get_xlim()
#         ylim = ax.get_ylim()
#         ax.grid()
#         if xw == 0:
#             if yi == 0:
#                 plt.text(
#                     np.mean(xlim),
#                     ylim[0] + 1.3 * (ylim[1] - ylim[0]),
#                     struct,
#                     fontsize=fs2,
#                     horizontalalignment="center",
#                 )
#                 plt.text(
#                     np.mean(xlim),
#                     ylim[0] + 1.2 * (ylim[1] - ylim[0]),
#                     selected_structures_org[xi],
#                     fontsize=fs2,
#                     horizontalalignment="center",
#                 )
#                 plt.text(
#                     np.mean(xlim),
#                     ylim[0] + 1.1 * (ylim[1] - ylim[0]),
#                     f"n= {len(x)}",
#                     fontsize=fs2,
#                     horizontalalignment="center",
#                 )
#             if xi == 0:
#                 plt.figtext(
#                     0.5,
#                     h1 + ((row - 1) * (yw + yy + h2)) - h2 / 2,
#                     metric,
#                     fontsize=fs3,
#                     horizontalalignment="center",
#                 )
#         else:
#             # ax.set_title(f"{selected_structures[xi]} vs {selected_metrics_abb[yi]} (n= {len(x)})", fontsize=fs2)
#             ax.set_title(f"n= {len(x)}", fontsize=fs2)
#             ax.set_xticklabels([])
#             ax.set_yticklabels([])
#
#         ax.spines["bottom"].set_color(colordict[selected_structures_cat[xi]])
#         ax.spines["bottom"].set_linewidth(lw3)
#         ax.spines["top"].set_color(colordict[selected_structures_cat[xi]])
#         ax.spines["top"].set_linewidth(lw3)
#         ax.spines["right"].set_color(colordict[selected_structures_cat[xi]])
#         ax.spines["right"].set_linewidth(lw3)
#         ax.spines["left"].set_color(colordict[selected_structures_cat[xi]])
#         ax.spines["left"].set_linewidth(lw3)
#
#         if xw != 0:
#             # Bottom histogram
#             ax = fig.add_axes(
#                 [
#                     w1 + ((col - 1) * (xw + xx + w2)) + xw,
#                     h1 + ((row - 1) * (yw + yy + h2)),
#                     xx,
#                     yw,
#                 ]
#             )
#             ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 1])
#             ax.set_xticks(xticks)
#             ax.set_yticks([])
#             ax.set_yticklabels([])
#             ax.set_xlim(left=xlim[0], right=xlim[1])
#             ax.grid()
#             # if yi==len(selected_metrics_abb):
#             ax.set_xlabel(selected_metrics_abb[yi], fontsize=fs2)
#             ax.invert_yaxis()
#
#             # Side histogram
#             ax = fig.add_axes(
#                 [
#                     w1 + ((col - 1) * (xw + xx + w2)),
#                     h1 + ((row - 1) * (yw + yy + h2)) + yw,
#                     xw,
#                     yy,
#                 ]
#             )
#             ax.hist(
#                 y, bins=nbins, color=[0.5, 0.5, 0.5, 1], orientation="horizontal"
#             )
#             ax.set_yticks(yticks)
#             ax.set_xticks([])
#             ax.set_xticklabels([])
#             ax.set_ylim(bottom=ylim[0], top=ylim[1])
#             ax.grid()
#             # if xi==0:
#             ax.set_ylabel(selected_structures[xi], fontsize=fs2)
#             ax.invert_xaxis()
#
# if save_flag:
#     plot_save_path = pic_root / "Structures_clean_thick.png"
#     plt.savefig(plot_save_path, format="png", dpi=1000)
#     plt.close()
# else:
#     plt.show()
#
# # %% Initial scatterplot
# ms = 0.5
# nrows = len(selected_metrics)
# ncols = len(selected_structures)
# w1 = 0.027
# w2 = 0.01
# w3 = 0.002
# h1 = 0.07
# h2 = 0.08
# h3 = 0.07
# xw = 0
# yw = 0
# xx = (1 - w1 - ((ncols - 1) * w2) - w3 - (ncols * xw)) / ncols
# yy = (1 - h1 - ((nrows - 1) * h2) - h3 - (nrows * yw)) / nrows
#
# fig = plt.figure(figsize=(16, 9))
#
# i = 0
#
# for yi, metric in enumerate(selected_metrics):
#     for xi, struct in enumerate(selected_structures):
#
#         x = cells.loc[cells["structure_name"] == struct, [metric]].squeeze()
#         y = cells.loc[
#             cells["structure_name"] == struct, [structure_metric]
#         ].squeeze()
#         selcel = (cells["structure_name"] == struct).to_numpy()
#         struct_pos = np.argwhere(selcel)
#         x = x.to_numpy()
#         y = y.to_numpy()
#         x = x / fac
#         y = y / fac
#
#         # select subplot
#         i = i + 1
#         row = nrows - np.ceil(i / ncols) + 1
#         row = row.astype(np.int64)
#         col = i % ncols
#         if col == 0:
#             col = ncols
#         print(f"{i}_{row}_{col}")
#
#         # Main scatterplot
#         ax = fig.add_axes(
#             [
#                 w1 + ((col - 1) * (xw + xx + w2)) + xw,
#                 h1 + ((row - 1) * (yw + yy + h2)) + yw,
#                 xx,
#                 yy,
#             ]
#         )
#         ax.plot(x, y, "b.", markersize=ms)
#         xticks = ax.get_xticks()
#         yticks = ax.get_yticks()
#         xlim = ax.get_xlim()
#         ylim = ax.get_ylim()
#         ax.grid()
#         if xw == 0:
#             if yi == 0:
#                 plt.text(
#                     np.mean(xlim),
#                     ylim[0] + 1.3 * (ylim[1] - ylim[0]),
#                     struct,
#                     fontsize=fs2,
#                     horizontalalignment="center",
#                 )
#                 plt.text(
#                     np.mean(xlim),
#                     ylim[0] + 1.2 * (ylim[1] - ylim[0]),
#                     selected_structures_org[xi],
#                     fontsize=fs2,
#                     horizontalalignment="center",
#                 )
#                 plt.text(
#                     np.mean(xlim),
#                     ylim[0] + 1.1 * (ylim[1] - ylim[0]),
#                     f"n= {len(x)}",
#                     fontsize=fs2,
#                     horizontalalignment="center",
#                 )
#             if xi == 0:
#                 plt.figtext(
#                     0.5,
#                     h1 + ((row - 1) * (yw + yy + h2)) - h2 / 2,
#                     metric,
#                     fontsize=fs3,
#                     horizontalalignment="center",
#                 )
#         else:
#             # ax.set_title(f"{selected_structures[xi]} vs {selected_metrics_abb[yi]} (n= {len(x)})", fontsize=fs2)
#             ax.set_title(f"n= {len(x)}", fontsize=fs2)
#             ax.set_xticklabels([])
#             ax.set_yticklabels([])
#
#         ax.spines["bottom"].set_color(colordict[selected_structures_cat[xi]])
#         ax.spines["bottom"].set_linewidth(lw3)
#         ax.spines["top"].set_color(colordict[selected_structures_cat[xi]])
#         ax.spines["top"].set_linewidth(lw3)
#         ax.spines["right"].set_color(colordict[selected_structures_cat[xi]])
#         ax.spines["right"].set_linewidth(lw3)
#         ax.spines["left"].set_color(colordict[selected_structures_cat[xi]])
#         ax.spines["left"].set_linewidth(lw3)
#
#         if xw != 0:
#             # Bottom histogram
#             ax = fig.add_axes(
#                 [
#                     w1 + ((col - 1) * (xw + xx + w2)) + xw,
#                     h1 + ((row - 1) * (yw + yy + h2)),
#                     xx,
#                     yw,
#                 ]
#             )
#             ax.hist(x, bins=nbins, color=[0.5, 0.5, 0.5, 1])
#             ax.set_xticks(xticks)
#             ax.set_yticks([])
#             ax.set_yticklabels([])
#             ax.set_xlim(left=xlim[0], right=xlim[1])
#             ax.grid()
#             # if yi==len(selected_metrics_abb):
#             ax.set_xlabel(selected_metrics_abb[yi], fontsize=fs2)
#             ax.invert_yaxis()
#
#             # Side histogram
#             ax = fig.add_axes(
#                 [
#                     w1 + ((col - 1) * (xw + xx + w2)),
#                     h1 + ((row - 1) * (yw + yy + h2)) + yw,
#                     xw,
#                     yy,
#                 ]
#             )
#             ax.hist(
#                 y, bins=nbins, color=[0.5, 0.5, 0.5, 1], orientation="horizontal"
#             )
#             ax.set_yticks(yticks)
#             ax.set_xticks([])
#             ax.set_xticklabels([])
#             ax.set_ylim(bottom=ylim[0], top=ylim[1])
#             ax.grid()
#             # if xi==0:
#             ax.set_ylabel(selected_structures[xi], fontsize=fs2)
#             ax.invert_xaxis()
#
# if save_flag:
#     plot_save_path = pic_root / "Structures_clean_fine.png"
#     plt.savefig(plot_save_path, format="png", dpi=1000)
#     plt.close()
# else:
#     plt.show()
#
# # Save cleaned dataset
# cells.to_csv(data_root / dataset_clean)
