#%%

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
import statsmodels.api as sm
import pickle
import psutil
import os, platform
import sys, importlib

# Third party

# Relative
from stemcellorganellesizescaling.analyses.utils.pairwise_stats_and_regression import fit_ols, calculate_pairwisestats
importlib.reload(sys.modules["stemcellorganellesizescaling.analyses.utils.pairwise_stats_and_regression"])
from stemcellorganellesizescaling.analyses.utils.pairwise_stats_and_regression import fit_ols, calculate_pairwisestats

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

#%%
# Resolve directories
data_root = dirs[0]
pic_root = dirs[1]

dataset = "SizeScaling_20201006_clean.csv"
dataset_comp = "SizeScaling_20201006_comp.csv"
statsOUTdir = "Stats_20201006"

# Load datasets
cells = pd.read_csv(data_root / dataset)
cells_COMP = pd.read_csv(data_root / dataset_comp)

# Create directory to store stats
(data_root / statsOUTdir).mkdir(exist_ok=True)

# %% Define feature pair sets for cell and nuclear metrics

# %% Select feature sets
FS = {}
FS['cell_metrics_AVH'] = ["Cell surface area", "Cell volume", "Cell height"]
FS['nuc_metrics_AVH'] = ["Nuclear surface area", "Nuclear volume", "Nucleus height"]
FS['cell_metrics_AV'] = ["Cell surface area", "Cell volume"]
FS['nuc_metrics_AV'] = ["Nuclear surface area", "Nuclear volume"]
FS['cell_metrics_H'] = ["Cell height"]
FS['nuc_metrics_H'] = ["Nucleus height"]
FS['cellnuc_metrics'] = [
    "Cell surface area",
    "Cell volume",
    "Cell height",
    "Nuclear surface area",
    "Nuclear volume",
    "Nucleus height",
    "Cytoplasmic volume",
]
# struct_metrics = [
#     "Structure volume",
#     "Number of pieces",
#     "Piece average",
#     "Piece std",
#     "Piece CoV",
#     "Piece sum",
# ]
FS['struct_metrics'] = [
    "Structure volume",
    "Number of pieces"
]

# %% Part 1 pairwise stats cell and nucleus measurement
print('Cell and nucleus metrics')
D = {}
for xi, xlabel in enumerate(FS['cellnuc_metrics']):
    for yi, ylabel in enumerate(FS['cellnuc_metrics']):
        if xlabel is not ylabel:
            print(f"{xlabel} vs {ylabel}")
            x = cells[xlabel].squeeze().to_numpy()
            x = np.expand_dims(x, axis=1)
            y = cells[ylabel].squeeze().to_numpy()
            y = np.expand_dims(y, axis=1)
            D.update(calculate_pairwisestats(x, y, xlabel, ylabel, 'None'))
# prepare directory
save_dir = (data_root / statsOUTdir / 'cell_nuc_metrics')
save_dir.mkdir(exist_ok=True)
#save
manifest = pd.DataFrame(columns=['pairstats','size'], index=np.arange(len(D)))
i = 0
for key in tqdm(D,'saving all computed statistics'):
    pfile = save_dir / f"{key}.pickle"
    if pfile.is_file():
        pfile.unlink()
    with open(pfile, "wb") as f:
        pickle.dump(D[key], f)
    manifest.loc[i,'pairstats'] = key
    manifest.loc[i, 'size'] = len(D[key])
    i += 1
manifest.to_csv(save_dir / 'cell_nuc_metrics_manifest.csv')

print('done')

# %% Part 2 pairwise stats cell and nucleus measurement
print('Cell and nucleus metrics vs structure metrics')
D = {}
for xi, xlabel in enumerate(FS['cellnuc_metrics']):
    for yi, ylabel in enumerate(FS['struct_metrics']):
            print(f"{xlabel} vs {ylabel}")
            selected_structures = cells["structure_name"].unique()
            for si, struct in enumerate(selected_structures):
                print(f"{struct}")
                x = cells.loc[cells["structure_name"] == struct, xlabel].squeeze().to_numpy()
                x = np.expand_dims(x, axis=1)
                y = cells.loc[cells["structure_name"] == struct, ylabel].squeeze().to_numpy()
                y = np.expand_dims(y, axis=1)
                D.update(calculate_pairwisestats(x, y, xlabel, ylabel, struct))
# prepare directory
save_dir = (data_root / statsOUTdir / 'cellnuc_struct_metrics')
save_dir.mkdir(exist_ok=True)
# save
manifest = pd.DataFrame(columns=['pairstats', 'size'], index=np.arange(len(D)))
i = 0
for key in tqdm(D, 'saving all computed statistics'):
    pfile = save_dir / f"{key}.pickle"
    if pfile.is_file():
        pfile.unlink()
    with open(pfile, "wb") as f:
        pickle.dump(D[key], f)
    manifest.loc[i, 'pairstats'] = key
    manifest.loc[i, 'size'] = len(D[key])
    i += 1
manifest.to_csv(save_dir / 'cellnuc_struct_metrics_manifest.csv')

print('done')


# # %%
#
# elif struct_flag is True:
# selected_structures = df_in["structure_name"].unique()
# for si, struct in enumerate(selected_structures):
#     print(f"{struct}")
#     x = (
#         df_in.loc[df_in["structure_name"] == struct, feature1]
#             .squeeze()
#             .to_numpy()
#     )
#     x = np.expand_dims(x, axis=1)
#     y = (
#         df_in.loc[df_in["structure_name"] == struct, feature2]
#             .squeeze()
#             .to_numpy()
#     )
#     y = np.expand_dims(y, axis=1)
#
#     (
#         xi,
#         rs_vecL,
#         pred_matL,
#         rs_vecC,
#         pred_matC,
#         xii,
#         yii,
#         zii,
#         cell_dens,
#         x_ra,
#         y_ra,
#     ) = calculate_pairwisestats(x, y)
#
#     PairStats[f"{feature1}_{feature1}_{struct}_xi"] = xi
#     PairStats[f"{feature1}_{feature2}_{struct}_rs_vecL"] = rs_vecL
#     PairStats[
#         f"{feature1}_{feature2}_{struct}_pred_matL"
#     ] = pred_matL
#     PairStats[f"{feature1}_{feature2}_{struct}_rs_vecC"] = rs_vecC
#     PairStats[
#         f"{feature1}_{feature2}_{struct}_pred_matC"
#     ] = pred_matC
#     PairStats[f"{feature1}_{feature2}_{struct}_xii"] = xii
#     PairStats[f"{feature1}_{feature2}_{struct}_yii"] = yii
#     PairStats[f"{feature1}_{feature2}_{struct}_zii"] = zii
#     PairStats[
#         f"{feature1}_{feature2}_{struct}_cell_dens"
#     ] = cell_dens
#     PairStats[f"{feature1}_{feature2}_{struct}_x_ra"] = x_ra
#     PairStats[f"{feature1}_{feature2}_{struct}_y_ra"] = y_ra
#
#
#
# # %%
#
# PSlist = pd.DataFrame(columns=['pairstats','size'], index=np.arange(len(PS)))
# i = 0
# for key in tqdm(PS,'wow'):
#     pfile = data_root / 'wf20200915' / f"{key}.pickle"
#     if pfile.is_file():
#         pfile.unlink()
#     with open(pfile, "wb") as f:
#         pickle.dump(PS[key], f)
#     # # print(key, '->', PS[key])
#     PSlist.loc[i,'pairstats'] = key
#     PSlist.loc[i, 'size'] = len(PS[key])
#     i += 1
# PSlist.to_csv(data_root / 'wf20200915' / 'PSlist.csv'
#
#
#
#
# #%%
#
#         #     (
#         #         xi,
#         #         rs_vecL,
#         #         pred_matL,
#         #         rs_vecC,
#         #         pred_matC,
#         #         xii,
#         #         yii,
#         #         zii,
#         #         cell_dens,
#         #         x_ra,
#         #         y_ra,
#         #     ) = calculate_pairwisestats(x, y)
#         #
#         #
#         #
#         #
#         #
#         #
#         #
#         # x = cells[features_4_comp].squeeze().to_numpy()
#         # y = cells[ylabel].squeeze().to_numpy()
#         #
#         #     fittedmodel, _ = fit_ols(x, y, type)
#         #     yr = np.expand_dims(fittedmodel.resid, axis=1)
#         #     cells_COMP[f"{ylabel}_COMP_{type}_{xlabel}"] = yr
#
# # %%
# for f1, feature1 in enumerate(features_1):
#     for f2, feature2 in enumerate(features_2):
#         if feature1 is not feature2:
#             print(f"{feature1} vs {feature2}")
#
#             if struct_flag is False:
#                 x = df_in[feature1].squeeze().to_numpy()
#                 x = np.expand_dims(x, axis=1)
#                 y = df_in[feature2].squeeze().to_numpy()
#                 y = np.expand_dims(y, axis=1)
#
#                 D = calculate_pairwisestats(x, y, xlabel, ylabel)
#
#
#
#
#
#
#     # %%
#  %% Pairwise statistics
# PS = {}
# PS.update(pairstats(cells, cellnuc_metrics, cellnuc_metrics, False))
# PS.update(pairstats(cells, cellnuc_metrics, struct_metrics, True))
# PS.update(
#     pairstats(
#         cells_COMP, cell_metrics_COMP_Linear_Nuc, struct_metrics_COMP_Linear_Nuc, True
#     )
# )
# PS.update(
#     pairstats(
#         cells_COMP, cell_metrics_COMP_Complex_Nuc, struct_metrics_COMP_Complex_Nuc, True
#     )
# )
# PS.update(
#     pairstats(
#         cells_COMP, nuc_metrics_COMP_Linear_Cell, struct_metrics_COMP_Linear_Cell, True
#     )
# )
# PS.update(
#     pairstats(
#         cells_COMP,
#         nuc_metrics_COMP_Complex_Cell,
#         struct_metrics_COMP_Complex_Cell,
#         True,
#     )
# )
#
#
#
#
# cell_metrics_COMP_Linear_Nuc = [
#     "Cell surface area_COMP_Linear_Nuc_AVH",
#     "Cell volume_COMP_Linear_Nuc_AVH",
#     "Cell height_COMP_Linear_Nuc_AVH",
#     "Cell surface area_COMP_Linear_Nuc_AV",
#     "Cell volume_COMP_Linear_Nuc_AV",
#     "Cell height_COMP_Linear_Nuc_AV",
#     "Cell surface area_COMP_Linear_Nuc_H",
#     "Cell volume_COMP_Linear_Nuc_H",
#     "Cell height_COMP_Linear_Nuc_H",
# ]
# cell_metrics_COMP_Complex_Nuc = [
#     "Cell surface area_COMP_Complex_Nuc_AVH",
#     "Cell volume_COMP_Complex_Nuc_AVH",
#     "Cell height_COMP_Complex_Nuc_AVH",
#     "Cell surface area_COMP_Complex_Nuc_AV",
#     "Cell volume_COMP_Complex_Nuc_AV",
#     "Cell height_COMP_Complex_Nuc_AV",
#     "Cell surface area_COMP_Complex_Nuc_H",
#     "Cell volume_COMP_Complex_Nuc_H",
#     "Cell height_COMP_Complex_Nuc_H",
# ]
# nuc_metrics_COMP_Linear_Cell = [
#     "Nuclear surface area_COMP_Linear_Cell_AVH",
#     "Nuclear volume_COMP_Linear_Cell_AVH",
#     "Nucleus height_COMP_Linear_Cell_AVH",
#     "Nuclear surface area_COMP_Linear_Cell_AV",
#     "Nuclear volume_COMP_Linear_Cell_AV",
#     "Nucleus height_COMP_Linear_Cell_AV",
#     "Nuclear surface area_COMP_Linear_Cell_H",
#     "Nuclear volume_COMP_Linear_Cell_H",
#     "Nucleus height_COMP_Linear_Cell_H",
# ]
# nuc_metrics_COMP_Complex_Cell = [
#     "Nuclear surface area_COMP_Complex_Cell_AVH",
#     "Nuclear volume_COMP_Complex_Cell_AVH",
#     "Nucleus height_COMP_Complex_Cell_AVH",
#     "Nuclear surface area_COMP_Complex_Cell_AV",
#     "Nuclear volume_COMP_Complex_Cell_AV",
#     "Nucleus height_COMP_Complex_Cell_AV",
#     "Nuclear surface area_COMP_Complex_Cell_H",
#     "Nuclear volume_COMP_Complex_Cell_H",
#     "Nucleus height_COMP_Complex_Cell_H",
# ]
#
# struct_metrics_COMP_Linear_Nuc = [
#     "Structure volume_COMP_Linear_Nuc_AVH",
#     "Number of pieces_COMP_Linear_Nuc_AVH",
#     "Piece average_COMP_Linear_Nuc_AVH",
#     "Piece std_COMP_Linear_Nuc_AVH",
#     "Piece CoV_COMP_Linear_Nuc_AVH",
#     "Piece sum_COMP_Linear_Nuc_AVH",
#     "Structure volume_COMP_Linear_Nuc_AV",
#     "Number of pieces_COMP_Linear_Nuc_AV",
#     "Piece average_COMP_Linear_Nuc_AV",
#     "Piece std_COMP_Linear_Nuc_AV",
#     "Piece CoV_COMP_Linear_Nuc_AV",
#     "Piece sum_COMP_Linear_Nuc_AV",
#     "Structure volume_COMP_Linear_Nuc_H",
#     "Number of pieces_COMP_Linear_Nuc_H",
#     "Piece average_COMP_Linear_Nuc_H",
#     "Piece std_COMP_Linear_Nuc_H",
#     "Piece CoV_COMP_Linear_Nuc_H",
#     "Piece sum_COMP_Linear_Nuc_H",
# ]
# struct_metrics_COMP_Complex_Nuc = [
#     "Structure volume_COMP_Complex_Nuc_AVH",
#     "Number of pieces_COMP_Complex_Nuc_AVH",
#     "Piece average_COMP_Complex_Nuc_AVH",
#     "Piece std_COMP_Complex_Nuc_AVH",
#     "Piece CoV_COMP_Complex_Nuc_AVH",
#     "Piece sum_COMP_Complex_Nuc_AVH",
#     "Structure volume_COMP_Complex_Nuc_AV",
#     "Number of pieces_COMP_Complex_Nuc_AV",
#     "Piece average_COMP_Complex_Nuc_AV",
#     "Piece std_COMP_Complex_Nuc_AV",
#     "Piece CoV_COMP_Complex_Nuc_AV",
#     "Piece sum_COMP_Complex_Nuc_AV",
#     "Structure volume_COMP_Complex_Nuc_H",
#     "Number of pieces_COMP_Complex_Nuc_H",
#     "Piece average_COMP_Complex_Nuc_H",
#     "Piece std_COMP_Complex_Nuc_H",
#     "Piece CoV_COMP_Complex_Nuc_H",
#     "Piece sum_COMP_Complex_Nuc_H",
# ]
# struct_metrics_COMP_Linear_Cell = [
#     "Structure volume_COMP_Linear_Cell_AVH",
#     "Number of pieces_COMP_Linear_Cell_AVH",
#     "Piece average_COMP_Linear_Cell_AVH",
#     "Piece std_COMP_Linear_Cell_AVH",
#     "Piece CoV_COMP_Linear_Cell_AVH",
#     "Piece sum_COMP_Linear_Cell_AVH",
#     "Structure volume_COMP_Linear_Cell_AV",
#     "Number of pieces_COMP_Linear_Cell_AV",
#     "Piece average_COMP_Linear_Cell_AV",
#     "Piece std_COMP_Linear_Cell_AV",
#     "Piece CoV_COMP_Linear_Cell_AV",
#     "Piece sum_COMP_Linear_Cell_AV",
#     "Structure volume_COMP_Linear_Cell_H",
#     "Number of pieces_COMP_Linear_Cell_H",
#     "Piece average_COMP_Linear_Cell_H",
#     "Piece std_COMP_Linear_Cell_H",
#     "Piece CoV_COMP_Linear_Cell_H",
#     "Piece sum_COMP_Linear_Cell_H",
# ]
# struct_metrics_COMP_Complex_Cell = [
#     "Structure volume_COMP_Complex_Cell_AVH",
#     "Number of pieces_COMP_Complex_Cell_AVH",
#     "Piece average_COMP_Complex_Cell_AVH",
#     "Piece std_COMP_Complex_Cell_AVH",
#     "Piece CoV_COMP_Complex_Cell_AVH",
#     "Piece sum_COMP_Complex_Cell_AVH",
#     "Structure volume_COMP_Complex_Cell_AV",
#     "Number of pieces_COMP_Complex_Cell_AV",
#     "Piece average_COMP_Complex_Cell_AV",
#     "Piece std_COMP_Complex_Cell_AV",
#     "Piece CoV_COMP_Complex_Cell_AV",
#     "Piece sum_COMP_Complex_Cell_AV",
#     "Structure volume_COMP_Complex_Cell_H",
#     "Number of pieces_COMP_Complex_Cell_H",
#     "Piece average_COMP_Complex_Cell_H",
#     "Piece std_COMP_Complex_Cell_H",
#     "Piece CoV_COMP_Complex_Cell_H",
#     "Piece sum_COMP_Complex_Cell_H",
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
# PS.update(pairstats(cells, cellnuc_metrics, cellnuc_metrics, False))
#
#
#
#
#
# def pairstats(df_in, features_1, features_2, struct_flag):
#     """
#        Compute paired statistics
#
#        Parameters
#        ----------
#        df_in: cells dataframe
#        features_1: List of features in df_in forming one part of each pair
#        features_2: List of features in df_in forming the second part of each pair
#        struct_flag: Flag (True or False) indicating whether the analysis is per structure
#
#        Output
#        ----------
#        PairStats: structure with information
#        """
#     PairStats = {}
#
#     for f1, feature1 in enumerate(features_1):
#         for f2, feature2 in enumerate(features_2):
#             if feature1 is not feature2:
#                 print(f"{feature1} vs {feature2}")
#
#                 if struct_flag is False:
#                     x = df_in[feature1].squeeze().to_numpy()
#                     x = np.expand_dims(x, axis=1)
#                     y = df_in[feature2].squeeze().to_numpy()
#                     y = np.expand_dims(y, axis=1)
#
#                     (
#                         xi,
#                         rs_vecL,
#                         pred_matL,
#                         rs_vecC,
#                         pred_matC,
#                         xii,
#                         yii,
#                         zii,
#                         cell_dens,
#                         x_ra,
#                         y_ra,
#                     ) = calculate_pairwisestats(x, y)
#
#                     PairStats[f"{feature1}_{feature1}_xi"] = xi
#                     PairStats[f"{feature1}_{feature2}_rs_vecL"] = rs_vecL
#                     PairStats[f"{feature1}_{feature2}_pred_matL"] = pred_matL
#                     PairStats[f"{feature1}_{feature2}_rs_vecC"] = rs_vecC
#                     PairStats[f"{feature1}_{feature2}_pred_matC"] = pred_matC
#                     PairStats[f"{feature1}_{feature2}_xii"] = xii
#                     PairStats[f"{feature1}_{feature2}_yii"] = yii
#                     PairStats[f"{feature1}_{feature2}_zii"] = zii
#                     PairStats[f"{feature1}_{feature2}_cell_dens"] = cell_dens
#                     PairStats[f"{feature1}_{feature2}_x_ra"] = x_ra
#                     PairStats[f"{feature1}_{feature2}_y_ra"] = y_ra
#
#                 elif struct_flag is True:
#                     selected_structures = df_in["structure_name"].unique()
#                     for si, struct in enumerate(selected_structures):
#                         print(f"{struct}")
#                         x = (
#                             df_in.loc[df_in["structure_name"] == struct, feature1]
#                             .squeeze()
#                             .to_numpy()
#                         )
#                         x = np.expand_dims(x, axis=1)
#                         y = (
#                             df_in.loc[df_in["structure_name"] == struct, feature2]
#                             .squeeze()
#                             .to_numpy()
#                         )
#                         y = np.expand_dims(y, axis=1)
#
#                         (
#                             xi,
#                             rs_vecL,
#                             pred_matL,
#                             rs_vecC,
#                             pred_matC,
#                             xii,
#                             yii,
#                             zii,
#                             cell_dens,
#                             x_ra,
#                             y_ra,
#                         ) = calculate_pairwisestats(x, y)
#
#                         PairStats[f"{feature1}_{feature1}_{struct}_xi"] = xi
#                         PairStats[f"{feature1}_{feature2}_{struct}_rs_vecL"] = rs_vecL
#                         PairStats[
#                             f"{feature1}_{feature2}_{struct}_pred_matL"
#                         ] = pred_matL
#                         PairStats[f"{feature1}_{feature2}_{struct}_rs_vecC"] = rs_vecC
#                         PairStats[
#                             f"{feature1}_{feature2}_{struct}_pred_matC"
#                         ] = pred_matC
#                         PairStats[f"{feature1}_{feature2}_{struct}_xii"] = xii
#                         PairStats[f"{feature1}_{feature2}_{struct}_yii"] = yii
#                         PairStats[f"{feature1}_{feature2}_{struct}_zii"] = zii
#                         PairStats[
#                             f"{feature1}_{feature2}_{struct}_cell_dens"
#                         ] = cell_dens
#                         PairStats[f"{feature1}_{feature2}_{struct}_x_ra"] = x_ra
#                         PairStats[f"{feature1}_{feature2}_{struct}_y_ra"] = y_ra
#
#     return PairStats
#
#
# # %% Feature sets
# cell_metrics_AVH = ["Cell surface area", "Cell volume", "Cell height"]
# nuc_metrics_AVH = ["Nuclear surface area", "Nuclear volume", "Nucleus height"]
# cell_metrics_AV = ["Cell surface area", "Cell volume"]
# nuc_metrics_AV = ["Nuclear surface area", "Nuclear volume"]
# cell_metrics_H = ["Cell height"]
# nuc_metrics_H = ["Nucleus height"]
# cellnuc_metrics = [
#     "Cell surface area",
#     "Cell volume",
#     "Cell height",
#     "Nuclear surface area",
#     "Nuclear volume",
#     "Nucleus height",
#     "Cytoplasmic volume",
# ]
# struct_metrics = [
#     "Structure volume",
#     "Number of pieces",
#     "Piece average",
#     "Piece std",
#     "Piece CoV",
#     "Piece sum",
# ]
# #%% Compensate
# df1 = compensate(cells, cell_metrics_AVH, nuc_metrics_AVH, False, "Linear", "Nuc_AVH")
# df2 = compensate(cells, cell_metrics_AVH, nuc_metrics_AVH, False, "Complex", "Nuc_AVH")
# df3 = compensate(cells, nuc_metrics_AVH, cell_metrics_AVH, False, "Linear", "Cell_AVH")
# df4 = compensate(cells, nuc_metrics_AVH, cell_metrics_AVH, False, "Complex", "Cell_AVH")
# df5 = compensate(cells, struct_metrics, cell_metrics_AVH, True, "Linear", "Cell_AVH")
# df6 = compensate(cells, struct_metrics, cell_metrics_AVH, True, "Complex", "Cell_AVH")
# df7 = compensate(cells, struct_metrics, nuc_metrics_AVH, True, "Linear", "Nuc_AVH")
# df8 = compensate(cells, struct_metrics, nuc_metrics_AVH, True, "Complex", "Nuc_AVH")
# cells_COMP1 = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], axis=1)
#
# df1 = compensate(cells, cell_metrics_AVH, nuc_metrics_AV, False, "Linear", "Nuc_AV")
# df2 = compensate(cells, cell_metrics_AVH, nuc_metrics_AV, False, "Complex", "Nuc_AV")
# df3 = compensate(cells, nuc_metrics_AVH, cell_metrics_AV, False, "Linear", "Cell_AV")
# df4 = compensate(cells, nuc_metrics_AVH, cell_metrics_AV, False, "Complex", "Cell_AV")
# df5 = compensate(cells, struct_metrics, cell_metrics_AV, True, "Linear", "Cell_AV")
# df6 = compensate(cells, struct_metrics, cell_metrics_AV, True, "Complex", "Cell_AV")
# df7 = compensate(cells, struct_metrics, nuc_metrics_AV, True, "Linear", "Nuc_AV")
# df8 = compensate(cells, struct_metrics, nuc_metrics_AV, True, "Complex", "Nuc_AV")
# cells_COMP2 = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], axis=1)
#
# df1 = compensate(cells, cell_metrics_AVH, nuc_metrics_H, False, "Linear", "Nuc_H")
# df2 = compensate(cells, cell_metrics_AVH, nuc_metrics_H, False, "Complex", "Nuc_H")
# df3 = compensate(cells, nuc_metrics_AVH, cell_metrics_H, False, "Linear", "Cell_H")
# df4 = compensate(cells, nuc_metrics_AVH, cell_metrics_H, False, "Complex", "Cell_H")
# df5 = compensate(cells, struct_metrics, cell_metrics_H, True, "Linear", "Cell_H")
# df6 = compensate(cells, struct_metrics, cell_metrics_H, True, "Complex", "Cell_H")
# df7 = compensate(cells, struct_metrics, nuc_metrics_H, True, "Linear", "Nuc_H")
# df8 = compensate(cells, struct_metrics, nuc_metrics_H, True, "Complex", "Nuc_H")
# cells_COMP3 = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], axis=1)
#
# cells_COMP = pd.concat([cells_COMP1, cells_COMP2, cells_COMP3], axis=1)
# cells_COMP = cells_COMP.loc[:, ~cells_COMP.columns.duplicated()]
#
# # %% Compensated feature sets
# cell_metrics_COMP_Linear_Nuc = [
#     "Cell surface area_COMP_Linear_Nuc_AVH",
#     "Cell volume_COMP_Linear_Nuc_AVH",
#     "Cell height_COMP_Linear_Nuc_AVH",
#     "Cell surface area_COMP_Linear_Nuc_AV",
#     "Cell volume_COMP_Linear_Nuc_AV",
#     "Cell height_COMP_Linear_Nuc_AV",
#     "Cell surface area_COMP_Linear_Nuc_H",
#     "Cell volume_COMP_Linear_Nuc_H",
#     "Cell height_COMP_Linear_Nuc_H",
# ]
# cell_metrics_COMP_Complex_Nuc = [
#     "Cell surface area_COMP_Complex_Nuc_AVH",
#     "Cell volume_COMP_Complex_Nuc_AVH",
#     "Cell height_COMP_Complex_Nuc_AVH",
#     "Cell surface area_COMP_Complex_Nuc_AV",
#     "Cell volume_COMP_Complex_Nuc_AV",
#     "Cell height_COMP_Complex_Nuc_AV",
#     "Cell surface area_COMP_Complex_Nuc_H",
#     "Cell volume_COMP_Complex_Nuc_H",
#     "Cell height_COMP_Complex_Nuc_H",
# ]
# nuc_metrics_COMP_Linear_Cell = [
#     "Nuclear surface area_COMP_Linear_Cell_AVH",
#     "Nuclear volume_COMP_Linear_Cell_AVH",
#     "Nucleus height_COMP_Linear_Cell_AVH",
#     "Nuclear surface area_COMP_Linear_Cell_AV",
#     "Nuclear volume_COMP_Linear_Cell_AV",
#     "Nucleus height_COMP_Linear_Cell_AV",
#     "Nuclear surface area_COMP_Linear_Cell_H",
#     "Nuclear volume_COMP_Linear_Cell_H",
#     "Nucleus height_COMP_Linear_Cell_H",
# ]
# nuc_metrics_COMP_Complex_Cell = [
#     "Nuclear surface area_COMP_Complex_Cell_AVH",
#     "Nuclear volume_COMP_Complex_Cell_AVH",
#     "Nucleus height_COMP_Complex_Cell_AVH",
#     "Nuclear surface area_COMP_Complex_Cell_AV",
#     "Nuclear volume_COMP_Complex_Cell_AV",
#     "Nucleus height_COMP_Complex_Cell_AV",
#     "Nuclear surface area_COMP_Complex_Cell_H",
#     "Nuclear volume_COMP_Complex_Cell_H",
#     "Nucleus height_COMP_Complex_Cell_H",
# ]
#
# struct_metrics_COMP_Linear_Nuc = [
#     "Structure volume_COMP_Linear_Nuc_AVH",
#     "Number of pieces_COMP_Linear_Nuc_AVH",
#     "Piece average_COMP_Linear_Nuc_AVH",
#     "Piece std_COMP_Linear_Nuc_AVH",
#     "Piece CoV_COMP_Linear_Nuc_AVH",
#     "Piece sum_COMP_Linear_Nuc_AVH",
#     "Structure volume_COMP_Linear_Nuc_AV",
#     "Number of pieces_COMP_Linear_Nuc_AV",
#     "Piece average_COMP_Linear_Nuc_AV",
#     "Piece std_COMP_Linear_Nuc_AV",
#     "Piece CoV_COMP_Linear_Nuc_AV",
#     "Piece sum_COMP_Linear_Nuc_AV",
#     "Structure volume_COMP_Linear_Nuc_H",
#     "Number of pieces_COMP_Linear_Nuc_H",
#     "Piece average_COMP_Linear_Nuc_H",
#     "Piece std_COMP_Linear_Nuc_H",
#     "Piece CoV_COMP_Linear_Nuc_H",
#     "Piece sum_COMP_Linear_Nuc_H",
# ]
# struct_metrics_COMP_Complex_Nuc = [
#     "Structure volume_COMP_Complex_Nuc_AVH",
#     "Number of pieces_COMP_Complex_Nuc_AVH",
#     "Piece average_COMP_Complex_Nuc_AVH",
#     "Piece std_COMP_Complex_Nuc_AVH",
#     "Piece CoV_COMP_Complex_Nuc_AVH",
#     "Piece sum_COMP_Complex_Nuc_AVH",
#     "Structure volume_COMP_Complex_Nuc_AV",
#     "Number of pieces_COMP_Complex_Nuc_AV",
#     "Piece average_COMP_Complex_Nuc_AV",
#     "Piece std_COMP_Complex_Nuc_AV",
#     "Piece CoV_COMP_Complex_Nuc_AV",
#     "Piece sum_COMP_Complex_Nuc_AV",
#     "Structure volume_COMP_Complex_Nuc_H",
#     "Number of pieces_COMP_Complex_Nuc_H",
#     "Piece average_COMP_Complex_Nuc_H",
#     "Piece std_COMP_Complex_Nuc_H",
#     "Piece CoV_COMP_Complex_Nuc_H",
#     "Piece sum_COMP_Complex_Nuc_H",
# ]
# struct_metrics_COMP_Linear_Cell = [
#     "Structure volume_COMP_Linear_Cell_AVH",
#     "Number of pieces_COMP_Linear_Cell_AVH",
#     "Piece average_COMP_Linear_Cell_AVH",
#     "Piece std_COMP_Linear_Cell_AVH",
#     "Piece CoV_COMP_Linear_Cell_AVH",
#     "Piece sum_COMP_Linear_Cell_AVH",
#     "Structure volume_COMP_Linear_Cell_AV",
#     "Number of pieces_COMP_Linear_Cell_AV",
#     "Piece average_COMP_Linear_Cell_AV",
#     "Piece std_COMP_Linear_Cell_AV",
#     "Piece CoV_COMP_Linear_Cell_AV",
#     "Piece sum_COMP_Linear_Cell_AV",
#     "Structure volume_COMP_Linear_Cell_H",
#     "Number of pieces_COMP_Linear_Cell_H",
#     "Piece average_COMP_Linear_Cell_H",
#     "Piece std_COMP_Linear_Cell_H",
#     "Piece CoV_COMP_Linear_Cell_H",
#     "Piece sum_COMP_Linear_Cell_H",
# ]
# struct_metrics_COMP_Complex_Cell = [
#     "Structure volume_COMP_Complex_Cell_AVH",
#     "Number of pieces_COMP_Complex_Cell_AVH",
#     "Piece average_COMP_Complex_Cell_AVH",
#     "Piece std_COMP_Complex_Cell_AVH",
#     "Piece CoV_COMP_Complex_Cell_AVH",
#     "Piece sum_COMP_Complex_Cell_AVH",
#     "Structure volume_COMP_Complex_Cell_AV",
#     "Number of pieces_COMP_Complex_Cell_AV",
#     "Piece average_COMP_Complex_Cell_AV",
#     "Piece std_COMP_Complex_Cell_AV",
#     "Piece CoV_COMP_Complex_Cell_AV",
#     "Piece sum_COMP_Complex_Cell_AV",
#     "Structure volume_COMP_Complex_Cell_H",
#     "Number of pieces_COMP_Complex_Cell_H",
#     "Piece average_COMP_Complex_Cell_H",
#     "Piece std_COMP_Complex_Cell_H",
#     "Piece CoV_COMP_Complex_Cell_H",
#     "Piece sum_COMP_Complex_Cell_H",
# ]
# # %% Pairwise statistics
# PS = {}
# PS.update(pairstats(cells, cellnuc_metrics, cellnuc_metrics, False))
# PS.update(pairstats(cells, cellnuc_metrics, struct_metrics, True))
# PS.update(
#     pairstats(
#         cells_COMP, cell_metrics_COMP_Linear_Nuc, struct_metrics_COMP_Linear_Nuc, True
#     )
# )
# PS.update(
#     pairstats(
#         cells_COMP, cell_metrics_COMP_Complex_Nuc, struct_metrics_COMP_Complex_Nuc, True
#     )
# )
# PS.update(
#     pairstats(
#         cells_COMP, nuc_metrics_COMP_Linear_Cell, struct_metrics_COMP_Linear_Cell, True
#     )
# )
# PS.update(
#     pairstats(
#         cells_COMP,
#         nuc_metrics_COMP_Complex_Cell,
#         struct_metrics_COMP_Complex_Cell,
#         True,
#     )
# )
#
# # %% write out cells
# cells_COMP.to_csv(data_root / "wf20200915" / "SizeScaling_20200828_Comp.csv")
#
# # %% Saving
# pfile = data_root / "wf20200915" / "SizeScaling_20200828_PairStats.pickle"
# if pfile.is_file():
#     pfile.unlink()
# with open(pfile, "wb") as f:
#     pickle.dump(PS, f)
#
# # %% Fin
