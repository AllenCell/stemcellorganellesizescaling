#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cbook, colors as mcolors
import numpy as np
import matplotlib
import statsmodels.api as sm
from sklearn.utils import resample
import datetime
from scipy.stats import gaussian_kde
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from matplotlib import cm
import pickle
import seaborn as sns
import os, platform
import sys, importlib

# Third party

# Relative
from stemcellorganellesizescaling.analyses.utils.scatter_plotting_func import ascatter

importlib.reload(
    sys.modules["stemcellorganellesizescaling.analyses.utils.scatter_plotting_func"]
)
from stemcellorganellesizescaling.analyses.utils.scatter_plotting_func import ascatter

print("Libraries loaded succesfully")
if platform.system() == "Linux":
    1 / 0
###############################################################################

log = logging.getLogger(__name__)

###############################################################################

# %% Simple function to load statistics
def loadps(pairstats_root, x):
    with open(pairstats_root / f"{x}.pickle", "rb") as f:
        result = pickle.load(f)
    return result


# %% Feature sets
FS = {}
FS["cellnuc_metrics"] = [
    "Cell surface area",
    "Cell volume",
    "Cell height",
    "Nuclear surface area",
    "Nuclear volume",
    "Nucleus height",
    "Cytoplasmic volume",
]

FS["struct_metrics"] = ["Structure volume"]
FS["COMP_types"] = ["AVH", "AV", "H"]

# %% Select columns for heatmap
HM = {}
HM["cellnuc_heatmap"] = [
    "Cell volume",
    "Cell surface area",
    "Nuclear volume",
    "Nuclear surface area",
    "Cytoplasmic volume",
]

HM["cellnuc_heatmap_RES_metrics"] = [
    "cell_AV",
    "cell_V",
    "cell_A",
    "nuc_AV",
    "nuc_V",
    "nuc_A",
]
HM["cellnuc_heatmap_RES_abbs"] = [
    "Cell v+a",
    "Cell vol",
    "Cell area",
    "Nuc v+a",
    "Nuc vol",
    "Nuc area",
]

HM["struct_heatmap_metrics"] = "Structure volume"
HM["COMP_type"] = "AV"
HM["LIN_type"] = "Linear"

# %% Annotation
ann_root = Path("Z:/modeling/theok/Projects/Data/scoss/Data/Oct2021")
structures = pd.read_csv(ann_root / "structure_annotated_20201113.csv")

# %%
# PlotMat = pd.DataFrame()

samplevec = [10, 20, 30, 50, 100, 200, 300, 500, 1000, 1500, -1]
# samplevec = [10, 20, 30, -1]
repeats = 10
# for s, sample in tqdm(enumerate(samplevec), "Sample number"):
rules = [None]*10
count_data = np.zeros((len(rules),len(samplevec)))
for s, sample in enumerate(samplevec):
    for r in range(0, repeats):
        if sample == -1:
            data_root = Path(f"Z:/modeling/theok/Projects/Data/scoss/Data/Oct2021/")
        else:
            data_root = Path(f"Z:/modeling/theok/Projects/Data/scoss/Data/Subsample_Oct2021/{sample}_{r}/")

        # %% Load dataset
        print(f'{sample}_{r}')
        tableIN = "SizeScaling_20211101.csv"
        table_compIN = "SizeScaling_20211101_comp.csv"
        statsIN = "Stats_20211101"
        # Load dataset
        cells = pd.read_csv(data_root / tableIN)
        if sample == -1:
            ScaleMat = pd.read_csv(data_root / "Scale_20211101" / "ScaleStats_20201125.csv")
            ScaleCurve = pd.read_csv(
                data_root / "Scale_20211101" / "ScaleCurve_20201125.csv"
            )
        else:
            ScaleMat = pd.read_csv(data_root / "Stats_20211101" / "ScaleStats_20201125.csv")
            ScaleCurve = pd.read_csv(
                data_root / "Stats_20211101" / "ScaleCurve_20201125.csv"
            )

        # %% Start dataframe
        CellNucGrow = pd.DataFrame()
        CellNucGrow["cellnuc_name"] = FS["cellnuc_metrics"]
        for i, col in enumerate(FS["cellnuc_metrics"]):
            CellNucGrow[col] = np.nan

        # %% Part 1 pairwise stats cell and nucleus measurement
        ps = data_root / statsIN / "cell_nuc_metrics"
        for xi, xlabel in enumerate(FS["cellnuc_metrics"]):
            for yi, ylabel in enumerate(FS["cellnuc_metrics"]):
                if xlabel is not ylabel:
                    val = loadps(ps, f"{xlabel}_{ylabel}_rs_vecL")
                    pred_yL = loadps(ps, f"{xlabel}_{ylabel}_pred_matL")
                    cmin = 100 * np.percentile(val, [50])
                    if pred_yL[0] > pred_yL[-1]:
                        cmin = -cmin
                    CellNucGrow.loc[
                        CellNucGrow["cellnuc_name"] == xlabel, ylabel
                    ] = cmin

        # %% Start dataframe
        StructGrow = pd.DataFrame()
        StructGrow["structure_name"] = cells["structure_name"].unique()

        # %% Part 2 pairwise stats cell and nucleus measurement
        ps = data_root / statsIN / "cellnuc_struct_metrics"
        for xi, xlabel in enumerate(FS["cellnuc_metrics"]):
            for yi, ylabel in enumerate(FS["struct_metrics"]):
                selected_structures = cells["structure_name"].unique()
                StructGrow[f"{xlabel}_{ylabel}"] = np.nan
                StructGrow[f"{xlabel}_{ylabel}_min"] = np.nan
                StructGrow[f"{xlabel}_{ylabel}_max"] = np.nan
                for si, struct in enumerate(selected_structures):
                    val = loadps(ps, f"{xlabel}_{ylabel}_{struct}_rs_vecL")
                    pred_yL = loadps(ps, f"{xlabel}_{ylabel}_{struct}_pred_matL")
                    cmin = 100 * np.percentile(val, [50])
                    cmin_min = 100 * np.percentile(val, [5])
                    cmin_max = 100 * np.percentile(val, [95])
                    if pred_yL[0] > pred_yL[-1]:
                        cmin = -cmin
                        cmin_min = -cmin_min
                        cmin_max = -cmin_max
                    StructGrow.loc[
                        StructGrow["structure_name"] == struct, f"{xlabel}_{ylabel}"
                    ] = cmin
                    StructGrow.loc[
                        StructGrow["structure_name"] == struct, f"{xlabel}_{ylabel}_min"
                    ] = cmin_min
                    StructGrow.loc[
                        StructGrow["structure_name"] == struct, f"{xlabel}_{ylabel}_max"
                    ] = cmin_max

        ps = data_root / statsIN / "struct_composite_metrics"
        for xi, xlabel in enumerate(
            ["cellnuc_AV", "cell_A", "cell_V", "cell_AV", "nuc_A", "nuc_V", "nuc_AV"]
        ):
            for yi, ylabel in enumerate(FS["struct_metrics"]):
                selected_structures = cells["structure_name"].unique()
                StructGrow[f"{xlabel}_{ylabel}"] = np.nan
                for si, struct in enumerate(selected_structures):
                    valA = loadps(ps, f"cellnuc_AV_{ylabel}_{struct}_rs_vecL")
                    val = loadps(ps, f"{xlabel}_{ylabel}_{struct}_rs_vecL")
                    if xlabel == "cellnuc_AV":
                        cmin = 100 * (np.percentile(valA, [50]))
                        cmin_min = 100 * np.percentile(valA, [5])
                        cmin_max = 100 * np.percentile(valA, [95])
                        StructGrow.loc[
                            StructGrow["structure_name"] == struct, f"{xlabel}_{ylabel}"
                        ] = cmin
                        StructGrow.loc[
                            StructGrow["structure_name"] == struct,
                            f"{xlabel}_{ylabel}_min",
                        ] = cmin_min
                        StructGrow.loc[
                            StructGrow["structure_name"] == struct,
                            f"{xlabel}_{ylabel}_max",
                        ] = cmin_max
                    else:
                        cmin = 100 * (
                            np.percentile(valA, [50]) - np.percentile(val, [50])
                        )
                        StructGrow.loc[
                            StructGrow["structure_name"] == struct, f"{xlabel}_{ylabel}"
                        ] = cmin

        # %% Make heatmap by selecting columns
        keepcolumns = ["structure_name"]
        keepcolumns_min = ["structure_name"]
        keepcolumns_max = ["structure_name"]
        for xi, xlabel in enumerate(HM["cellnuc_heatmap"]):
            struct_metric = HM["struct_heatmap_metrics"]
            keepcolumns.append(f"{xlabel}_{struct_metric}")
            keepcolumns_min.append(f"{xlabel}_{struct_metric}_min")
            keepcolumns_max.append(f"{xlabel}_{struct_metric}_max")

        HeatMap = StructGrow[keepcolumns]
        HeatMap_min = StructGrow[keepcolumns_min]
        HeatMap_max = StructGrow[keepcolumns_max]

        keepcolumns = ["structure_name"]
        struct_metric = HM["struct_heatmap_metrics"]
        for xi, xlabel in enumerate(HM["cellnuc_heatmap_RES_metrics"]):
            keepcolumns.append(f"{xlabel}_{struct_metric}")

        HeatMapComp = StructGrow[keepcolumns]

        keepcolumns = ["structure_name"]
        struct_metric = HM["struct_heatmap_metrics"]
        keepcolumns.append(f"cellnuc_AV_{struct_metric}")
        keepcolumns.append(f"cellnuc_AV_{struct_metric}_min")
        keepcolumns.append(f"cellnuc_AV_{struct_metric}_max")
        HeatMapAll = StructGrow[keepcolumns]

        # %% Annotation
        ann_st = structures[["Location", "Structure", "Gene"]].astype("category")
        ann_st = ann_st.to_numpy()
        color_st = structures[["Color"]]

        #%% Plot Arrays
        plot_array = HeatMap
        plot_array = plot_array.set_index(plot_array["structure_name"], drop=True)
        plot_array = plot_array.drop(["structure_name"], axis=1)
        plot_array = plot_array.reindex(list(ann_st[:, -1]))
        pan = plot_array.to_numpy()

        plot_array_min = HeatMap_min
        plot_array_min = plot_array_min.set_index(
            plot_array_min["structure_name"], drop=True
        )
        plot_array_min = plot_array_min.drop(["structure_name"], axis=1)
        plot_array_min = plot_array_min.reindex(list(ann_st[:, -1]))
        pan_min = plot_array_min.to_numpy()

        plot_array_max = HeatMap_max
        plot_array_max = plot_array_max.set_index(
            plot_array_max["structure_name"], drop=True
        )
        plot_array_max = plot_array_max.drop(["structure_name"], axis=1)
        plot_array_max = plot_array_max.reindex(list(ann_st[:, -1]))
        pan_max = plot_array_max.to_numpy()

        plot_arrayComp = HeatMapComp
        plot_arrayComp = plot_arrayComp.set_index(
            plot_arrayComp["structure_name"], drop=True
        )
        plot_arrayComp = plot_arrayComp.drop(["structure_name"], axis=1)
        plot_arrayComp = plot_arrayComp.reindex(list(ann_st[:, -1]))
        panComp = plot_arrayComp.to_numpy()

        plot_arrayAll = HeatMapAll
        plot_arrayAll = plot_arrayAll.set_index(
            plot_arrayAll["structure_name"], drop=True
        )
        plot_arrayAll = plot_arrayAll.drop(["structure_name"], axis=1)
        plot_arrayAll = plot_arrayAll.reindex(list(ann_st[:, -1]))
        panAll = plot_arrayAll.to_numpy()

        plot_arrayCN = CellNucGrow
        plot_arrayCN = plot_arrayCN.set_index(plot_arrayCN["cellnuc_name"], drop=True)
        plot_arrayCN = plot_arrayCN.drop(["cellnuc_name"], axis=1)
        plot_arrayCN = plot_arrayCN.reindex(HM["cellnuc_heatmap"])
        plot_arrayCN = plot_arrayCN[HM["cellnuc_heatmap"]]
        panCN = plot_arrayCN.to_numpy()
        for i in np.arange(panCN.shape[0]):
            for j in np.arange(panCN.shape[1]):
                if i == j:
                    panCN[i, j] = 100
                if i < j:
                    panCN[i, j] = 0

        # %% Pull in Grow numbers
        scalemodel = "prc"
        struct_metric = HM["struct_heatmap_metrics"]
        growvec = np.zeros((pan.shape[0], 3))
        for i, struct in enumerate(list(ann_st[:, -1])):
            gv = ScaleMat[f"{struct_metric}_{struct}{scalemodel}"].to_numpy()
            growvec[i, 0] = np.percentile(gv, 50)
            growvec[i, 1] = np.percentile(gv, 5)
            growvec[i, 2] = np.percentile(gv, 95)

        growvecC = np.zeros((len(HM["cellnuc_heatmap"]), 1))
        for i, struct in enumerate(HM["cellnuc_heatmap"]):
            gv = ScaleMat[f"{struct}_{scalemodel}"].to_numpy()
            growvecC[i, 0] = np.percentile(gv, 50)

        # # %%
        # print(np.expand_dims(growvec[:, 0],axis=1).shape)
        # print(growvecC.shape)
        # print(plot_array.shape)
        # print(plot_array.to_numpy().flatten().shape)
        # print(plot_arrayCN.shape)
        # print(plot_arrayAll.shape)
        # print(plot_arrayComp.shape)

        # %% Here come the rules, but first to gather all the important numbers
        OrgScale = pd.DataFrame(data=growvec[:, 0], index=plot_array.index , columns=['ScalingRate'])
        CN_Scale = pd.DataFrame(data=growvecC, index=plot_arrayCN.index , columns=['ScalingRate'])
        OrgFit = plot_array
        CN_Fit = plot_arrayCN
        AllFit = plot_arrayAll['cellnuc_AV_Structure volume']
        Un_Fit = plot_arrayComp

        # %% Make
        rule_count = 0

        # %% Rule 1
        #Explained variance of structure volume using all metrics
        #Centrioles and endosomes below 25% - all other structures above 25%
        rules[rule_count] = f'Explained variance of structure volume using all metrics is below 25\% for centrioles and endosomes - and above 25\% for all other structures'
        Rule = True
        for i, v in AllFit.iteritems():
            if i == 'CETN2' or i == 'RAB5A':
                if v > 25:
                    Rule = False
            else: #other organelles
                if v < 25:
                    Rule = False
        if Rule is True:
            count_data[rule_count,s] +=1
        # print(f'Rule {rule_count+1} is {Rule}')
        rule_count +=1

        # %% Rule 2
        #Discarding nuclear envelope and plasma membrane, Top 3 is always nucleoli (GC, DFC) and speckles
        rules[rule_count] = f'Discarding nuclear envelope and plasma membrane, Top 3 is always nucleoli (GC, DFC) and speckles'
        Rule2_frame = AllFit.copy()
        Rule2_frame = Rule2_frame.drop(['LMNB1','AAVS1'])
        Rule2_frame.sort_values(ascending=False,inplace=True)
        Top3 = Rule2_frame.index[0:3].values.tolist()
        Top3_exp = ['FBL','NPM1','SON']
        if len(list(set(Top3) & set(Top3_exp))) == 3:
            Rule = True
        else:
            Rule = False
        if Rule is True:
            count_data[rule_count,s] +=1
        # print(f'Rule {rule_count+1} is {Rule}')
        rule_count +=1

        # %% Rule 3
        #Golgi, microtubules, mitochondria, lysosomes in the top 4.
        rules[rule_count] = f'Discarding nuclear envelope and plasma membrane, nucleoli (GC, DFC) and speckles (Top 5) the next organelles are Golgi, microtubules, mitochondria, lysosomes'
        Rule3_frame = AllFit.copy()
        Rule3_frame = Rule3_frame.drop(['LMNB1','AAVS1','FBL','NPM1','SON'])
        Rule3_frame.sort_values(ascending=False,inplace=True)
        Top4 = Rule3_frame.index[0:4].values.tolist()
        Top4_exp = ['TOMM20','TUBA1B','ST6GAL1','LAMP1']
        if len(list(set(Top4) & set(Top4_exp))) == 4:
            Rule = True
        else:
            Rule = False
        if Rule is True:
            count_data[rule_count,s] +=1
        # print(f'Rule {rule_count+1} is {Rule}')
        rule_count +=1

        # %% Rule 4
        #Unique explained variances: Nuclear V+A higher than Cell V+A for nuclear organelles
        rules[rule_count] = f'Unique explained variances: Nuclear V+A higher than Cell V+A for all nuclear organelles'
        Rule4_frame = Un_Fit.copy()
        Rule4_frame = Rule4_frame.drop(['SEC61B','ATP2A2','TOMM20','SLC25A17','RAB5A','LAMP1','ST6GAL1','CETN2','TUBA1B','AAVS1'])
        Nvec = Rule4_frame['nuc_AV_Structure volume'].values
        Cvec = Rule4_frame['cell_AV_Structure volume'].values
        if np.all(Nvec>Cvec):
            Rule = True
        else:
            Rule = False
        if Rule is True:
            count_data[rule_count,s] +=1
        # print(f'Rule {rule_count+1} is {Rule}')
        rule_count +=1

        # %% Rule 5
        rules[rule_count] = f'For Nuclear envelope the unique explained variance is higher for nuc. area than for nuc. vol.'
        Rule5_frame = Un_Fit.copy()
        Nscalar = Rule5_frame.at['LMNB1','nuc_A_Structure volume']
        Vscalar = Rule5_frame.at['LMNB1','nuc_V_Structure volume']
        if Nscalar > Vscalar:
            Rule = True
        else:
            Rule = False
        if Rule is True:
            count_data[rule_count,s] +=1
        # print(f'Rule {rule_count+1} is {Rule}')
        rule_count +=1

        # %% Rule 6
        rules[rule_count] = f'For Nuclear speckles the unique explained variance is higher for nuc. area than for nuc. vol.'
        Rule6_frame = Un_Fit.copy()
        Nscalar = Rule6_frame.at['SON','nuc_A_Structure volume']
        Vscalar = Rule6_frame.at['SON','nuc_V_Structure volume']
        if Nscalar > Vscalar:
            Rule = True
        else:
            Rule = False
        if Rule is True:
            count_data[rule_count,s] +=1
        # print(f'Rule {rule_count+1} is {Rule}')
        rule_count +=1

        # %% Rule 7
        rules[rule_count] = f'Cell area unique variance is always lower than 5% for all structures'
        Rule7_frame = Un_Fit.copy()
        Avec = Rule7_frame['cell_A_Structure volume'].values
        if np.all(Avec<5):
            Rule = True
        else:
            Rule = False
        if Rule is True:
            count_data[rule_count,s] +=1
        # print(f'Rule {rule_count+1} is {Rule}')
        rule_count +=1

        # %% Rule 8
        rules[rule_count] = f'Scaling metrics: Peroxisomes are always in top 3'
        Rule8_frame = OrgScale.copy()
        Rule8_frame.sort_values(ascending=False,inplace=True,by='ScalingRate')
        Top3 = Rule8_frame.index[0:3].values.tolist()
        if 'SLC25A17' in Top3:
            Rule = True
        else:
            Rule = False
        if Rule is True:
            count_data[rule_count,s] +=1
        # print(f'Rule {rule_count+1} is {Rule}')
        rule_count +=1

        # %% Rule 9
        rules[rule_count] = f'Nucleoli and microtubules are above 80%'
        Rule9_frame = OrgScale.copy()
        Rule9_frame.loc[['FBL','NPM1','TUBA1B'],'ScalingRate']
        if np.all(Rule9_frame.loc[['FBL','NPM1','TUBA1B'],'ScalingRate'].values>80):
            Rule = True
        else:
            Rule = False
        if Rule is True:
            count_data[rule_count,s] +=1
        rule_count +=1

        # %% Rule 10
        rules[rule_count] = f'Centrioles, endosomes always in bottom 2'
        Rule10_frame = OrgScale.copy()
        Rule10_frame.sort_values(ascending=True,inplace=True,by='ScalingRate')
        Bot2 = Rule10_frame.index[0:2].values.tolist()
        Bot2_exp = ['RAB5A', 'CETN2']
        if len(list(set(Bot2) & set(Bot2_exp))) == 2:
            Rule = True
        else:
            Rule = False
        if Rule is True:
            count_data[rule_count,s] +=1
        rule_count +=1



#%% Make dataset
ResMat = pd.DataFrame(data=count_data, index = rules, columns=samplevec)

# %%

sns.heatmap(ResMat, annot=True)
plt.show()

# %%
ResMat2 = ResMat.to_numpy()
fig = plt.figure(figsize=(16.4, 8))
fs = 10
ax = fig.add_axes([0.7, 0.1, 0.3, 0.8])
ax.imshow(
    ResMat2,
    aspect="auto",
    cmap="Greens",
    vmin=0,
    vmax=10,
)

for i in range(ResMat2.shape[0]):
    for j in range(ResMat2.shape[1]):
        val = np.int(ResMat2[i, j])
        if abs(val) > 7:
            text = ax.text(
                j,
                i,
                val,
                ha="center",
                va="center",
                color="w",
                fontsize=fs,
            )
        else:
            text = ax.text(
                j,
                i,
                val,
                ha="center",
                va="center",
                color="k",
                fontsize=fs,
            )
ylabs = ResMat.index.values
ax.set_yticks(np.arange(len(ylabs)))
ax.set_yticklabels(ylabs.tolist())
xlabs = ResMat.columns
ax.set_xticks(np.arange(len(xlabs)))
ax.set_xticklabels(xlabs.tolist())

plt.show()

# %%

#
#
# NumberVec = np.concatenate(
#             (
#                 np.expand_dims(growvec[:, 0], axis=1),
#                 growvecC,
#                 np.expand_dims(plot_array.to_numpy().flatten(), axis=1),
#                 np.expand_dims(
#                     plot_arrayCN.to_numpy()[np.tril_indices(5, k=-1)], axis=1
#                 ),
#                 np.expand_dims(plot_arrayAll.to_numpy()[:, 0], axis=1),
#                 np.expand_dims(plot_arrayComp.to_numpy().flatten(), axis=1),
#             ),
#             axis=0,
#         )
#         if sample == -1:
#             PlotMat[f"All"] = NumberVec.squeeze()
#             break
#         else:
#             PlotMat[f"{sample}_{r}"] = NumberVec.squeeze()
#
# # %%
# # xvec = [10, 20, 30, 50, 100, 200, 300, 500, 1000, 1500]
# # repeats = 3
#
# xvec = [10, 20]
# repeats = 2
#
# yvec = np.zeros((len(xvec), 2))
# yerr = np.zeros((len(xvec), 2))
# metric_org = PlotMat["All"].to_numpy()
# var_th = 5
# for x, xv in enumerate(xvec):
#     scores = np.zeros((repeats, 2))
#     for r in range(0, repeats):
#         metric_i = PlotMat[f"{xv}_{r}"].to_numpy()
#         scores[r, 0] = np.sqrt(np.mean((metric_org - metric_i) ** 2))
#         scores[r, 1] = np.sum((np.abs(metric_org - metric_i)) > var_th)
#     yvec[x, 0] = np.mean(scores[:, 0])
#     yvec[x, 1] = np.mean(scores[:, 1])
#     yerr[x, 0] = np.std(scores[:, 0])
#     yerr[x, 1] = np.std(scores[:, 1])
#
# # %% plot
# plt.rcParams["svg.fonttype"] = "none"
# fig, ax = plt.subplots(figsize=(7, 7))
# ax.errorbar(xvec, yvec[:, 0], yerr=yerr[:, 0], color="red")
# ax.set_ylabel("RMS", color="red", fontsize=14)
# ax.set_xticks(xvec)
# ax.grid()
# ax.set_ylim(bottom=0)
# ax2 = ax.twinx()
# ax2.errorbar(xvec, yvec[:, 1], yerr=yerr[:, 1], color="blue")
# ax2.set_ylabel("# cases with large differences (out of 210)", color="blue", fontsize=14)
# ax2.set_ylim(bottom=0)
#
# # Resolve directories
# pic_root = pic_root = Path("Z:/modeling/theok/Projects/Data/scoss/Pics/Oct2021")
# pic_rootT = pic_root / "subsampling"
# pic_rootT.mkdir(exist_ok=True)
#
# plot_save_path = pic_rootT / f"SubsamplingSizeScaling_20220103_v1.svg"
# plt.savefig(plot_save_path, format="svg")
#
# plt.show()
