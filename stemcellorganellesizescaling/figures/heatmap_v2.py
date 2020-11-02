#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
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

# Third party

# Relative
from organelle_size_scaling.utils import regression_20200907

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

save_flag = 0
plt.rcParams.update({"font.size": 8})

# %% Start

# Resolve directories
data_root = dirs[0]
pic_root = dirs[1]

tableIN = "SizeScaling_20201012.csv"
table_compIN = "SizeScaling_20201012_comp.csv"
statsIN = "Stats_20201012"
# Load dataset
cells = pd.read_csv(data_root / tableIN)
print(np.any(cells.isnull()))
cells_COMP = pd.read_csv(data_root / table_compIN)
print(np.any(cells_COMP.isnull()))
structures = pd.read_csv(data_root / 'annotation' / "structure_annotated_20201019.csv")
Grow = pd.read_csv(data_root / 'growing' / "Growthstats_20201012.csv")
print(np.any(cells_COMP.isnull()))

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
FS["cellnuc_abbs"] = [
    "Cell area",
    "Cell vol",
    "Cell height",
    "Nuclear area",
    "Nuclear vol",
    "Nucleus height",
    "Cyto vol",
]
FS["cellnuc_COMP_metrics"] = [
    "Cell surface area",
    "Cell volume",
    "Cell height",
    "Nuclear surface area",
    "Nuclear volume",
    "Nucleus height",
]

FS["cellnuc_COMP_abbs"] = [
    "Cell area *N",
    "Cell vol *N",
    "Cell height *N",
    "Nuclear area *C",
    "Nuclear vol *C",
    "Nucleus height *C",
]

FS["struct_metrics"] = ["Structure volume", "Number of pieces"]
FS["COMP_types"] = ["AVH","AV","H"]

# %% Start dataframe
CellNucGrow = pd.DataFrame()
CellNucGrow['cellnuc_name'] = FS["cellnuc_metrics"]
for i, col in enumerate(FS["cellnuc_metrics"]):
    CellNucGrow[col] = np.nan

# %% Part 1 pairwise stats cell and nucleus measurement
print('Cell and nucleus metrics')
ps = (data_root / statsIN / 'cell_nuc_metrics')
for xi, xlabel in enumerate(FS['cellnuc_metrics']):
    for yi, ylabel in enumerate(FS['cellnuc_metrics']):
        if xlabel is not ylabel:
            print(f"{xlabel} vs {ylabel}")
            val = loadps(ps, f"{xlabel}_{ylabel}_rs_vecL")
            pred_yL = loadps(ps, f"{xlabel}_{ylabel}_pred_matL")
            cmin = np.round(100*np.percentile(val,[50]))
            if pred_yL[0] > pred_yL[-1]:
                cmin = -cmin
            CellNucGrow.loc[CellNucGrow['cellnuc_name']==xlabel,ylabel] = cmin

# %% Start dataframe
StructGrow = pd.DataFrame()
StructGrow['structure_name'] = cells['structure_name'].unique()

# %% Part 2 pairwise stats cell and nucleus measurement
print('Cell and nucleus metrics vs structure metrics')
ps = (data_root / statsIN / 'cellnuc_struct_metrics')
for xi, xlabel in enumerate(FS['cellnuc_metrics']):
    for yi, ylabel in enumerate(FS['struct_metrics']):
        print(f"{xlabel} vs {ylabel}")
        selected_structures = cells["structure_name"].unique()
        StructGrow[f"{xlabel}_{ylabel}"] = np.nan
        StructGrow[f"{xlabel}_{ylabel}_min"] = np.nan
        StructGrow[f"{xlabel}_{ylabel}_max"] = np.nan
        for si, struct in enumerate(selected_structures):
            val = loadps(ps, f"{xlabel}_{ylabel}_{struct}_rs_vecC")
            pred_yL = loadps(ps, f"{xlabel}_{ylabel}_{struct}_pred_matC")
            cmin = np.round(100 * np.percentile(val, [50]))
            cmin_min = np.round(100 * np.percentile(val, [5]))
            cmin_max = np.round(100 * np.percentile(val, [95]))
            if pred_yL[0] > pred_yL[-1]:
                cmin = -cmin
                cmin_min = -cmin_min
                cmin_max = -cmin_max
            StructGrow.loc[StructGrow['structure_name'] == struct, f"{xlabel}_{ylabel}"] = cmin
            StructGrow.loc[StructGrow['structure_name'] == struct, f"{xlabel}_{ylabel}_min"] = cmin_min
            StructGrow.loc[StructGrow['structure_name'] == struct, f"{xlabel}_{ylabel}_max"] = cmin_max

ps = (data_root / statsIN / 'cellnuc_struct_COMP_metrics')
comp_columns = list(cells_COMP.columns)
for xi, xlabel in enumerate(
    ['nuc_metrics_AVH', 'nuc_metrics_AV', 'nuc_metrics_H', 'cell_metrics_AVH', 'cell_metrics_AV',
     'cell_metrics_H']):
    for zi, zlabel in enumerate(FS['cellnuc_metrics']):
        for ti, type in enumerate(["Linear", "Complex"]):
            col2 = f"{zlabel}_COMP_{type}_{xlabel}"
            if col2 in comp_columns:
                print(col2)
                for yi, ylabel in enumerate(FS['struct_metrics']):
                    selected_structures = cells_COMP["structure_name"].unique()
                    col1 = f"{ylabel}_COMP_{type}_{xlabel}"
                    StructGrow[f"{zlabel}_{col1}"] = np.nan
                    for si, struct in enumerate(selected_structures):
                        val = loadps(ps, f"{col2}_{col1}_{struct}_rs_vecL")
                        pred_yL = loadps(ps, f"{col2}_{col1}_{struct}_pred_matL")
                        cmin = np.round(100 * np.percentile(val, [50]))
                        if pred_yL[0] > pred_yL[-1]:
                            cmin = -cmin
                        StructGrow.loc[StructGrow['structure_name'] == struct, f"{zlabel}_{col1}"] = cmin

# %% Select columns for heatmap
HM = {}
HM["cellnuc_heatmap"] = [
    "Cell volume",
    "Cell surface area",
    "Cell height",
    "Nuclear volume",
    "Nuclear surface area",
    "Nucleus height",
    "Cytoplasmic volume",
]
HM["cellnuc_heatmap_abbs"] = [
    "Cell vol",
    "Cell area",
    "Cell height",
    "Nuc vol",
    "Nuc area",
    "Nuc height",
    "Cyto vol",
]
HM["cellnuc_heatmap_COMP_metrics"] = [
    "Cell volume",
    "Cell surface area",
    "Cell height",
    "Nuclear volume",
    "Nuclear surface area",
    "Nucleus height",
]

HM["cellnuc_COMP_abbs"] = [
    "Cell vol *N",
    "Cell area *N",
    "Cell height *N",
    "Nuc vol *C",
    "Nuc area *C",
    "Nuc height *C",
]

HM["struct_heatmap_metrics"] = "Structure volume"
HM["COMP_type"] = "AVH"
HM["LIN_type"] = "Complex"

# %% Make heatmap by selecting columns
keepcolumns = ['structure_name']
keepcolumns_min = ['structure_name']
keepcolumns_max = ['structure_name']
for xi, xlabel in enumerate(HM['cellnuc_heatmap']):
    struct_metric = HM["struct_heatmap_metrics"]
    keepcolumns.append(f"{xlabel}_{struct_metric}")
    keepcolumns_min.append(f"{xlabel}_{struct_metric}_min")
    keepcolumns_max.append(f"{xlabel}_{struct_metric}_max")

HeatMap = StructGrow[keepcolumns]
HeatMap_min = StructGrow[keepcolumns_min]
HeatMap_max = StructGrow[keepcolumns_max]

keepcolumns = ['structure_name']
for xi, xlabel in enumerate(HM['cellnuc_heatmap_COMP_metrics']):
    struct_metric = HM["struct_heatmap_metrics"]
    lin_type = HM["LIN_type"]
    comp_type = HM["COMP_type"]
    if str(xlabel).startswith('Cell'):
        keepcolumns.append(f"{xlabel}_{struct_metric}_COMP_{lin_type}_nuc_metrics_{comp_type}")
    elif str(xlabel).startswith('Nuc'):
        keepcolumns.append(f"{xlabel}_{struct_metric}_COMP_{lin_type}_cell_metrics_{comp_type}")
    else:
        1/0

HeatMapComp = StructGrow[keepcolumns]

# %% Annotation
ann_st = structures[[ 'Structure', 'Group', 'Gene']].astype('category')
cat_columns = ann_st.select_dtypes(['category']).columns
num_st = pd.DataFrame()
num_st[cat_columns] = ann_st[cat_columns].apply(lambda x: x.cat.codes)
num_st[['Gene', 'Group', 'Structure']] = np.nan
ann_st = ann_st.to_numpy()
color_st = structures[['Color']]

# %% plot function
# def heatmap(selected_metrics, cells, num_st, ann_st, structures, struct_metric, COMP_flag, LIN_flag, xlabels, save_flag, pic_root):
# Parameters and settings
lw = 1
mstrlength = 10
fs = 10

# %% measurements
w1 = -1
w2 = 0.01
w3 = 0.05
w4 = 0.02
w5 = 0.01
w6 = 0.01
w7 = 0.01
w8 = 0.01
w9 = 0.01
w10 = 0.05

x1 = 0.2
x2s = 0.05
w1 = w6+x2s
x2 = x1+w2+x1
x3s = 0.03
x3 = (1-(w1+x1+w2+x1+w3+x3s+w4+x3s+w5))/2
x4 = 0.2
x5 = 0.03
x6 = 0.2
x7 = 0.2
x8 = 1-w6-x4-w7-x5-w8-x6-w9-x7-w10-w5
xa =x2


h1 = 0.02
h2 = 0.02
h3 = 0.06
h4 = 0.06
h5 = 0.02

y1 =  0
ya = 0
y2s = 0
y2 = 0
y3s = 0
y3 = ((h1+y1+ya+y2s+y2)-(h1+y3s+h2+y3s))/2
y4 = 0.6
y5 = 1-(h1+y1+ya+y2s+y2+h3+y4+h5)
y6 = 1-(h1+y1+ya+y2s+y2+h4+h5)
yh = h1+y1+ya+y2s+y2
yh = 0

#%% Plot Array
plot_array = HeatMap
plot_array = plot_array.set_index(plot_array['structure_name'],drop=True)
plot_array = plot_array.drop(['structure_name'],axis=1)
plot_array = plot_array.reindex(list(ann_st[:,-1]))
pan = plot_array.to_numpy()

plot_array_min = HeatMap_min
plot_array_min = plot_array_min.set_index(plot_array_min['structure_name'],drop=True)
plot_array_min = plot_array_min.drop(['structure_name'],axis=1)
plot_array_min = plot_array_min.reindex(list(ann_st[:,-1]))
pan_min = plot_array_min.to_numpy()

plot_array_max = HeatMap_max
plot_array_max = plot_array_max.set_index(plot_array_max['structure_name'],drop=True)
plot_array_max = plot_array_max.drop(['structure_name'],axis=1)
plot_array_max = plot_array_max.reindex(list(ann_st[:,-1]))
pan_max = plot_array_max.to_numpy()

plot_arrayComp = HeatMapComp
plot_arrayComp = plot_arrayComp.set_index(plot_arrayComp['structure_name'],drop=True)
plot_arrayComp = plot_arrayComp.drop(['structure_name'],axis=1)
plot_arrayComp = plot_arrayComp.reindex(list(ann_st[:,-1]))
panComp = plot_arrayComp.to_numpy()

plot_arrayCN = CellNucGrow
plot_arrayCN = plot_arrayCN.set_index(plot_arrayCN['cellnuc_name'],drop=True)
plot_arrayCN = plot_arrayCN.drop(['cellnuc_name'],axis=1)
plot_arrayCN = plot_arrayCN.reindex(HM["cellnuc_heatmap"])
plot_arrayCN = plot_arrayCN[HM["cellnuc_heatmap"]]
panCN = plot_arrayCN.to_numpy()
for i in np.arange(panCN.shape[0]):
    for j in np.arange(panCN.shape[1]):
        if i==j:
            panCN[i,j] = 100
        if i<j:
            panCN[i, j] = 0

# %% Pull in Grow numbers
struct_metric = HM["struct_heatmap_metrics"]
growvec = np.zeros((pan.shape[0], 3))
for i, struct in enumerate(list(ann_st[:,-1])):
    growvec[i,0] = Grow.loc[50, f"{struct_metric}_{struct}_50"]
    growvec[i, 1] = Grow.loc[51, f"{struct_metric}_{struct}_25"]
    growvec[i, 2] = Grow.loc[51, f"{struct_metric}_{struct}_75"]

growvecC = np.zeros((len(HM["cellnuc_heatmap"]), 1))
for i, struct in enumerate(HM["cellnuc_heatmap"]):
    growvecC[i] = Grow.loc[50, f"{struct}_mean"]
growvecC[0]=100

# %% other parameters
rot = -20
alpha = 0.5
lw=5


# %%layout
fig = plt.figure(figsize=(16, 9))
plt.rcParams.update({"font.size": 10})

# Annotation
axAnn = fig.add_axes([w6, yh+h3, x4, y4])
# Annotation
axAnn.imshow(num_st, aspect='auto', cmap='Pastel1')
for i in range(len(num_st)):
    for j in range(len(num_st.columns)):
        string = ann_st[i, j]
        if len(string) > mstrlength:
            string = string[0:mstrlength]
        text = axAnn.text(j, i, string,
                          ha="center", va="center", color=color_st.loc[i,'Color'], fontsize=fs)
axAnn.axis('off')

# Organelle Growth rates
axOrgGrow = fig.add_axes([w6+x4+w7, yh+h3, x5, y4])
axOrgGrow.imshow(np.expand_dims(growvec[:,0],axis=0).T, aspect='auto', cmap='viridis',vmin=0, vmax=100)
for i in range(len(growvec[:,0])):
        val = np.int(growvec[i, 0])
        text = axOrgGrow.text(0, i, val,
                          ha="center", va="center", color="w", fontsize=fs, fontweight='bold')
axOrgGrow.set_yticks([])
axOrgGrow.set_yticklabels([])
xlabels = ['Grow %']
axOrgGrow.set_xticks(range(len(xlabels)))
axOrgGrow.set_xticklabels(xlabels,rotation=rot,horizontalalignment='left')

# Cell Growth rates
axCellGrow = fig.add_axes([w6+x4+w7, yh+h3+y4, x5, y5])
axCellGrow.imshow(growvecC, aspect='auto', cmap='viridis',vmin=0, vmax=100)
for i in range(len(growvecC)):
        val = np.int(growvecC[i, 0])
        text = axCellGrow.text(0, i, val,
                          ha="center", va="center", color="w", fontsize=fs, fontweight='bold')
axCellGrow.set_yticks(range(len(HM["cellnuc_heatmap"])))
axCellGrow.set_yticklabels(HM["cellnuc_heatmap"])
axCellGrow.set_xticks([])
axCellGrow.set_xticklabels([])

# Organelle Variance rates
axOrgVar = fig.add_axes([w6+x4+w7+x5+w8, yh+h3, x6, y4])
axOrgVar.imshow(pan, aspect='auto', cmap='seismic',vmin=-100, vmax=100)
for i in range(len(plot_array)):
    for j in range(len(plot_array.columns)):
        val = np.int(pan[i, j])
        text = axOrgVar.text(j, i, val,
                          ha="center", va="center", color="w", fontsize=fs, fontweight='bold')
axOrgVar.set_yticks([])
axOrgVar.set_yticklabels([])
xlabels = HM["cellnuc_heatmap_abbs"]
axOrgVar.set_xticks(range(len(xlabels)))
axOrgVar.set_xticklabels(xlabels,rotation=rot,horizontalalignment='left')

# Cell Variance rates
axCellVar = fig.add_axes([w6+x4+w7+x5+w8, yh+h3+y4, x6, y5])
axCellVar.imshow(panCN, aspect='auto', cmap='seismic',vmin=-100, vmax=100)
for i in range(len(plot_arrayCN)):
    for j in range(len(plot_arrayCN.columns)):
        val = np.int(panCN[i, j])
        text = axCellVar.text(j, i, val,
                          ha="center", va="center", color="w", fontsize=fs, fontweight='bold')
axCellVar.set_yticks([])
axCellVar.set_yticklabels([])
axCellVar.set_xticks([])
axCellVar.set_xticklabels([])
axCellVar.axis('off')

# Cell Comp rates
axCompVar= fig.add_axes([w6+x4+w7+x5+w8+x6+w9, yh+h3, x7, y4])
axCompVar.imshow(panComp, aspect='auto', cmap='seismic',vmin=-100, vmax=100)
for i in range(len(plot_arrayComp)):
    for j in range(len(plot_arrayComp.columns)):
        val = np.int(panComp[i, j])
        text = axCompVar.text(j, i, val,
                          ha="center", va="center", color="w", fontsize=fs, fontweight='bold')
axCompVar.set_yticks([])
axCompVar.set_yticklabels([])
xlabels = HM["cellnuc_COMP_abbs"]
axCompVar.set_xticks(range(len(xlabels)))
axCompVar.set_xticklabels(xlabels,rotation=rot,horizontalalignment='left')
axCompVar.set_title('Residual variance')

# GrowVarS
axGrowVarS= fig.add_axes([w6+x4+w7+x5+w8+x6+w9+x7+w10, yh+h4, x8, y6])
for i in np.arange(len(growvec)):
    growval = growvec[i,0]
    growmin = growvec[i, 1]
    growmax = growvec[i, 2]
    struct = ann_st[i, 2]
    if (struct == 'SON') or (struct == 'ATP2A2'):
        va = 'top'
    else:
        va = 'baseline'
    if (struct == 'HIST1H2BJ'):
        ha = 'right'
    else:
        ha = 'left'
    axGrowVarS.plot(pan[i,0],growval,'.',markersize=10,color=color_st.loc[i,'Color'])
    axGrowVarS.plot([pan_min[i, 0], pan_max[i, 0]], [growval, growval], color=color_st.loc[i, 'Color'], alpha=alpha,linewidth=lw)
    axGrowVarS.plot([pan[i, 0], pan[i, 0]], [growmin, growmax], color=color_st.loc[i, 'Color'], alpha=alpha,linewidth=lw)
    axGrowVarS.text(pan[i, 0],growval, struct, fontsize=16, color=color_st.loc[i, 'Color'], verticalalignment=va,fontweight='bold',horizontalalignment=ha)
axGrowVarS.set_ylim(bottom=0,top=125)
axGrowVarS.set_xlim(left=0, right=100)
axGrowVarS.set_xlabel('Variance of structure volume explained by cell volume (%)')
axGrowVarS.set_ylabel('Relative structure growth rate compared to cell volume growth (%)')
axGrowVarS.grid()

axExpVarBar= fig.add_axes([w6+x4+w7+x5+w8+x6+w9-0.05, yh+h3+y4+.25, x7+.03, .05])
axExpVarBar.imshow(np.expand_dims(np.linspace(-100,100,201),axis=0), aspect='auto', cmap='seismic',vmin=-100, vmax=100)
text = axExpVarBar.text(0, 0, '-100', ha="left", va="center", color="w", fontsize=fs, fontweight='bold')
text = axExpVarBar.text(200, 0, '100', ha="right", va="center", color="w", fontsize=fs, fontweight='bold')
text = axExpVarBar.text(100, 0, '0', ha="center", va="center", color="k", fontsize=fs, fontweight='bold')
axExpVarBar.set_yticks([])
axExpVarBar.set_yticklabels([])
axExpVarBar.set_xticks([50, 150])
axExpVarBar.set_xticklabels(['Negative correlation', 'Positive correlation'])
axExpVarBar.set_title('Explained Variance (%)')

axGrowBar= fig.add_axes([w6+x4+w7+x5+w8+x6+w9-0.05, yh+h3+y4+.12, x7+.03, .05])
axGrowBar.imshow(np.expand_dims(np.linspace(0,100,101),axis=0), aspect='auto', cmap='viridis',vmin=0, vmax=100)
text = axGrowBar.text(0, 0, '0', ha="left", va="center", color="w", fontsize=fs, fontweight='bold')
text = axGrowBar.text(50, 0, '50', ha="center", va="center", color="w", fontsize=fs, fontweight='bold')
text = axGrowBar.text(100, 0, '100', ha="right", va="center", color="w", fontsize=fs, fontweight='bold')
axGrowBar.set_yticks([])
axGrowBar.set_yticklabels([])
axGrowBar.set_xticks([])
axGrowBar.set_xticklabels([])
axGrowBar.set_title('Growth rate (%)')

plot_save_path = pic_root / f"heatmap_20201022_complex.png"
plt.savefig(plot_save_path, format="png", dpi=1000)
plt.close()

# plt.show()

# %%



