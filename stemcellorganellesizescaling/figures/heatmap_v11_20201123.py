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
###############################################################################

log = logging.getLogger(__name__)

###############################################################################
#%% Directories
if platform.system() == "Windows":
    data_root = Path("E:/DA/Data/scoss/Data/Nov2020/")
    pic_root = Path("E:/DA/Data/scoss/Pics/Nov2020/")
elif platform.system() == "Linux":

    data_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Data/2020/")
    pic_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Pics/2020/")
dirs = []
dirs.append(data_root)
dirs.append(pic_root)

save_flag = 0
plt.rcParams.update({"font.size": 8})

# %% Start

# Resolve directories
data_root = dirs[0]
pic_root = dirs[1]

tableIN = "SizeScaling_20201102.csv"
table_compIN = "SizeScaling_20201102_comp.csv"
statsIN = "Stats_20201102"
# Load dataset
cells = pd.read_csv(data_root / tableIN)
print(np.any(cells.isnull()))
cells_COMP = pd.read_csv(data_root / table_compIN)
print(np.any(cells_COMP.isnull()))
structures = pd.read_csv(data_root / 'annotation' / "structure_annotated_20201113.csv")
Grow = pd.read_csv(data_root / 'growing' / "Growthstats_20201102.csv")
ScaleMat = pd.read_csv(data_root / 'growing' / "ScaleStats_20201118.csv")
ScaleCurve = pd.read_csv(data_root / 'growing' / "ScaleCurve_20201118.csv")
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
            val = loadps(ps, f"{xlabel}_{ylabel}_{struct}_rs_vecL")
            pred_yL = loadps(ps, f"{xlabel}_{ylabel}_{struct}_pred_matL")
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

# ps = (data_root / statsIN / 'struct_composite_metrics_bu')
# for xi, xlabel in enumerate(
#         ['nuc_metrics_AVH', 'nuc_metrics_AV', 'nuc_metrics_H', 'cell_metrics_AVH', 'cell_metrics_AV',
#          'cell_metrics_H']):
#     for yi, ylabel in enumerate(FS['struct_metrics']):
#         selected_structures = cells["structure_name"].unique()
#         StructGrow[f"{xlabel}_{ylabel}"] = np.nan
#         for si, struct in enumerate(selected_structures):
#             val = loadps(ps, f"{xlabel}_{ylabel}_{struct}_rs_vecL")
#             cmin = np.round(100 * np.percentile(val, [50]))
#             StructGrow.loc[StructGrow['structure_name'] == struct, f"{xlabel}_{ylabel}"] = cmin

ps = (data_root / statsIN / 'struct_composite_metrics')
for xi, xlabel in enumerate(
    ['cellnuc_AV','cell_A', 'cell_V', 'cell_AV', 'nuc_A', 'nuc_V', 'nuc_AV']):
    for yi, ylabel in enumerate(FS['struct_metrics']):
        selected_structures = cells["structure_name"].unique()
        StructGrow[f"{xlabel}_{ylabel}"] = np.nan
        for si, struct in enumerate(selected_structures):
            valA = loadps(ps, f"cellnuc_AV_{ylabel}_{struct}_rs_vecL")
            val = loadps(ps, f"{xlabel}_{ylabel}_{struct}_rs_vecL")
            if xlabel == 'cellnuc_AV':
                cmin = np.round(100 * (np.percentile(valA, [50])))
                cmin_min = np.round(100 * np.percentile(valA, [5]))
                cmin_max = np.round(100 * np.percentile(valA, [95]))
                StructGrow.loc[StructGrow['structure_name'] == struct, f"{xlabel}_{ylabel}"] = cmin
                StructGrow.loc[StructGrow['structure_name'] == struct, f"{xlabel}_{ylabel}_min"] = cmin_min
                StructGrow.loc[StructGrow['structure_name'] == struct, f"{xlabel}_{ylabel}_max"] = cmin_max
            else:
                cmin = np.round(100 * (np.percentile(valA, [50])-np.percentile(val, [50])))
                StructGrow.loc[StructGrow['structure_name'] == struct, f"{xlabel}_{ylabel}"] = cmin

# %% Select columns for heatmap
HM = {}
HM["cellnuc_heatmap"] = [
    "Cell volume",
    "Cell surface area",
    "Nuclear volume",
    "Nuclear surface area",
    "Cytoplasmic volume",
]
HM["cellnuc_heatmap_abbs"] = [
    "Cell vol",
    "Cell area",
    "Nuc vol",
    "Nuc area",
    "Cyto vol",
]
HM["cellnuc_heatmap_COMP_metrics"] = [
    "Cell volume",
    "Cell surface area",
    "Nuclear volume",
    "Nuclear surface area",
]

HM["cellnuc_COMP_abbs"] = [
    "Cell vol*",
    "Cell area*",
    "Nuc vol*",
    "Nuc area*",
]

HM["cellnuc_heatmap_RES_metrics"] = ['cell_AV', 'cell_V', 'cell_A', 'nuc_AV', 'nuc_V', 'nuc_A']
HM["cellnuc_heatmap_RES_abbs"] =    ['Cell V+A', 'Cell vol', 'Cell area', 'Nuc V+A', 'Nuc Vol', 'Nuc Area']

HM["struct_heatmap_metrics"] = "Structure volume"
HM["COMP_type"] = "AV"
HM["LIN_type"] = "Linear"

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
struct_metric = HM["struct_heatmap_metrics"]
for xi, xlabel in enumerate(HM["cellnuc_heatmap_RES_metrics"]):
    keepcolumns.append(f"{xlabel}_{struct_metric}")

HeatMapComp = StructGrow[keepcolumns]

keepcolumns = ['structure_name']
struct_metric = HM["struct_heatmap_metrics"]
keepcolumns.append(f"cellnuc_AV_{struct_metric}")
keepcolumns.append(f"cellnuc_AV_{struct_metric}_min")
keepcolumns.append(f"cellnuc_AV_{struct_metric}_max")
HeatMapAll = StructGrow[keepcolumns]

# %% Annotation
ann_st = structures[[ 'Location', 'Structure', 'Gene']].astype('category')
ann_st = ann_st.to_numpy()
color_st = structures[['Color']]

#%% Plot Arrays
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

plot_arrayAll = HeatMapAll
plot_arrayAll = plot_arrayAll.set_index(plot_arrayAll['structure_name'],drop=True)
plot_arrayAll = plot_arrayAll.drop(['structure_name'],axis=1)
plot_arrayAll = plot_arrayAll.reindex(list(ann_st[:,-1]))
panAll = plot_arrayAll.to_numpy()

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
scalemodel = 'prc'
struct_metric = HM["struct_heatmap_metrics"]
growvec = np.zeros((pan.shape[0], 3))
for i, struct in enumerate(list(ann_st[:,-1])):
    gv = ScaleMat[f"{struct_metric}_{struct}{scalemodel}"].to_numpy()
    growvec[i,0] = np.percentile(gv,50)
    growvec[i, 1] = np.percentile(gv, 5)
    growvec[i, 2] = np.percentile(gv, 95)

growvecC = np.zeros((len(HM["cellnuc_heatmap"]), 1))
for i, struct in enumerate(HM["cellnuc_heatmap"]):
    gv = ScaleMat[f"{struct}_{scalemodel}"].to_numpy()
    growvecC[i,0] = np.percentile(gv,50)

# growvecC[0]=100

# %% Bins and bin center of growth rates
# nbins = 50
# xbincenter = Grow.loc[np.arange(nbins),'bins'].to_numpy()
# start_bin = int(Grow.loc[50,'bins'])
# end_bin = int(Grow.loc[51,'bins'])
# perc_values = [5, 25, 50, 75, 95]
# growfac = 2
ps = data_root / "growing"
cell_doubling = loadps(ps, f"cell_doubling")

# %% measurements
w1 = -1
w2 = 0
w3 = 0.01
w4 = 0.02
w5 = 0.01
w6 = 0.01
w7 = 0.003
w8 = 0.01
w9 = 0.01
w10 = 0.08
w11 = -0.02
w12 = 0.01
w13 = 0.06
w14 = 0.005

x3s = 0.03
x8s = 0.14
x8 = 0.28
x3 = (1-(w10+x8+x8s+x3s+w4+x3s+w5))/2
x4 = 0.13
x4r = 0.013
x4l = x4-x4r
x5 = 0.03
x6 = 0.2
x7 = 0.2
x7l = x7/6
x7r = x7-x7l
# x8 = 1-w6-x4-w7-x5-w8-x6-w9-x7-w12-x7-w10-x8s-w5
x9 = x7l+w12+x7r-w11+w9
x10 = x7r
x2s = 0.03
xw = w6+x4+w7+x5+w8+x6+w9+x7l+w12+x7r
x1 = 1-xw-w13-w14
h1 = 0.03
h2 = 0.005
h3 = 0.05
h4 = 0.04
h5 = 0.005
h6 = 0.015
h7 = 0.035
h8 = 0.02
h9 = 0.015
h10 = 0.04

y3s = 0.03
y6s = 0.14
y6 = 0.28
y3 = ((h4+y6+y6s)-(y3s+h2+y3s))/2
y4 = 0.355
# y6 = 1-(h1+y1+ya+y2s+y2+h4+h5+y6s)
yh = h4+y6+y6s
y5 = 1-(yh+y4+h5+h3)
y7 = (y5+h5-h6-h7-h8-h9)/3
y1 =  0.15
ya = 0.02
y2s = 0.03
y2 = 1-yh-h3-y1-ya-y2s-h10-y7-h9

print(y4/15-y5/5)

# %% other parameters
rot = -20
alpha = 0.5
lw_scatter=5

lw = 1
lw2 = 2
mstrlength = 10
fs = 12
fs_num = 13
fs_scatter = 12
fs_ann = 12
fn = 'Arial'

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], min(amount, c[1]), c[2])

# %%
# Relative
from stemcellorganellesizescaling.analyses.utils.scatter_plotting_func import ascatter, oscatter, ocscatter
from stemcellorganellesizescaling.analyses.utils.grow_plotting_func import growplot

importlib.reload(
    sys.modules["stemcellorganellesizescaling.analyses.utils.scatter_plotting_func"]
)
importlib.reload(
    sys.modules["stemcellorganellesizescaling.analyses.utils.grow_plotting_func"]
)

from stemcellorganellesizescaling.analyses.utils.scatter_plotting_func import ascatter, oscatter, ocscatter
from stemcellorganellesizescaling.analyses.utils.grow_plotting_func import growplot


# %%layout
fig = plt.figure(figsize=(12, 12))
plt.rcParams.update({"font.size": fs})
plt.rcParams['font.sans-serif'] = fn
plt.rcParams['font.family'] = "sans-serif"

# Scale4
axScale4 = fig.add_axes([w3+x3s, y3s, x3, y3])
# Scale4 side
axScale4S = fig.add_axes([w3, y3s, x3s, y3])
# Scale4 bottom
axScale4B = fig.add_axes([w3+x3s, 0, x3, y3s])
ps = data_root / statsIN / "cellnuc_struct_metrics"
oscatter(
    axScale4,
    axScale4B,
    axScale4S,
    HM['cellnuc_heatmap'][0],
    'FBL',
    HM['cellnuc_heatmap'][0],
    f"{structures.loc[structures['Gene']=='FBL','Structure'].values[0]} vol",
    'Structure volume',
    cells,
    ps,
    kde_flag=True,
    fourcolors_flag=False,
    colorpoints_flag=True,
    rollingavg_flag=True,
    ols_flag=True,
    N2=1000,
    fs2=fs,
    fs=fs,
    fn = fn,
    typ=['vol','vol'],
)

# Scale5
axScale5 = fig.add_axes([w3+x3s+x3+x3s+w4, y3s, x3, y3])
# Scale5 side
axScale5S = fig.add_axes([w3+x3+x3s+w4, y3s, x3s, y3])
# Scale5 bottom
axScale5B = fig.add_axes([w3+x3s+x3+x3s+w4, 0, x3, y3s])
ps = data_root / statsIN / "cellnuc_struct_metrics"
oscatter(
    axScale5,
    axScale5B,
    axScale5S,
    HM['cellnuc_heatmap'][2],
    'FBL',
    HM['cellnuc_heatmap'][2],
    f"{structures.loc[structures['Gene']=='FBL','Structure'].values[0]} vol",
    'Structure volume',
    cells,
    ps,
    kde_flag=True,
    fourcolors_flag=False,
    colorpoints_flag=True,
    rollingavg_flag=True,
    ols_flag=True,
    N2=1000,
    fs2=fs,
    fs=fs,
    fn=fn,
    typ=['vol','vol'],
)

# Scale1
axScale1 = fig.add_axes([w3+x3s, y3s+y3+h2+y3s, x3, y3])
# Scale1 side
axScale1S = fig.add_axes([w3, y3s+y3+h2+y3s, x3s, y3])
# Scale1 bottom
axScale1B = fig.add_axes([w3+x3s, 0+y3+h2+y3s, x3, y3s])
ps = data_root / statsIN / "cellnuc_struct_metrics"
oscatter(
    axScale1,
    axScale1B,
    axScale1S,
    HM['cellnuc_heatmap'][0],
    'TOMM20',
    HM['cellnuc_heatmap'][0],
    f"{structures.loc[structures['Gene']=='TOMM20','Structure'].values[0]} vol",
    'Structure volume',
    cells,
    ps,
    kde_flag=True,
    fourcolors_flag=False,
    colorpoints_flag=True,
    rollingavg_flag=True,
    ols_flag=True,
    N2=1000,
    fs2=fs,
    fs=fs,
    fn=fn,
    typ=['vol','vol'],
)

# Scale2
axScale2 = fig.add_axes([w3+x3s+x3+x3s+w4, y3s+y3+h2+y3s, x3, y3])
# Scale2 side
axScale2S = fig.add_axes([w3+x3+x3s+w4, y3s+y3+h2+y3s, x3s, y3])
# Scale2 bottom
axScale2B = fig.add_axes([w3+x3s+x3+x3s+w4, 0+y3+h2+y3s, x3, y3s])
# Plot
ps = data_root / statsIN / "cellnuc_struct_metrics"
oscatter(
    axScale2,
    axScale2B,
    axScale2S,
    HM['cellnuc_heatmap'][0],
    'RAB5A',
    HM['cellnuc_heatmap'][0],
    f"{structures.loc[structures['Gene']=='RAB5A','Structure'].values[0]} vol",
    'Structure volume',
    cells,
    ps,
    kde_flag=True,
    fourcolors_flag=False,
    colorpoints_flag=True,
    rollingavg_flag=True,
    ols_flag=True,
    N2=1000,
    fs2=fs,
    fs=fs,
    fn=fn,
    typ=['vol','vol'],
)

# # GrowVarS side
axGrowVarSS= fig.add_axes([w3+x3s+x3+w4+x3s+x3+w10+x8, h4, x8s, y6])
yrange = [20, 110]
pos =  np.argwhere(np.logical_and(growvec[:,0]>yrange[0],growvec[:,0]<yrange[1]))
yarray = growvec[pos,0].squeeze()
temp = yarray.argsort()
ranks = np.empty_like(temp).astype(np.float)
ranks[temp] = np.linspace(yrange[0],yrange[1],len(yarray))
growy = growvec[:,0].copy()
growy[pos] = np.expand_dims(ranks,axis=1)
dis = np.sort(growy)
dis = (dis[1]-dis[0])/2
for i in np.arange(len(growvec)):
    growval = growvec[i,0]
    struct = ann_st[i, 1]
    growyval = growy[i]
    axGrowVarSS.plot(0,growval,'.',markersize=10,color=color_st.loc[i, 'Color'])
    axGrowVarSS.plot([0, 10], [growval, growyval],color=color_st.loc[i, 'Color'])
    axGrowVarSS.plot([0, 10], [growval, growyval], color=color_st.loc[i, 'Color'])
    axGrowVarSS.fill([10, 20, 20, 10, 10], [growyval-dis, growyval-dis, growyval+dis, growyval+dis, growyval-dis], facecolor=color_st.loc[i, 'Color'],edgecolor='None')
    axGrowVarSS.text(21,growyval, struct, fontsize=fs,color='k', verticalalignment='center',fontweight='normal',horizontalalignment='left',fontname=fn)
axGrowVarSS.set_ylim(bottom=0,top=115)
axGrowVarSS.set_xlim(left=0, right=100)
axGrowVarSS.axis("off")

# GrowVarS
axGrowVarS= fig.add_axes([w3+x3s+x3+w4+x3s+x3+w10, h4, x8, y6])
for i in np.arange(len(growvec)):
    growval = growvec[i,0]
    growmin = growvec[i, 1]
    growmax = growvec[i, 2]
    struct = ann_st[i, 2]
    axGrowVarS.plot([panAll[i, 1], panAll[i, 2]], [growval, growval], color='k',linewidth=.75)
    axGrowVarS.plot([panAll[i, 0], panAll[i, 0]], [growmin, growmax], color='k',linewidth=.75)
    axGrowVarS.plot(panAll[i, 0], growval, 'o', markersize=7,
                    mfc=color_st.loc[i, 'Color'], mec='k', linewidth=1)

axGrowVarS.set_ylim(bottom=0,top=115)
axGrowVarS.set_xlim(left=0, right=100)
axGrowVarS.set_xlabel('Structure volume explained by cell and nuclear metrics (%)')
axGrowVarS.set_ylabel('Structure volume scaling as cell volume doubles  (%)')
axGrowVarS.grid()

# GrowVarS bottom
axGrowVarSB= fig.add_axes([w3+x3s+x3+w4+x3s+x3+w10, h4+y6, x8+x8s/2, y6s])
xrange = [10, 100*(1*(x8s/2+x8)/x8)]
pos =  np.argwhere(np.logical_and(panAll[:,0]>xrange[0],panAll[:,0]<xrange[1]))
xarray = panAll[pos,0].squeeze()
temp = xarray.argsort()
ranks = np.empty_like(temp).astype(np.float)
ranks[temp] = np.linspace(xrange[0],xrange[1],len(xarray))
panx = panAll[:,0].copy()
panx[pos] = np.expand_dims(ranks,axis=1)
disx = np.sort(panx)
disx = (disx[1]-disx[0])/2
for i in np.arange(len(growvec)):
    growval = growvec[i,0]
    struct = ann_st[i, 1]
    panxval = panx[i]
    axGrowVarSB.plot(panAll[i,0],0,'.',markersize=10,color=color_st.loc[i, 'Color'])
    axGrowVarSB.plot([panAll[i,0], panxval], [0, 0.8], color=color_st.loc[i, 'Color'])
    axGrowVarSB.fill([panxval-disx, panxval+disx, panxval+disx, panxval-disx],[0.8, 0.8, 1.15, 1.15],facecolor=color_st.loc[i, 'Color'], edgecolor='None')
    axGrowVarSB.text(panxval, 1.2, struct, fontsize=fs, fontname=fn, color='k', verticalalignment='bottom',fontweight='normal',horizontalalignment='center',rotation=90)
axGrowVarSB.set_ylim(bottom=0,top=5)
axGrowVarSB.set_xlim(left=0, right=100*(1*(x8s/2+x8)/x8))
axGrowVarSB.axis("off")

# Annotation
axAnn = fig.add_axes([w6+x4l, yh+h3, x4r, y4])
newcolors = np.zeros((len(color_st), 4))
ylabels=[]
for i, c in enumerate(color_st['Color']):
    newcolors[i,:] = mcolors.to_rgba(c)
scmap = ListedColormap(newcolors)
axAnn.imshow(np.expand_dims(np.arange(len(color_st)),axis=1), cmap=scmap, aspect='auto')
for i in range(len(ann_st)):
    text = axAnn.text(-.6, i, ann_st[i,1],
                          ha="right", va="center", color='k', fontname = fn)
    # text = axAnn.text(-3, i, ann_st[i, 0],
    #                   ha="left", va="center", color='k', fontsize=fs_ann, fontname=fn)
axAnn.axis('off')

# Organelle Growth rates
axOrgGrow = fig.add_axes([w6+x4+w7, yh+h3, x5, y4])
axOrgGrow.imshow(np.expand_dims(growvec[:,0],axis=0).T, aspect='auto', cmap='Greens',vmin=0, vmax=100)
for i in range(len(growvec[:,0])):
        val = np.int(growvec[i, 0])
        text = axOrgGrow.text(0, i, val,
                          ha="center", va="center", color="w", fontsize=fs_num, fontweight='bold', fontname = fn)
axOrgGrow.set_yticks([])
axOrgGrow.set_yticklabels([])
axOrgGrow.set_xticks([])
axOrgGrow.text(0,1.03*len(growvec)-.5,'Scaling',horizontalalignment='center',verticalalignment = 'center', fontname = fn)
axOrgGrow.text(0,1.06*len(growvec)-.5,'rate (%)',horizontalalignment='center',verticalalignment = 'center', fontname = fn)

# Cell Growth rates
axCellGrow = fig.add_axes([w6+x4+w7, yh+h3+y4, x5, y5])
axCellGrow.imshow(growvecC, aspect='auto', cmap='Greens',vmin=0, vmax=100)
for i in range(len(growvecC)):
        val = np.int(growvecC[i, 0])
        text = axCellGrow.text(0, i, val,
                          ha="center", va="center", color="w", fontsize=fs_num, fontweight='bold', fontname = fn)
axCellGrow.set_yticks(range(len(HM["cellnuc_heatmap"])))
axCellGrow.set_yticklabels(HM["cellnuc_heatmap"],fontname = fn)
axCellGrow.set_xticks([])
axCellGrow.set_xticklabels([])

# Organelle Variance rates
axOrgVar = fig.add_axes([w6+x4+w7+x5+w8, yh+h3, x6, y4])
axOrgVar.imshow(pan, aspect='auto', cmap='RdBu_r',vmin=-100, vmax=100)
for i in range(len(plot_array)):
    for j in range(len(plot_array.columns)):
        val = np.int(pan[i, j])
        text = axOrgVar.text(j, i, val,
                          ha="center", va="center", color="w", fontsize=fs_num, fontweight='bold',fontname = fn)
axOrgVar.set_yticks([])
axOrgVar.set_yticklabels([])
xlabels = HM["cellnuc_heatmap_abbs"]
axOrgVar.set_xticks([])
for j, xlabel in enumerate(xlabels):
    xls = xlabel.split( )
    axOrgVar.text(j,1.03*len(plot_array)-.5,xls[0],horizontalalignment='center',verticalalignment = 'center', fontname = fn)
    axOrgVar.text(j,1.06*len(plot_array)-.5,xls[1],horizontalalignment='center',verticalalignment = 'center', fontname = fn)

# Cell Variance rates
axCellVar = fig.add_axes([w6+x4+w7+x5+w8, yh+h3+y4, x6, y5])
axCellVar.imshow(panCN, aspect='auto', cmap='RdBu_r',vmin=-100, vmax=100)
for i in range(len(plot_arrayCN)):
    for j in range(len(plot_arrayCN.columns)):
        val = np.int(panCN[i, j])
        text = axCellVar.text(j, i, val,
                          ha="center", va="center", color="w", fontsize=fs_num, fontweight='bold', fontname = fn)
axCellVar.set_yticks([])
axCellVar.set_yticklabels([])
axCellVar.set_xticks([])
axCellVar.set_xticklabels([])
axCellVar.axis('off')

# Composite model All rates
axAllVarC= fig.add_axes([w6+x4+w7+x5+w8+x6+w9, yh+h3, x7l, y4])
axAllVarC.imshow(np.expand_dims(panAll[:,0],axis=1), aspect='auto', cmap='RdBu_r',vmin=-100, vmax=100)
for i in range(len(plot_arrayAll)):
    for j in range(1):
        val = np.int(panAll[i, j])
        text = axAllVarC.text(j, i, val,
                          ha="center", va="center", color="w", fontsize=fs_num, fontweight='bold', fontname = fn)
axAllVarC.set_yticks([])
axAllVarC.set_yticklabels([])
xlabels = ['All metrics']
axAllVarC.set_xticks([])
for j, xlabel in enumerate(xlabels):
    xls = xlabel.split( )
    axAllVarC.text(j,1.03*len(plot_arrayAll)-.5,xls[0],horizontalalignment='center',verticalalignment = 'center', fontname = fn)
    axAllVarC.text(j,1.06*len(plot_arrayAll)-.5,xls[1],horizontalalignment='center',verticalalignment = 'center', fontname = fn)

# Composite model unique rates
axCompVarC= fig.add_axes([w6+x4+w7+x5+w8+x6+w9+x7l+w12, yh+h3, x7r, y4])
axCompVarC.imshow(panComp, aspect='auto', cmap='Oranges',vmin=0, vmax=20)
for i in range(len(plot_arrayComp)):
    for j in range(len(plot_arrayComp.columns)):
        val = np.int(panComp[i, j])
        text = axCompVarC.text(j, i, val,
                          ha="center", va="center", color="w", fontsize=fs_num, fontweight='bold', fontname = fn)
axCompVarC.set_yticks([])
axCompVarC.set_yticklabels([])
xlabels = HM["cellnuc_heatmap_RES_abbs"]
axCompVarC.set_xticks([])
for j, xlabel in enumerate(xlabels):
    xls = xlabel.split( )
    axCompVarC.text(j,1.03*len(plot_arrayComp)-.5,xls[0],horizontalalignment='center',verticalalignment = 'center', fontname = fn)
    axCompVarC.text(j,1.06*len(plot_arrayComp)-.5,xls[1],horizontalalignment='center',verticalalignment = 'center', fontname = fn)

axAnn2 = fig.add_axes([w6+x4+w7+x5+w8+x6+w9+x7l+w12+x7r+0.005, yh+h3, x4r, y4])
axAnn2.imshow(np.expand_dims(np.arange(len(color_st)),axis=1), cmap=scmap, aspect='auto')
axAnn2.axis('off')

axUniVarBar= fig.add_axes([w6+x4+w7+x5+w8+x6+w9+x7l+w12, yh+h3+y4+h6, x10, y7])
axUniVarBar.imshow(np.expand_dims(np.linspace(0,20,101),axis=0), aspect='auto', cmap='Oranges',vmin=0, vmax=20)
text = axUniVarBar.text(0, 0, '0', ha="left", va="center", color="k", fontsize=fs, fontweight='bold')
text = axUniVarBar.text(50, 0, '10', ha="center", va="center", color="w", fontsize=fs, fontweight='bold')
text = axUniVarBar.text(100, 0, '20', ha="right", va="center", color="w", fontsize=fs, fontweight='bold')
axUniVarBar.set_yticks([])
axUniVarBar.set_yticklabels([])
axUniVarBar.set_xticks([])
axUniVarBar.set_xticklabels([])
axUniVarBar.set_title('Unique expl. var. (%)',verticalalignment='top',fontsize=fs)

axExpVarBar= fig.add_axes([w6+x4+w7+x5+w8+x6+w11, yh+h3+y4+h6+y7+h7, x9, y7])
axExpVarBar.imshow(np.expand_dims(np.linspace(-100,100,201),axis=0), aspect='auto', cmap='RdBu_r',vmin=-100, vmax=100)
text = axExpVarBar.text(0, 0, '-100', ha="left", va="center", color="w", fontsize=fs, fontweight='bold')
text = axExpVarBar.text(200, 0, '100', ha="right", va="center", color="w", fontsize=fs, fontweight='bold')
text = axExpVarBar.text(100, 0, '0', ha="center", va="center", color="k", fontsize=fs, fontweight='bold')
axExpVarBar.set_yticks([])
axExpVarBar.set_yticklabels([])
axExpVarBar.set_xticks([50, 150])
axExpVarBar.set_xticklabels(['Neg. corr.', 'Pos. corr.'],verticalalignment='center')
axExpVarBar.set_title('Explained Variance (%)',verticalalignment='top',fontsize=fs)

axGrowBar= fig.add_axes([w6+x4+w7+x5+w8+x6+w11, yh+h3+y4+h6+y7+h7+y7+h8, x9, y7])
axGrowBar.imshow(np.expand_dims(np.linspace(0,100,101),axis=0), aspect='auto', cmap='Greens',vmin=0, vmax=100)
text = axGrowBar.text(0, 0, '0', ha="left", va="center", color="k", fontsize=fs, fontweight='bold')
text = axGrowBar.text(50, 0, '50', ha="center", va="center", color="w", fontsize=fs, fontweight='bold')
text = axGrowBar.text(100, 0, '100', ha="right", va="center", color="w", fontsize=fs, fontweight='bold')
axGrowBar.set_yticks([])
axGrowBar.set_yticklabels([])
axGrowBar.set_xticks([])
axGrowBar.set_xticklabels([])
axGrowBar.set_title('Scaling rate relative to cell volume (%)',verticalalignment='top',fontsize=fs)

# GrowCell
axGrowCell = fig.add_axes([xw+w13, yh+h3, x1, y1])
growplot(
    axGrowCell,
    'Cell volume_mean',
    ['RAB5A','SLC25A17','TOMM20'],
    ScaleCurve,
    structures,
    fs,
)

# Density bar
axColorDens = fig.add_axes([xw+w13, yh+h3+y1+ya+y2s+y2+h10, x1, y7])
# create spectral color bar
cpmap = plt.cm.get_cmap(plt.cm.plasma)
cpmap = cpmap(np.linspace(0, 1, 100) ** 0.4)
cpmap[0:10, 3] = np.linspace(0.3, 1, 10)
cpmap = ListedColormap(cpmap)
axColorDens.imshow(np.expand_dims(np.linspace(0,100,101),axis=0), aspect='auto', cmap=cpmap,vmin=0, vmax=100)
axColorDens.plot([5, 5],[-.5, .5],'k',linewidth=lw)
axColorDens.plot([50, 50],[-.5, .5],'k',linewidth=lw)
text = axColorDens.text(0, 0, '0', ha="left", va="center_baseline", color="gray", fontsize=fs, fontweight='normal')
text = axColorDens.text(20, 0, '20', ha="center", va="center_baseline", color="gray", fontsize=fs, fontweight='normal')
text = axColorDens.text(40, 0, '40', ha="center", va="center_baseline", color="gray", fontsize=fs, fontweight='normal')
text = axColorDens.text(60, 0, '60', ha="center", va="center_baseline", color="gray", fontsize=fs, fontweight='normal')
text = axColorDens.text(80, 0, '80', ha="center", va="center_baseline", color="gray", fontsize=fs, fontweight='normal')
text = axColorDens.text(100, 0, '100', ha="right", va="center_baseline", color="gray", fontsize=fs, fontweight='normal')
text = axColorDens.text(0, 1.2, '5% ofcells with', ha="left", va="center", color="k", fontsize=fs, fontweight='normal')
text = axColorDens.text(0, 1.8, 'lowest density', ha="left", va="center", color="k", fontsize=fs, fontweight='normal')
text = axColorDens.text(75, 1.2, '50% of cells with', ha="center", va="center", color="k", fontsize=fs, fontweight='normal')
text = axColorDens.text(75, 1.8, 'highest density', ha="center", va="center", color="k", fontsize=fs, fontweight='normal')
axColorDens.set_yticks([])
axColorDens.set_yticklabels([])
axColorDens.set_xticks([-0.5, 5, 50, 100.5])
axColorDens.set_xticklabels([])
axColorDens.set_title('Color map for cell density in scatter plots',verticalalignment='top',fontsize=fs)

# Grow
axGrow = fig.add_axes([xw+w13, yh+h3+y1+ya+y2s, x1, y2])
# Grow bottom
axGrowB = fig.add_axes([xw+w13, yh+h3+y1+ya, x1, y2s])
# Grow side
axGrowS = fig.add_axes([xw+w13-x2s, yh+h3+y1+ya+y2s, x2s, y2])
# Plot
ps = data_root / statsIN / "cell_nuc_metrics"
ascatter(
    axGrow,
    axGrowB,
    axGrowS,
    HM['cellnuc_heatmap'][0],
    HM['cellnuc_heatmap'][2],
    HM['cellnuc_heatmap'][0],
    HM['cellnuc_heatmap'][2],
    cells,
    ps,
    kde_flag=True,
    fourcolors_flag=False,
    colorpoints_flag=True,
    rollingavg_flag=True,
    ols_flag=True,
    N2=1000,
    fs2=fs,
    fs=fs,
    cell_doubling = cell_doubling,
    typ=['vol','vol']
)

# Cell Size
xlim = axGrowB.get_xlim()
ylim = axGrowB.get_ylim()
tf = (0.108333**3)
axGrowB.plot([tf*cell_doubling[0], tf*cell_doubling[0]], ylim, '--', linewidth=lw2,color=[0.5, 0.5, 1, 0.5])
axGrowB.plot([tf*cell_doubling[1], tf*cell_doubling[1]], ylim, '--', linewidth=lw2,color=[0.5, 0.5, 1, 0.5])

# Transition
axTransition = fig.add_axes([xw+w13, yh+h3+y1, x1, ya])
axTransition.set_xlim(left=xlim[0], right=xlim[1])
ylm = -1
axTransition.set_ylim(bottom=ylm, top=0)
axTransition.plot([xlim[0], tf*cell_doubling[0]], [ylm, 0], '--', linewidth=lw2,color=[0.5, 0.5, 1, 0.5])
axTransition.plot([xlim[1], tf*cell_doubling[1]], [ylm, 0], '--', linewidth=lw2, color=[0.5, 0.5, 1, 0.5])
axTransition.axis('off')

plot_save_path = pic_root / f"heatmap_v11_20201123_res1000.png"
plt.savefig(plot_save_path, format="png", dpi=1000)
plot_save_path = pic_root / f"heatmap_v11_20201123_res600.png"
plt.savefig(plot_save_path, format="png", dpi=600)
plot_save_path = pic_root / f"heatmap_v11_20201123_res300.png"
plt.savefig(plot_save_path, format="png", dpi=300)
plot_save_path = pic_root / f"heatmap_v11_20201123.svg"
plt.savefig(plot_save_path, format="svg")
plt.close()

# plt.show()



