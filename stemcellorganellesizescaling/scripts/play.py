#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, platform
import sys, importlib

# Third party
from stemcellorganellesizescaling.analyses.utils.scatter_plotting_func import (
    organelle_scatter,
    fscatter,
    compensated_scatter,
    organelle_scatterT,
    compensated_scatter_t
)

importlib.reload(
    sys.modules["stemcellorganellesizescaling.analyses.utils.scatter_plotting_func"]
)
from stemcellorganellesizescaling.analyses.utils.scatter_plotting_func import (
    organelle_scatter,
    fscatter,
    compensated_scatter,
    organelle_scatterT,
    compensated_scatter_t
)

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

# Resolve directories
data_root = dirs[0]
pic_root = dirs[1]

tableIN = "SizeScaling_20201012.csv"
table_compIN = "SizeScaling_20201012_comp.csv"
statsIN = "Stats_20201012"
# Load dataset
cells = pd.read_csv(data_root / tableIN)
np.any(cells.isnull())
cells_COMP = pd.read_csv(data_root / table_compIN)
np.any(cells_COMP.isnull())

# Remove outliers
# %% Parameters, updated directories
save_flag = 0  # save plot (1) or show on screen (0)
plt.rcParams.update({"font.size": 12})
pic_root = pic_root / "scatter_plots"
pic_root.mkdir(exist_ok=True)

# %% Feature sets
FS = {}
FS["cell_metrics_AVH"] = ["Cell surface area", "Cell volume", "Cell height"]
FS["nuc_metrics_AVH"] = ["Nuclear surface area", "Nuclear volume", "Nucleus height"]
FS["cell_metrics_AV"] = ["Cell surface area", "Cell volume"]
FS["nuc_metrics_AV"] = ["Nuclear surface area", "Nuclear volume"]
FS["cell_metrics_H"] = ["Cell height"]
FS["nuc_metrics_H"] = ["Nucleus height"]
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
    "Cell area",
    "Cell vol",
    "Cell height",
    "Nuclear area",
    "Nuclear vol",
    "Nucleus height",
]

FS["selected_structures"] = [
    "LMNB1",
    "ST6GAL1",
    "TOMM20",
    "SEC61B",
    "ATP2A2",
    "LAMP1",
    "RAB5A",
    "SLC25A17",
    "TUBA1B",
    "TJP1",
    "NUP153",
    "FBL",
    "NPM1",
    "SON",
]
FS["other_structures"] = list(
    set(cells["structure_name"].unique()) - set(FS["selected_structures"])
)
# struct_metrics = [
#     "Structure volume",
#     "Number of pieces",
#     "Piece average",
#     "Piece std",
#     "Piece CoV",
#     "Piece sum",
# ]
FS["struct_metrics"] = ["Structure volume", "Number of pieces"]
FS["COMP_types"] = ["AVH","AV","H"]


# %%
xvec = [1]
yvec = [4]
pair1 = np.stack((xvec, yvec)).T

# %%
xvec = [1, 1, 6, 1, 4, 6]
yvec = [4, 6, 4, 0, 3, 3]
pair6 = np.stack((xvec, yvec)).T

# %%
N = 13
xvec = np.random.choice(len(FS["cellnuc_metrics"]), N)
yvec = np.random.choice(len(FS["cellnuc_metrics"]), N)
pairN = np.stack((xvec, yvec)).T

# %%
L = len(FS["cellnuc_metrics"])
pair21 = np.zeros((int(L * (L - 1) / 2), 2)).astype(np.int)
i = 0
for f1 in np.arange(L):
    for f2 in np.arange(L):
        if f2 > f1:
            pair21[i, :] = [f1, f2]
            i += 1


#%%
bcells = pd.read_csv(data_root / 'babycells' / 'babybump_20201016.csv')
plotname = "test"
ps = data_root / statsIN / "cell_nuc_metrics"
fscatter(
    FS["cellnuc_metrics"],
    FS["cellnuc_abbs"],
    pair6,
    cells,
    ps,
    False,
    pic_root,
    f"{plotname}_plain",
    kde_flag=True,
    fourcolors_flag=False,
    colorpoints_flag=True,
    rollingavg_flag=True,
    ols_flag=True,
    N2=1000,
    plotcells=bcells
)

#%% Plot some more
pic_rootT = pic_root / "cell_nuc_metrics"
pic_rootT.mkdir(exist_ok=True)
ps = data_root / statsIN / "cell_nuc_metrics"

kde_flagL = [False, False, False, True, True, True, True, True, True]
fourcolors_flagL = [False, False, False, False, True, False, False, True, False]
colorpoints_flagL = [False, False, False, False, False, True, False, False, True]
rollingavg_flagL = [False, True, True, False, False, False, True, True, True]
ols_flagL = [False, False, True, False, False, False, True, True, True]
Name = [
    "plain",
    "roll",
    "ols",
    "galaxy",
    "arch",
    "color",
    "galaxy_ro",
    "arch_ro",
    "color_ro",
]

PS = {}
PS["pair1"] = pair1
PS["pair6"] = pair6
PS["pair21"] = pair21

for key in PS:
    pair = PS[key]
    for (
        i,
        (kde_flag, fourcolors_flag, colorpoints_flag, rollingavg_flag, ols_flag, name),
    ) in enumerate(
        zip(
            kde_flagL,
            fourcolors_flagL,
            colorpoints_flagL,
            rollingavg_flagL,
            ols_flagL,
            Name,
        )
    ):
        plotname = f"{key}_{name}"
        print(plotname)
        fscatter(
            FS["cellnuc_metrics"],
            FS["cellnuc_abbs"],
            pair,
            cells,
            ps,
            True,
            pic_rootT,
            plotname,
            kde_flag=kde_flag,
            fourcolors_flag=fourcolors_flag,
            colorpoints_flag=colorpoints_flag,
            rollingavg_flag=rollingavg_flag,
            ols_flag=ols_flag,
            N2=1000,
        )

#%%
plotname = "test"
ps = data_root / statsIN / "cellnuc_struct_metrics"
organelle_scatter(
    FS["cellnuc_metrics"],
    FS["cellnuc_abbs"],
    FS["selected_structures"],
    "Structure volume",
    cells,
    ps,
    False,
    pic_root,
    plotname,
    kde_flag=True,
    fourcolors_flag=False,
    colorpoints_flag=True,
    rollingavg_flag=True,
    ols_flag=True,
    N2=100,
    plotcells=bcells,
)

#%% Plot some more
pic_rootT = pic_root / "cellnuc_struct_metrics"
pic_rootT.mkdir(exist_ok=True)
ps = data_root / statsIN / "cellnuc_struct_metrics"

kde_flagL = [False, False, False, True, True, True, True, True, True]
fourcolors_flagL = [False, False, False, False, True, False, False, True, False]
colorpoints_flagL = [False, False, False, False, False, True, False, False, True]
rollingavg_flagL = [False, True, True, False, False, False, True, True, True]
ols_flagL = [False, False, True, False, False, False, True, True, True]
Name = [
    "plain",
    "roll",
    "ols",
    "galaxy",
    "arch",
    "color",
    "galaxy_ro",
    "arch_ro",
    "color_ro",
]


for i in np.arange(2):
    if i == 0:
        sel_struct = FS["selected_structures"]
        key = "sel"
    elif i == 1:
        sel_struct = FS["other_structures"]
        key = "other"
    for sm, struct_metric in enumerate(FS["struct_metrics"]):
        for (
            j,
            (
                kde_flag,
                fourcolors_flag,
                colorpoints_flag,
                rollingavg_flag,
                ols_flag,
                name,
            ),
        ) in enumerate(
            zip(
                kde_flagL,
                fourcolors_flagL,
                colorpoints_flagL,
                rollingavg_flagL,
                ols_flagL,
                Name,
            )
        ):
            plotname = f"{key}_{struct_metric}_{name}"
            print(plotname)
            organelle_scatter(
                FS["cellnuc_metrics"],
                FS["cellnuc_abbs"],
                sel_struct,
                struct_metric,
                cells,
                ps,
                True,
                pic_rootT,
                plotname,
                kde_flag=kde_flag,
                fourcolors_flag=fourcolors_flag,
                colorpoints_flag=colorpoints_flag,
                rollingavg_flag=rollingavg_flag,
                ols_flag=ols_flag,
                N2=100,
            )

# %%
ps = data_root / statsIN / "cellnuc_struct_COMP_metrics"
compensated_scatter(
    FS["cellnuc_COMP_metrics"],
    FS["cellnuc_COMP_abbs"],
    FS["selected_structures"],
    "AVH",
    'Linear',
    "Structure volume",
    cells_COMP,
    ps,
    False,
    pic_root,
    plotname,
    kde_flag=True,
    fourcolors_flag=False,
    colorpoints_flag=True,
    rollingavg_flag=True,
    ols_flag=True,
    N2=100,
)
# %% OK plots
pic_rootT = pic_root / "cellnuc_struct_COMP_metrics"
pic_rootT.mkdir(exist_ok=True)
ps = data_root / statsIN / "cellnuc_struct_COMP_metrics"

kde_flagL = [False, False, False, True, True, True, True, True, True]
fourcolors_flagL = [False, False, False, False, True, False, False, True, False]
colorpoints_flagL = [False, False, False, False, False, True, False, False, True]
rollingavg_flagL = [False, True, True, False, False, False, True, True, True]
ols_flagL = [False, False, True, False, False, False, True, True, True]
Name = [
    "plain",
    "roll",
    "ols",
    "galaxy",
    "arch",
    "color",
    "galaxy_ro",
    "arch_ro",
    "color_ro",
]

for c, comp_type, in enumerate(FS["COMP_types"]):
    for ti, lin_type in enumerate(["Linear", "Complex"]):
        for i in np.arange(2):
            if i == 0:
                sel_struct = FS["selected_structures"]
                key = "sel"
            elif i == 1:
                sel_struct = FS["other_structures"]
                key = "other"
            for sm, struct_metric in enumerate(FS["struct_metrics"]):
                for (
                    j,
                    (
                        kde_flag,
                        fourcolors_flag,
                        colorpoints_flag,
                        rollingavg_flag,
                        ols_flag,
                        name,
                    ),
                ) in enumerate(
                    zip(
                        kde_flagL,
                        fourcolors_flagL,
                        colorpoints_flagL,
                        rollingavg_flagL,
                        ols_flagL,
                        Name,
                    )
                ):
                    plotname = f"{key}_{struct_metric}_{lin_type}_{comp_type}_{name}"
                    print(plotname)

                    compensated_scatter(
                        FS["cellnuc_COMP_metrics"],
                        FS["cellnuc_COMP_abbs"],
                        FS["selected_structures"],
                        comp_type,
                        lin_type,
                        struct_metric,
                        cells_COMP,
                        ps,
                        True,
                        pic_rootT,
                        plotname,
                        kde_flag=kde_flag,
                        fourcolors_flag=fourcolors_flag,
                        colorpoints_flag=colorpoints_flag,
                        rollingavg_flag=rollingavg_flag,
                        ols_flag=ols_flag,
                        N2=100,
                    )

#%%
plotname = "x"
ps = data_root / statsIN / "cellnuc_struct_metrics"
pic_rootT = pic_root / "forpres"
pic_rootT.mkdir(exist_ok=True)
organelle_scatterT(
    FS["cellnuc_metrics"],
    FS["cellnuc_abbs"],
    ['ST6GAL1','SON'],
    "Structure volume",
    cells,
    ps,
    True,
    pic_rootT,
    plotname,
    kde_flag=True,
    fourcolors_flag=False,
    colorpoints_flag=True,
    rollingavg_flag=True,
    ols_flag=True,
    N2=100,
    plotcells=bcells,
)

#%%
plotname = "x_c"
pic_rootT = pic_root / "forpres"
pic_rootT.mkdir(exist_ok=True)
ps = data_root / statsIN / "cellnuc_struct_COMP_metrics"
compensated_scatter_t(
    FS["cellnuc_COMP_metrics"],
    FS["cellnuc_COMP_abbs"],
    ['ST6GAL1','SON'],
    "AVH",
    'Linear',
    "Structure volume",
    cells_COMP,
    ps,
    True,
    pic_rootT,
    plotname,
    kde_flag=True,
    fourcolors_flag=False,
    colorpoints_flag=True,
    rollingavg_flag=True,
    ols_flag=True,
    N2=100,
)

# %%





