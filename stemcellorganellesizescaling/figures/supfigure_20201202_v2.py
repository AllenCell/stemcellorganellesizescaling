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
from skimage.morphology import binary_erosion
from scipy.ndimage.morphology import distance_transform_edt
import vtk
from aicsshparam import shtools
from tqdm import tqdm
import statsmodels.api as sm

# Third party

# Relative
from stemcellorganellesizescaling.analyses.utils.scatter_plotting_func import ascatter

print("Libraries loaded succesfully")
###############################################################################

log = logging.getLogger(__name__)

###############################################################################
#%% Directories
if platform.system() == "Windows":
    data_root = Path("E:/DA/Data/scoss/Data/Nov2020")
    pic_root = Path("E:/DA/Data/scoss/Pics/Nov2020")
elif platform.system() == "Linux":
    1/0

# %% Resolve directories and load data
tableIN = "SizeScaling_20201102.csv"
statsIN = "Stats_20201102"
# Load dataset
cells = pd.read_csv(data_root / tableIN)
np.any(cells.isnull())

# Remove outliers
# %% Parameters, updated directories
save_flag = 0  # save plot (1) or show on screen (0)
plt.rcParams.update({"font.size": 12})
pic_root = pic_root / "supplemental"
pic_root.mkdir(exist_ok=True)

# %% Feature sets
FS={}
FS["cellnuc_metrics"] = [
    "Cell surface area",
    "Cell volume",
    "Cell height",
    "Nuclear surface area",
    "Nuclear volume",
    "Nucleus height",
    "Cytoplasmic volume",
]

# %% Preparation for nuclear area

# %%
def compute_areas_and_volumes(r, stretch_factor, cylinder_flag):
    # r radius expressed as a number between 0 and 1

    #% Parameters
    x = 200  # x-dimension of 3d cell image
    y = 200  # y-dimension of 3d cell image
    z = 200  # z-dimension of 3d cell image
    # r = 0.5  # radius expressed as a number between 0 and 1
    r_sd = 0  # standard deviation of radius
    c_sd = 0  # standard deviation of distance from center
    # stretch_factor = 1 # make higher than 1 to generate ellipsoids
    # cylinder_flag = False # make True for cylinders

    # Make 3d matrices for x,y,z channels
    # Make x shape
    xvec = 1 + np.arange(x)
    xmat = np.repeat(xvec[:, np.newaxis], y, axis=1)
    xnd = np.repeat(xmat[:, :, np.newaxis], z, axis=2)
    # Make y shape
    yvec = 1 + np.arange(y)
    ymat = np.repeat(yvec[np.newaxis, :], x, axis=0)
    ynd = np.repeat(ymat[:, :, np.newaxis], z, axis=2)
    # Make z shape
    zvec = 1 + np.arange(z)
    zmat = np.repeat(zvec[np.newaxis, :], y, axis=0)
    znd = np.repeat(zmat[np.newaxis, :, :], x, axis=0)
    # set offset
    xc = x / 2 + (x * c_sd * np.random.uniform(-1, 1, 1))
    yc = y / 2 + (y * c_sd * np.random.uniform(-1, 1, 1))
    zc = z / 2 + (z * c_sd * np.random.uniform(-1, 1, 1))
    # set radius
    xr = x * r + (x * r_sd * np.random.uniform(-1, 1, 1))
    yr = y * r + (y * r_sd * np.random.uniform(-1, 1, 1))
    zr = z * r + (z * r_sd * np.random.uniform(-1, 1, 1))
    xr = (xr / stretch_factor)
    yr = (yr / stretch_factor)

    # Equations for spheres and ellipsoids
    if cylinder_flag is False:
        sphereI = (
            np.square(xnd - xc) / (xr ** 2)
            + np.square(ynd - yc) / (yr ** 2)
            + np.square(znd - zc) / (zr ** 2)
            < 1
        )
    elif cylinder_flag is True:
        print('Making cylinder')
        sphereI = (
            np.square(xnd - xc) / (xr ** 2)
            + np.square(ynd - yc) / (yr ** 2)
            < 1
        )
        sphereI[:, :, 0] = 0
        sphereI[:, :, -1] = 0

    # re-arrange to ZYX (as expected for AICS images)
    sphereI = np.moveaxis(sphereI,[0, 1, 2],[2, 1, 0])
    sphereI = 255* sphereI.astype('uint8')

    #% Create variables to store results
    column_names = ['Vol. analytically', 'Vol. sum voxels','Vol. mesh', 'Area analytically', 'Area sum contour voxels', 'Area pixelate', 'Area mesh']
    res = pd.DataFrame(columns = column_names)
    res = res.append(pd.Series(), ignore_index=True)

    #% Analytically calculate volume and area
    if cylinder_flag is False:
        # Volume of ellipsoid
        res.loc[res.index[0], 'Vol. analytically'] = 4/3 * np.pi * (xr * yr * zr)
        # Area of ellipsoid
        res.loc[res.index[0], 'Area analytically'] = 4 * np.pi * (((((xr*yr)**1.6) + ((xr*zr)**1.6) + ((yr*zr)**1.6))/3)**(1/(1.6)))
    elif cylinder_flag is True:
        # Volume of cylinder
        res.loc[res.index[0], 'Vol. analytically'] = np.pi * (xr * yr) * (z-2) # did not verify
        # Area of cylinder
        res.loc[res.index[0], 'Area analytically'] = (2 * np.pi * np.sqrt((xr * yr)) * (z-2)) + (2 * np.pi * (xr * yr)) # did not verify

    #% Summing of voxels to calculate volume
    # Volume of ellipsoid
    res.loc[res.index[0], 'Vol. sum voxels'] = np.sum(sphereI==255)

    #% Summing of contour voxels to calculate area
    seg_surface = np.logical_xor(sphereI, binary_erosion(sphereI)).astype(np.uint8)
    res.loc[res.index[0], 'Area sum contour voxels'] = np.count_nonzero(seg_surface)

    #% Summing of outside surfaces using
    pxl_z, pxl_y, pxl_x = np.nonzero(seg_surface)
    dx = np.array([ 0, -1,  0,  1,  0,  0])
    dy = np.array([ 0,  0,  1,  0, -1,  0])
    dz = np.array([-1,  0,  0,  0,  0,  1])
    surface_area = 0
    for (k, j, i) in zip(pxl_z, pxl_y, pxl_x):
        surface_area += 6 - np.sum(sphereI[k+dz,j+dy,i+dx]==255)
    res.loc[res.index[0], 'Area pixelate'] = surface_area

    #% Meshing
    mesh, _, _ = shtools.get_mesh_from_image(
        image = sphereI, # cell membrane
        sigma=0, # no gaussian smooth
        lcc = False, # do not compute largest connected component
        translate_to_origin = True
    )
    # print(f'Number of points in the mesh: {mesh.GetNumberOfPoints()}')
    massp = vtk.vtkMassProperties()
    massp.SetInputData(mesh)
    massp.Update()

    res.loc[res.index[0], 'Vol. mesh'] = massp.GetVolume()
    res.loc[res.index[0], 'Area mesh'] = massp.GetSurfaceArea()

    return res

# %% Compute some analytics
column_names = ['Vol. analytically', 'Vol. sum voxels','Vol. mesh', 'Area analytically', 'Area sum contour voxels', 'Area pixelate', 'Area mesh']
plot_array = pd.DataFrame(columns = column_names)
r_range = np.linspace(0.15,0.3,10)
for i, r in tqdm(enumerate(r_range),'Looping over various radii'):
    rest = compute_areas_and_volumes(r=r, stretch_factor=1, cylinder_flag=False)
    plot_array = plot_array.append(rest, ignore_index=True)

#%%
nobins = 250
pval = 10
x = cells['Nuclear volume'].to_numpy()
y = cells['Nuclear surface area'].to_numpy()
hist,bins = np.histogram(x,nobins)
xi = np.digitize(x, bins, right=False)
z = np.zeros(x.shape)
for i, bin in enumerate(np.unique(xi)):
    pos = np.argwhere(xi==bin)
    posS = np.argwhere(np.all((
        y[pos].squeeze() < np.percentile(y[pos].squeeze(), pval),
        y[pos].squeeze() < np.percentile(y[pos].squeeze(), pval)),axis=0))
    if bins[i]>2e5 and bins[i]<8e5:
            z[pos[posS]] = 1

xs = x[z==1]*((0.108333) ** 3)
ys = y[z==1]*((0.108333) ** 2)

x = x*((0.108333) ** 3)
y = y*((0.108333) ** 2)


# %% measurements
w1 = 0.03
w2 = 0.1
w3 = 0.01
w4 = 0.05
w5 = 0.05

x1 = 0.8
x2 = 1-w1-x1-w2-w3
x3 = 0.3
x3w = 0.03
x4 = 0.3
x5 = 1-w1-x3-w4-x4-w5-w3


h1 = 0.03
h2 = 0.05
h3 = 0.01
y1 = 0
y2 = 1-h1-y1-h2-h3
y2h = 0.05

# %%layout
fs = 10
fig = plt.figure(figsize=(16, 6))
plt.rcParams.update({"font.size": fs})

# # SIM
# axSim = fig.add_axes([w1, h1, x1, y1])
# axSim.text(0.5,0.5,'axSim',horizontalalignment='center')
# axSim.set_xticks([]),axSim.set_yticks([])

# # PCA
# axPCA = fig.add_axes([w1+x1+w2, h1, x2, y1])
# axPCA.text(0.5,0.5,'axPCA',horizontalalignment='center')
# axPCA.set_xticks([]),axPCA.set_yticks([])

# Nuc
axNuc = fig.add_axes([w1, h1+y1+h2, x3, y2])
# Nuc side
axNucS = fig.add_axes([w1-x3w, h1+y1+h2, x3w, y2])
# Nuc bottom
axNucB = fig.add_axes([w1, h1+y1+h2-y2h, x3, y2h])
ps = data_root / statsIN / "cellnuc_struct_metrics"
ps = data_root / statsIN / "cell_nuc_metrics"
ascatter(
    axNuc,
    axNucB,
    axNucS,
    FS['cellnuc_metrics'][4],
    FS['cellnuc_metrics'][3],
    FS['cellnuc_metrics'][4],
    FS['cellnuc_metrics'][3],
    cells,
    ps,
    kde_flag=False,
    fourcolors_flag=False,
    colorpoints_flag=False,
    rollingavg_flag=False,
    ols_flag=False,
    N2=1000,
    fs2=fs,
    fs=fs,
    cell_doubling = [],
    typ=['vol','area']
)

axNuc.plot(xs,ys,'.',color='peru')
axNuc.plot(plot_array['Vol. sum voxels']*((0.108333) ** 3),plot_array['Area pixelate']*((0.108333) ** 2),'m--',linewidth=1)


#% Fitting
xL = xs.copy()
xL = sm.add_constant(xL)
modelL = sm.OLS(ys, xL)
fittedmodelL = modelL.fit()
rsL = fittedmodelL.rsquared
xLplot = np.sort(xs.copy())
yLplot = fittedmodelL.predict(sm.add_constant(xLplot))
axNuc.plot(xLplot,yLplot,'-',color='cyan')

xC = xs.copy()
xC = xC**(2/3)
xC = sm.add_constant(xC)
modelC = sm.OLS(ys, xC)
fittedmodelC = modelC.fit()
rsC = fittedmodelC.rsquared
xCplot = np.sort(xs.copy())
yCplot = fittedmodelC.predict(sm.add_constant(xCplot**(2/3)))
axNuc.plot(xCplot,yCplot,':',color='cyan')

#% Fitting
xL = x.copy()
xL = sm.add_constant(xL)
modelL = sm.OLS(y, xL)
fittedmodelL = modelL.fit()
rsLa = fittedmodelL.rsquared
xLplot = np.sort(x.copy())
yLplot = fittedmodelL.predict(sm.add_constant(xLplot))
axNuc.plot(xLplot,yLplot,'-',color='black')

xC = x.copy()
xC = xC**(2/3)
xC = sm.add_constant(xC)
modelC = sm.OLS(y, xC)
fittedmodelC = modelC.fit()
rsCa = fittedmodelC.rsquared
xCplot = np.sort(x.copy())
yCplot = fittedmodelC.predict(sm.add_constant(xCplot**(2/3)))
axNuc.plot(xCplot,yCplot,':',color='black')


axNuc.legend([f"All cells (n={len(cells)})",f"Cells with spherical nuclei (sn) (n={len(xs)})",
              'Line describing vol. vs. area for perfect spheres',
              f"Linear model for sn (R\u00b2={np.round(100*rsL,2)})",
              f"Non-lin. model with correct scaling (R\u00b2={np.round(100*rsC,2)})",
              f"Linear model for all cells (R\u00b2={np.round(100 * rsLa, 2)})",
              f"Non-lin. model with correct scaling (R\u00b2={np.round(100 * rsCa, 2)})"],loc='lower right',framealpha=1,fontsize=8.8)


# Lin
axLin = fig.add_axes([w1+x3+w4, h1+y1+h2, x4, y2])
CompMat = pd.read_csv(data_root / 'supplementalfiguredata' / 'LinCom_20201202.csv' )
LinPatchColor = [0, 0,  1, .5]
LinPointColor = [0, 0, .5, .5]
ComPatchColor = [1, 0,  0, .5]
ComPointColor = [.5,0,  0, .5]
ylevel = -1
yfac = 0.1
ms = 3
lw = 1
axLin.plot([0, 100], [0, 100], 'k--', linewidth=lw)
for i, typ in enumerate(CompMat['Type'].unique()):
    MatLin = CompMat.loc[CompMat['Type']==typ,['Lin', 'Lin_min', 'Lin_max']].to_numpy()
    MatCom = CompMat.loc[CompMat['Type'] == typ, ['Com', 'Com_min', 'Com_max']].to_numpy()
    order = np.argsort(MatLin[:,0])
    for j, ord in enumerate(order):
        xi = MatLin[ord, [1, 2]]
        yi = MatCom[ord, [1, 2]]
        xc = MatLin[ord, 0]
        yc = MatCom[ord, 0]
        axLin.plot([xc, xc], yi, 'r', linewidth=lw)
        axLin.plot(xi, [yc, yc], 'r', linewidth=lw)
for i, typ in enumerate(CompMat['Type'].unique()):
    MatLin = CompMat.loc[CompMat['Type']==typ,['Lin', 'Lin_min', 'Lin_max']].to_numpy()
    MatCom = CompMat.loc[CompMat['Type'] == typ, ['Com', 'Com_min', 'Com_max']].to_numpy()
    order = np.argsort(MatLin[:,0])
    for j, ord in enumerate(order):
        xi = MatLin[ord, [1, 2]]
        yi = MatCom[ord, [1, 2]]
        xc = MatLin[ord, 0]
        yc = MatCom[ord, 0]
        axLin.plot(xc, yc, 'b.', markersize=ms)
axLin.set_xlim(left=-5,right=100)
axLin.set_ylim(bottom=-5,top=100)
axLin.grid()
MatLin = CompMat[['Lin', 'Lin_min', 'Lin_max']].to_numpy()
MatCom = CompMat[['Com', 'Com_min', 'Com_max']].to_numpy()
fac = 1
pos1 = np.argwhere(MatLin[:,2]<MatCom[:,1])
pos2 = np.argwhere(MatLin[:,0]+fac<MatCom[:,0])
print(f"larger than 1%   {len(np.intersect1d(pos1,pos2))}")
fac = 5
pos1 = np.argwhere(MatLin[:,2]<MatCom[:,1])
pos2 = np.argwhere(MatLin[:,0]+fac<MatCom[:,0])
print(f"larger than 5%   {len(np.intersect1d(pos1,pos2))}")
for i,p in enumerate(np.intersect1d(pos1,pos2)):
    print(f"{CompMat.iloc[p,1]} {CompMat.iloc[p,2]} {CompMat.iloc[p,3]} {CompMat.iloc[p,6]}")




axLin.set_xlabel('Explained Variance for linear models (%)')
axLin.set_ylabel('Explained Variance for non-linear models (%)')



#Scale
axScale = fig.add_axes([w1+x3+w4+x4+w5, h1+y1+h2, x5, y2])
axScale.text(0.5,0.5,'axPCA',horizontalalignment='center')
axScale.set_xticks([]),axScale.set_yticks([])

# plot_save_path = pic_root / f"figure_v8.png"
# plt.savefig(plot_save_path, format="png", dpi=1000)
# plt.close()

plt.show()

# %%


