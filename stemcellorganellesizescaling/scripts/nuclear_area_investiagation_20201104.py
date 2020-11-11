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
    data_root = Path("E:/DA/Data/scoss/Data/Nov2020")
    pic_root = Path("E:/DA/Data/scoss/Pics/Nov2020")
elif platform.system() == "Linux":
    data_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Data/Nov2020")
    pic_root = Path("/allen/aics/modeling/theok/Projects/Data/scoss/Pics/Nov2020")
dirs = []
dirs.append(data_root)
dirs.append(pic_root)

# %% Start

# Resolve directories
data_root = dirs[0]
pic_root = dirs[1]

tableIN = "SizeScaling_20201102.csv"
table_compIN = "SizeScaling_20201102_comp.csv"
statsIN = "Stats_20201102"
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


# %%
xvec = [4]
yvec = [3]
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
plotname = "test"
ps = data_root / statsIN / "cell_nuc_metrics"
fscatter(
    FS["cellnuc_metrics"],
    FS["cellnuc_abbs"],
    pair1,
    cells,
    ps,
    False,
    pic_root,
    f"{plotname}_plain",
    kde_flag=False,
    fourcolors_flag=False,
    colorpoints_flag=True,
    rollingavg_flag=True,
    ols_flag=True,
    N2=1000,
)

# %% Loop
column_names = ['Vol. analytically', 'Vol. sum voxels','Vol. mesh', 'Area analytically', 'Area sum contour voxels', 'Area pixelate', 'Area mesh']
plot_array = pd.DataFrame(columns = column_names)
r_range = np.linspace(0.15,0.3,10)
for i, r in tqdm(enumerate(r_range),'Looping over various radii'):
    rest = compute_areas_and_volumes(r=r, stretch_factor=1, cylinder_flag=False)
    plot_array = plot_array.append(rest, ignore_index=True)

#%%
x = cells['Nuclear volume']
y = cells['Nuclear surface area']
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(x,y,'r.',markersize=5)
ax.grid()
ax.plot(plot_array['Vol. analytically'],plot_array['Area analytically'],'b',linewidth=5)
ax.plot(plot_array['Vol. sum voxels'],plot_array['Area pixelate'],'g',linewidth=5)
ax.plot(plot_array['Vol. mesh'],plot_array['Area mesh'],'m',linewidth=5)
ax.legend(['Cells','Analytically','Current metrics','Mesh'])
ax.set_title('Sphere')
ax.set_xlabel('Nuclear volume')
ax.set_ylabel('Nuclear surface area')
plt.show()

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

xs = x[z==1]
ys = y[z==1]

fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(x,y,'.',markersize=.5,color = 'gray')
ax.plot(xs,ys,'b.',markersize=5)
ax.grid()


# %


#% Fitting
xL = xs.copy()
xL = sm.add_constant(xL)
modelL = sm.OLS(ys, xL)
fittedmodelL = modelL.fit()
rsL = fittedmodelL.rsquared
yL = fittedmodelL.predict(xL)
ax.plot(xs,yL,'r.',markersize=1)

xC = xs.copy()
xC = xC**(2/3)
xC = sm.add_constant(xC)
modelC = sm.OLS(ys, xC)
fittedmodelC = modelC.fit()
rsC = fittedmodelC.rsquared
yC = fittedmodelC.predict(xC)
ax.plot(xs,yC,'w.',markersize=1)

plt.show()

# %% Loop

xL = x.copy()
xL = sm.add_constant(xL)
modelL = sm.OLS(y, xL)
fittedmodelL = modelL.fit()
rsL = fittedmodelL.rsquared
print(rsL)

xC = x.copy()
xC = xC**(2/3)
xC = sm.add_constant(xC)
modelC = sm.OLS(y, xC)
fittedmodelC = modelC.fit()
rsC = fittedmodelC.rsquared
print(rsC)

# %% Loop
x = cells['Cell volume']
y = cells['Cell surface area']

xL = x.copy()
xL = sm.add_constant(xL)
modelL = sm.OLS(y, xL)
fittedmodelL = modelL.fit()
rsL = fittedmodelL.rsquared
print(rsL)

xC = x.copy()
xC = xC**(2/3)
xC = sm.add_constant(xC)
modelC = sm.OLS(y, xC)
fittedmodelC = modelC.fit()
rsC = fittedmodelC.rsquared
print(rsC)
