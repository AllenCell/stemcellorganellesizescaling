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

# Third party

# Relative

print("Libraries loaded succesfully")
###############################################################################

log = logging.getLogger(__name__)

###############################################################################

# %% function defintion of regression model compensation
def fit_ols(x, y, type, xa=np.nan):
    """
       Fit OLS model

       Parameters
       ----------
       x: S*F numpy array
       y: S*1 numpy array
       type: string indicating linear or complex
       xa: s*F numpy array (optional)

       Output
       ----------
       fittedmodel: object containing fitted regression model
       ya: s*1 numpy array of predicted values

    """
    try:
        S, F = x.shape
    except:
        x = np.expand_dims(x, axis=1)
        S, F = x.shape

    if np.any(np.isnan(xa)):
        ya = np.nan

    if type is "Linear":
        x = sm.add_constant(x)
        model = sm.OLS(y, x)
        fittedmodel = model.fit()
        if not np.any(np.isnan(xa)):
            xa = sm.add_constant(xa)
            ya = fittedmodel.predict(xa)
    elif type is "Complex":
        # first make interaction terms
        for f1 in np.arange(F):
            for f2 in np.arange(F):
                if f1 > f2:
                    x = np.concatenate(
                        (x, np.expand_dims(x[:, f1] * x[:, f2], axis=1)), axis=1
                    )
        # add power terms
        S, F = x.shape
        powers = [1, 1 / 2, 1 / 3, 2, 3]
        x_dm = np.zeros([S, 0])
        for f in np.arange(F):
            for p in powers:
                x_dm = np.concatenate(
                    (x_dm, np.expand_dims(x[:, f] ** p, axis=1)), axis=1
                )
        # remove columns with imaginary values
        remove_columns = np.argwhere(
            np.any(np.iscomplex(x_dm) | np.isnan(x_dm) | np.isinf(x_dm), axis=0)
        )
        x_dm = np.delete(x_dm, remove_columns, axis=1)
        # normalize
        mindm = x_dm.min(0)
        rangedm = x_dm.ptp(0)
        x_dm = (x_dm - mindm) / rangedm
        # all the same for xa is exists
        if not np.any(np.isnan(xa)):
            # first make interaction terms
            for f1 in np.arange(F):
                for f2 in np.arange(F):
                    if f1 > f2:
                        xa = np.concatenate(
                            (xa, np.expand_dims(xa[:, f1] * xa[:, f2], axis=1)), axis=1
                        )
            # add power terms
            Sa, F = xa.shape
            powers = [1, 1 / 2, 1 / 3, 2, 3]
            xa_dm = np.zeros([Sa, 0])
            for f in np.arange(F):
                for p in powers:
                    xa_dm = np.concatenate(
                        (xa_dm, np.expand_dims(xa[:, f] ** p, axis=1)), axis=1
                    )
            # remove columns with imaginary values
            remove_columns_a = np.argwhere(
                np.any(np.iscomplex(xa_dm) | np.isnan(xa_dm) | np.isinf(xa_dm), axis=0)
            )
            xa_dm = np.delete(xa_dm, remove_columns_a, axis=1)
            # check
            if remove_columns.size > 0 or remove_columns_a.size > 0:
                if not np.array_equal(remove_columns, remove_columns_a):
                    1 / 0
            # normalize
            xa_dm = (xa_dm - mindm) / rangedm

        x_dm = sm.add_constant(x_dm)
        model = sm.OLS(y, x_dm)
        fittedmodel = model.fit()
        if not np.any(np.isnan(xa)):
            xa_dm = sm.add_constant(xa_dm)
            ya = fittedmodel.predict(xa_dm)

    return fittedmodel, ya


# %% function defintion of regression model compensation
def calculate_pairwisestats(x, y, xlabel, ylabel, struct):
    """
       Calculate residual values

       Parameters
       ----------
       x: S*1 numpy array
       y: S*1 numpy array
       xlabel: label of x array
       ylabel: label of y array
       struct: Name of structure of 'None' if across all structures

       Output
       ----------
       D: dictionary with the following values:
           xi: nbins*1 sampled values on x
           rs_vecL: Nbootstrap*1 r-squared values simple linear model
           pred_matL: nbins*1 average prediction values for xi for simple linear model
           rs_vecC: Nbootstrap*1 r-squared values complex design matrix model
           pred_matC: nbins*1 average prediction values for xi for complex design matrix model
           xii: nbins*nbins samples values on x
           yii: nbins*nbins samples values on y
           zii: nbins*nbins density estimates on the x y grid
           cell_dens: S*1 cell-specific densities
           x_ra: (nbins-1)*1 bins on x
           y_ra: (nbins-1)*6 mean and 5,25,50,75,95 percentile values on y
    """
    # Parameters
    # Nbootstrap = 100
    Nbootstrap = 5
    nbins = 100
    N = 10000
    # N = 1000
    minbinsize = 50
    rs = int(datetime.datetime.utcnow().timestamp())

    S, F = x.shape

    # sampling on x and y
    xii, yii = np.mgrid[x.min() : x.max() : nbins * 1j, y.min() : y.max() : nbins * 1j]
    xi = xii[:, 0]
    xi = np.expand_dims(xi, axis=1)

    # bootstrap regression - make arrays
    rs_vecL = np.zeros([Nbootstrap, 1])
    pred_matL = np.zeros([nbins, Nbootstrap])
    rs_vecC = np.zeros([Nbootstrap, 1])
    pred_matC = np.zeros([nbins, Nbootstrap])

    for i in tqdm(range(Nbootstrap), "Bootstrapping"):
        # bootstrap and prepare design matrices
        xx, yy = resample(x, y)
        modelL, yiL = fit_ols(xx, yy, "Linear", xa=xi)
        rs_vecL[i] = modelL.rsquared
        pred_matL[:, i] = yiL
        modelC, yiC = fit_ols(xx, yy, "Complex", xa=xi)
        rs_vecC[i] = modelC.rsquared
        pred_matC[:, i] = yiC

    pred_matL = np.mean(pred_matL, axis=1)
    pred_matC = np.mean(pred_matC, axis=1)

    # density estimate
    xS, yS = resample(
        x.squeeze(),
        y.squeeze(),
        replace=False,
        n_samples=np.amin([N, len(x)]),
        random_state=rs,
    )
    k = gaussian_kde(np.vstack([xS, yS]))
    zii = k(np.vstack([xii.flatten(), yii.flatten()]))
    cell_dens = k(np.vstack([x.flatten(), y.flatten()]))
    # make into cumulative sum
    zii = zii / np.sum(zii)
    ix = np.argsort(zii)
    zii = zii[ix]
    zii = np.cumsum(zii)
    jx = np.argsort(ix)
    zii = zii[jx]
    zii = zii.reshape(xii.shape)

    # rolling average
    idx = np.digitize(x.squeeze(), xi.squeeze())
    x_ra = np.zeros((nbins - 1, 1))
    y_ra = np.zeros((nbins - 1, 6))
    for n in range(nbins - 1):
        x_ra[n] = np.mean([xi[n], xi[n + 1]])
        sc = np.argwhere(idx == (n + 1))
        if len(sc) < minbinsize:
            y_ra[n, :] = np.nan
        else:
            y_ra[n, 0] = np.mean(y[sc])
            y_ra[n, 1:6] = np.percentile(y[sc], [5, 25, 50, 75, 95])

    # Fill dictionary
    D = {}
    if struct == "None":
        D[f"{xlabel}_{ylabel}_xi"] = xi
        D[f"{xlabel}_{ylabel}_rs_vecL"] = rs_vecL
        D[f"{xlabel}_{ylabel}_pred_matL"] = pred_matL
        D[f"{xlabel}_{ylabel}_rs_vecC"] = rs_vecC
        D[f"{xlabel}_{ylabel}_pred_matC"] = pred_matC
        D[f"{xlabel}_{ylabel}_xii"] = xii
        D[f"{xlabel}_{ylabel}_yii"] = yii
        D[f"{xlabel}_{ylabel}_zii"] = zii
        D[f"{xlabel}_{ylabel}_cell_dens"] = cell_dens
        D[f"{xlabel}_{ylabel}_x_ra"] = x_ra
        D[f"{xlabel}_{ylabel}_y_ra"] = y_ra
    else:
        D[f"{xlabel}_{ylabel}_{struct}_xi"] = xi
        D[f"{xlabel}_{ylabel}_{struct}_rs_vecL"] = rs_vecL
        D[f"{xlabel}_{ylabel}_{struct}_pred_matL"] = pred_matL
        D[f"{xlabel}_{ylabel}_{struct}_rs_vecC"] = rs_vecC
        D[f"{xlabel}_{ylabel}_{struct}_pred_matC"] = pred_matC
        D[f"{xlabel}_{ylabel}_{struct}_xii"] = xii
        D[f"{xlabel}_{ylabel}_{struct}_yii"] = yii
        D[f"{xlabel}_{ylabel}_{struct}_zii"] = zii
        D[f"{xlabel}_{ylabel}_{struct}_cell_dens"] = cell_dens
        D[f"{xlabel}_{ylabel}_{struct}_x_ra"] = x_ra
        D[f"{xlabel}_{ylabel}_{struct}_y_ra"] = y_ra

    return D

# %% function defintion of regression model compensation
def explain_var_compositemodels(x, y, xlabel, ylabel, struct):
    """
       Calculate residual values

       Parameters
       ----------
       x: S*N numpy array
       y: S*1 numpy array
       xlabel: label of x array
       ylabel: label of y array
       struct: Name of structure of 'None' if across all structures

       Output
       ----------
       D: dictionary with the following values:
           rs_vecL: Nbootstrap*1 r-squared values simple linear model
           rs_vecC: Nbootstrap*1 r-squared values complex design matrix model

    """
    # Parameters
    # Nbootstrap = 100
    Nbootstrap = 5

    # bootstrap regression - make arrays
    rs_vecL = np.zeros([Nbootstrap, 1])
    rs_vecC = np.zeros([Nbootstrap, 1])

    for i in tqdm(range(Nbootstrap), "Bootstrapping"):
        # bootstrap and prepare design matrices
        xx, yy = resample(x, y)
        modelL, _ = fit_ols(xx, yy, "Linear")
        rs_vecL[i] = modelL.rsquared
        modelC, _ = fit_ols(xx, yy, "Complex")
        rs_vecC[i] = modelC.rsquared

    # Fill dictionary
    D = {}
    if struct == "None":
        D[f"{xlabel}_{ylabel}_rs_vecL"] = rs_vecL
        D[f"{xlabel}_{ylabel}_rs_vecC"] = rs_vecC
    else:
        D[f"{xlabel}_{ylabel}_{struct}_rs_vecL"] = rs_vecL
        D[f"{xlabel}_{ylabel}_{struct}_rs_vecC"] = rs_vecC

    return D

# %% function defintion of bootstrapping the regression model
def bootstrap_linear_and_log_model(x, y, xlabel, ylabel, type, cell_doubling, struct, Nbootstrap=100):
    """
       Calculate residual values

       Parameters
       ----------
       x: S*1 numpy array
       y: S*1 numpy array
       xlabel: label of x array
       ylabel: label of y array
       type: Linear or Complex model
       cell_doubling: cell volume in voxels of cell doubling
       struct: Name of structure of 'None' if across all structures
       Nbootstrap: Number of bootstraps (default is 100)

       Output
       ----------
       Scale_rates: N (no of bootstraps) by 2 (linear and log-log model)
       Scale_plot_data - dictionary


    """
    # Parameters
    # Nbootstrap = 5

    # bootstrap regression - make arrays
    Scale_rates = np.zeros([Nbootstrap, 2])

    for i in tqdm(range(Nbootstrap), "Bootstrapping"):
        # bootstrap and prepare design matrices
        xx, yy = resample(x, y)
        model, _ = fit_ols(xx, yy, type)
        xC = cell_doubling.copy()
        xC = sm.add_constant(xC)
        yC = model.predict(xC)
        Scale_rates[i, 0] = np.round(100 * (yC[1] - yC[0]) / yC[0], 2)
        model_ll, _ = fit_ols(np.log2(xx), np.log2(yy), type)
        Scale_rates[i, 1] = np.round(100 * model_ll.params[1], 2)

    return Scale_rates
