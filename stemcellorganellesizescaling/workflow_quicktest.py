#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library
import logging
from pathlib import Path
import sys, importlib
import os, platform
import pandas as pd
import numpy as np
import time
from sklearn.utils import resample
from scipy.stats import gaussian_kde


# Third party

print("Libraries loaded succesfully")
###############################################################################

log = logging.getLogger(__name__)

###############################################################################

def workflow_quicktest(N, N2):

    # N = 1e4
    fac = 1
    # N2 = 1e6

    #%% Directories
    if platform.system() == "Windows":
        data_root = Path("E:/DA/Data/scoss/Data/")
        pic_root = Path("E:/DA/Data/scoss/Pics/")
    elif platform.system() == "Linux":
        data_root = Path(
            "/allen/aics/modeling/theok/Projects/Data/scoss/Data"
        )
        pic_root = Path(
            "/allen/aics/modeling/theok/Projects/Data/scoss/Pics"
        )

    #%%  Test outlier stuff
    dataset = "SizeScaling_20201006.csv"
    cells = pd.read_csv(data_root / dataset)

    # %% Sample
    x = cells['Cell volume'].to_numpy() / fac
    y = cells['Cell surface area'].to_numpy() /fac
    rs = 1
    xS, yS = resample(
                    x, y, replace=False, n_samples=np.amin([int(N), len(x)]), random_state=rs
                )
    xS2, yS2 = resample(
                    x, y, replace=False, n_samples=np.amin([int(N2), len(x)]), random_state=rs
                )

    # %% Density estimation
    t = time.time()
    print(np.vstack([xS, yS]).shape)
    k = gaussian_kde(np.vstack([xS, yS]))
    elapsed = time.time() - t
    print(f"Density estimation with {len(xS)} samples: {np.round(elapsed)}s")
    if int(N2)>0:
        t = time.time()
        print(np.vstack([xS2, yS2]).shape)
        cell_dens = k(np.vstack([xS2.flatten(), yS2.flatten()]))
        elapsed = time.time() - t
        print(f"Assigning density to {len(xS2)} samples: {np.round(elapsed)}s")

if __name__ == "__main__":
    # Map command line arguments to function arguments.
    workflow_quicktest(*sys.argv[1:])

