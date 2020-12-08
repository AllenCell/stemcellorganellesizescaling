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

def workflow_quicktest(N, checkflag):

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
    x = cells['Cell volume'].to_numpy()
    y = cells['Cell surface area'].to_numpy()
    rs = 1
    xS, yS = resample(
                    x, y, replace=False, n_samples=np.amin([int(N), len(x)]), random_state=rs
                )

    # %% Density estimation
    t = time.time()
    k = gaussian_kde(np.vstack([x, y]))
    elapsed = time.time() - t
    print(f"Density estimation with {len(xS)} samples: {np.round(elapsed)}s")
    t = time.time()
    if int(checkflag)==1:
        cell_dens = k(np.vstack([x.flatten(), y.flatten()]))
        elapsed = time.time() - t
        print(f"Assigning density to {len(x)} samples: {np.round(elapsed)}s")

if __name__ == "__main__":
    # Map command line arguments to function arguments.
    workflow_quicktest(*sys.argv[1:])

