"""
Plot planes from joint analysis files.

Usage:
    plot_analysis.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from dedalus.extras import plot_tools


    # Plot writes
with h5py.File("analysis/analysis_s1.h5", mode='r') as file:
    # Load datasets
    E = file['tasks']['<E>']
    t = E.dims[0]['sim_time']

    # convert time array into dataframe and save to file
    DF = pd.DataFrame(t[:]) 
    DF.to_csv("t.out", sep = ' ')
    
    # Save to ascii files
    np.savetxt("E.out", E[:,0,0], delimiter=" ")

# Reload time file and remove first line...
with open('t.out', 'r') as fin:
    data = fin.read().splitlines(True)
with open('t.out', 'w') as fout:
    fout.writelines(data[1:])    
    

