#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:49:36 2025

@author: antontalankin
"""
import numpy as np 
import pandas as pd
import re
import os 
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
from tqdm import tqdm  # Import progress bar


file_list=os.listdir()

# Extract numbers before .csv using regex
numbers = [float(re.search(r'_(\d+)\.csv$', fname).group(1)) for fname in file_list if fname.endswith('.csv')]
# Sort the numbers in increasing order
t1 = np.array(sorted(numbers))/1000 # t1 values sorted in increasing order (in sec)

# Filter only .csv files
csv_files = [fname for fname in file_list if fname.endswith('.csv')]

# Sort by the first number in the filename
sorted_files = sorted(csv_files, key=lambda fname: int(re.match(r'(\d+)', fname).group(1)))

# Create a t2 numpy array 
df_t2=pd.read_csv(sorted_files[0])
t2=df_t2['t'].to_numpy()/1000 # t2 values (in sec)

t2=t2



# Create a np - 2d array of values (amplitudes, sorted in increasing order)
amp=[]
for i in sorted_files:
    df=pd.read_csv(i)
    a=df['a'].to_numpy()
    amp.append(a)
    
signal=np.array(amp)

# Create a DataFrame where:
# - The first row contains T1 values as column headers
# - The first column contains T2 values as row headers
df = pd.DataFrame(signal, index=t1, columns=t2)

# Rename index and columns for clarity
df.index.name = "T1 (s)"
df.columns.name = "T2 (s)"

# Save to CSV file
csv_filename = "nmr_t1_t2_data.csv"  # Change filename if needed
df.to_csv(csv_filename)

print(f"CSV file '{csv_filename}' has been saved successfully.")

