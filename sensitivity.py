# -*- coding: utf-8 -*-

from epw import epw
from uwg import UWG
import numpy as np
import math
from numba import jit
from SALib.sample import sobol
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
from SALib.sample import morris as ms
from SALib.analyze import morris as ma
from SALib.plotting import morris as mp
from SALib.sample import saltelli
import pandas as pd

epw_path_a = "resources/GT_33.77463936796479_-84.39704008595767_2020.epw"

model = UWG.from_param_args(epw_path=epw_path_a, bldheight=10, blddensity=0.5,
                            vertohor=0.8, grasscover=0.1, treecover=0.1, zone='7') 

model.generate()
model.simulate()

def calculate_degree_days(hourly_temps, base_temp=18.3):
    """
    Calculate Heating Degree Days (HDD) and Cooling Degree Days (CDD) from hourly temperature data.

    Parameters:
    - hourly_temps: list of 8760 hourly temperature values in degrees Fahrenheit.
    - base_temp: base temperature for calculating HDD and CDD, default is 18.3°C.
    Returns:
    - A tuple containing total HDD and CDD.
    """
    # Ensure the input list has 8760 values (24 hours * 365 days)
    if len(hourly_temps) != 8760:
        raise ValueError("Input list must contain 8760 hourly temperature values.")

    # Calculate daily average temperatures
    daily_avgs = [sum(hourly_temps[i:i+24])/24 for i in range(0, 8760, 24)]
    # Calculate HDD and CDD

    hdd = sum(base_temp - temp for temp in daily_avgs if temp < base_temp)
    cdd = sum(temp - base_temp for temp in daily_avgs if temp > base_temp)

    return hdd, cdd


# Example hourly temperature data (replace this with your actual data)
hourly_temps = [60] * 8760  # Example data, replace with your real temperatures

# Calculate HDD and CDD
hdd, cdd = calculate_degree_days(hourly_temps)
print(f"Heating Degree Days: {hdd}, Cooling Degree Days: {cdd}")

np.array([1,2,3]) - np.array([3,4,5])



@jit(nopython=True)
def rmse(y1, y2):
    return np.linalg.norm(y1 - y2) / np.sqrt(len(y1))


def get_SA_metric(epw_path_before_sim, simulated_UWG_model, days_simulated):
    e=epw()
    e.read(epw_path_before_sim)
    epw_df = e.dataframe
    if len(e.dataframe) > 8760:
        epw_df = e.dataframe.drop([i+1417 for i in range(24)])

    simulated_UWG_model.UCMData[0].canTemp-273.15 # This is the canyon air temperature it seems
    ambTemp = [model.UCMData[hour].canTemp-273.15  for hour in range(days_simulated*24) ]

    return rmse(np.array(epw_df['Dry Bulb Temperature'].values[:(24*days_simulated)]), np.array(ambTemp))

get_SA_metric(epw_path_a, model, 10)


test_problem = {
    'num_vars': 5,
    'names': ['bldheight', 'blddensity', 'vertohor', 'grasscover', 'treecover'],
    'bounds': [[3,100],
               [0.05, 1],
               [0.1, 1],
               [0, 1],
               [0, 1],
               ],
    'groups': None
}


# We'll perform a Sobol Sensitivity Analysis below.
# Generate samples
param = saltelli.sample(test_problem, 100, calc_second_order=True)

#edit - changed calc_second_order to true

param[:, 1:4] = np.clip(param[:, 1:4], 0, 1)
param[:, 1:4] /= param[:, 1:4].sum(axis=1, keepdims=True)

pd.DataFrame(param, columns = test_problem['names'])

rand_vals = np.random.uniform(0.1, 0.9, size=(param.shape[0], 3))
param[:, [1, 3, 4]] = rand_vals / rand_vals.sum(axis=1, keepdims=True)

df = pd.DataFrame(param, columns=test_problem['names'])

sim_days = 1


i = 0

bldheight=float(df.iloc[i][0].round(4))
print("bldheight", bldheight)

blddensity=float(df.iloc[i][1].round(4))
print("blddensity", blddensity)

vertohor=float(df.iloc[i][2].round(4))
print("vertohor", vertohor)

grasscover=float(df.iloc[i][3].round(4))
print("grasscover", grasscover)

treecover=float(df.iloc[i][4].round(4))
print("treecover", treecover)


# Run model (example)
Y = np.zeros([df.shape[0]])

for i, X in enumerate(df.iterrows()):

    print("bldheight", bldheight)

    blddensity=float(df.iloc[i][1].round(4))
    print("blddensity", blddensity)

    vertohor=float(df.iloc[i][2].round(4))
    print("vertohor", vertohor)

    grasscover=float(df.iloc[i][3].round(4))
    print("grasscover", grasscover)

    treecover=float(df.iloc[i][4].round(4))
    print("treecover", treecover)

    print("###")
    print("RUN", i)

    try:    

        model1 = UWG.from_param_args(epw_path=epw_path_a, bldheight=bldheight, blddensity=blddensity, vertohor=vertohor, grasscover=grasscover, treecover=treecover, zone='7', nday= sim_days) # Atlanta is in climate zone 7  --> https://www.walterreeves.com/landscaping/hardiness-zones-for-georgia/

        model1.generate()
        model1.simulate()
        
        Y[i] = get_SA_metric(epw_path_a, model1, sim_days)
    
    except: 
        Y[i] = None
        print("Some issue with inputs.")