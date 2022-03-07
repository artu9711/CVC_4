import numpy as np
import math
import pandas as pd
import os
import time
from glob import glob
import shutil
import sys

with open('execution_time.txt', 'r') as f:
    last_time_running = int(f.read())
if last_time_running > 20.0:
    update_parameters()

last_run_file = max(last_params_path+'/*', key=os.path.getctime)
with open(filename, 'rb') as f:
     spikes_last_running = pickle.load(f)['exc_spikes_from']
if spikes_last_running > 98735:
    update_parameters()

path, dirs, files = next(os.walk(path)); len_runnings = len(files)
if len_runnings < 8: 
    quit()

best_params_path = 'best_params'
last_params_path = 'last_params'

def get_df(path):
    keys = ['image_name','exc_spikes_from','inh_spikes_from', 'node_exc', 'gamma_power_exc','node_tot','gamma_power_tot', 'seed']
    output = pd.DataFrame()
    for filename in glob.iglob(f'{path}/*'):
        results_dict = {}
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        results_dict = dict((k, data[k]) for k in keys if k in data)
        output = output.append(results_dict, ignore_index=True)
    df = output.dropna()
    replacement_mapping_dict = {"sin_5": 2.5, "sin_6": 3, "sin_7": 3.5,"sin_8": 4, "sin_9": 4.5, "sin_10": 5,"sin_11": 5.5, "sin_12": 6}
    df["image_name"] = df["image_name"].replace(replacement_mapping_dict)
    df.rename({'image_name': 'size'}, axis=1, inplace=True)
    df = df.sort_values(by='size')
    return df

def get_score(means_spikes,means_gammas):
    score_spikes = scipy.stats.spearmanr(means_spikes,[2.5,3,3.5,4,4.5,5,5.5,6])
    score_gammas = scipy.stats.spearmanr(means_gammas,[2.5,3,3.5,4,4.5,5,5.5,6])  
    return score_gammas - score_spikes

df_best_params = get_df(best_params_path)
df_last_params = get_df(last_params_path)

best_score = get_score(df_best_params['exc_spikes_from'],df_best_params['gamma_power_exc'])
last_score = get_score(df_last_params['exc_spikes_from'],df_last_params['gamma_power_exc'])

if last_score > best_score:
    save_best_parameters()
    remove_contents(best_params_path)
    move_contents(last_params_path,best_params_path)


def move_contents(origin_path_name,goal_path_name):
    for filename in os.listdir(origin_path_name):
        file_path = os.path.join(origin_path_name, filename)
        shutil.move(file_path + ".foo",os.path.join(goal_path_name, filename) + ".foo")

update_parameters()

def save_best_parameters(parameters_dict):
    save_dict(parameters_dict,'best_parameters')
  
def update_parameters(old_parameters_dict):
    new_parameters_dict = {}
    new_parameters_dict['weight_large_range_exc_exc'] = old_parameters_dict['weight_large_range_exc_exc'] + (-1)**np.random.randint(0,2) * 
    while new_parameters_dict['weight_large_range_exc_exc'] < 0.0 or new_parameters_dict['weight_large_range_exc_exc'] > 0.05:
        new_parameters_dict['weight_large_range_exc_exc'] = old_parameters_dict['weight_large_range_exc_exc'] + (-1)**np.random.randint(0,2) * 
     
    new_parameters_dict['weight_large_range_exc_inh'] = old_parameters_dict['weight_large_range_exc_inh'] + (-1)**np.random.randint(0,2) * 
    while new_parameters_dict['weight_large_range_exc_inh'] < 0.0 or new_parameters_dict['weight_large_range_exc_inh'] > 0.1:
        new_parameters_dict['weight_large_range_exc_inh'] = old_parameters_dict['weight_large_range_exc_inh'] + (-1)**np.random.randint(0,2) * 
    
    new_parameters_dict['weight_inh_exc_'] = old_parameters_dict['weight_inh_exc_'] + (-1)**np.random.randint(0,2) * 
    while new_parameters_dict['weight_inh_exc_'] < 0.0 or new_parameters_dict['weight_inh_exc_'] > 0.1:
        new_parameters_dict['weight_inh_exc_'] = old_parameters_dict['weight_inh_exc_'] + (-1)**np.random.randint(0,2) * 
    
    new_parameters_dict['weight_inh_inh_'] = old_parameters_dict['weight_inh_inh_'] + (-1)**np.random.randint(0,2) * 
    while new_parameters_dict['weight_inh_exc_'] < 0.0 or new_parameters_dict['weight_inh_exc_'] > 0.1:
        new_parameters_dict['weight_inh_exc_'] = old_parameters_dict['weight_inh_exc_'] + (-1)**np.random.randint(0,2) * 
    
    new_parameters_dict['weight_exc_exc'] = old_parameters_dict['weight_exc_exc'] + (-1)**np.random.randint(0,2) * 
    while new_parameters_dict['weight_inh_exc_'] < 0.0 or new_parameters_dict['weight_inh_exc_'] > 0.1:
        new_parameters_dict['weight_inh_exc_'] = old_parameters_dict['weight_inh_exc_'] + (-1)**np.random.randint(0,2) * 
    
    new_parameters_dict['weight_exc_inh'] = old_parameters_dict['weight_exc_inh'] + (-1)**np.random.randint(0,2) * 
    while new_parameters_dict['weight_inh_exc_'] < 0.0 or new_parameters_dict['weight_inh_exc_'] > 0.1:
        new_parameters_dict['weight_inh_exc_'] = old_parameters_dict['weight_inh_exc_'] + (-1)**np.random.randint(0,2) * 
    
    new_parameters_dict['input_weight_poiss_inh'] = old_parameters_dict['input_weight_poiss_inh'] + (-1)**np.random.randint(0,2) * 
    while new_parameters_dict['weight_inh_exc_'] < 0.0 or new_parameters_dict['weight_inh_exc_'] > 0.1:
        new_parameters_dict['weight_inh_exc_'] = old_parameters_dict['weight_inh_exc_'] + (-1)**np.random.randint(0,2) * 
    
    return new_parameters_dict
    
    
   

  
  






