import numpy as np
import math
import pandas as pd
import os
import time
import glob
import shutil
import sys
import ast
import pickle
from scipy import stats

best_params_path = 'best_params'
last_params_path = 'last_params'

####### funciones

def get_df(path):
    keys = ['image_name','exc_spikes_from','inh_spikes_from', 'node_exc', 'gamma_power_exc','node_tot','gamma_power_tot', 'seed']
    output = pd.DataFrame()
    for filename in glob.iglob(f'{path}/results_*'):
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
    score_spikes, pvalue_spikes = stats.spearmanr(means_spikes,[2.5,3,3.5,4,4.5,5,5.5,6])
    score_gammas, pvalue_gammas = stats.spearmanr(means_gammas,[2.5,3,3.5,4,4.5,5,5.5,6])  
    return score_gammas*(1-pvalue_gammas) - score_spikes*(1-pvalue_spikes), score_spikes, score_gammas

def create_folder(path_name):
    if not os.path.exists(path_name):
        os.makedirs(path_name)

def remove_contents(path_name):
    for filename in os.listdir(path_name):
        file_path = os.path.join(path_name, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            continue

def move_contents(origin_path_name,goal_path_name):
    for filename in os.listdir(origin_path_name):
        file_path = os.path.join(origin_path_name, filename)
        shutil.move(file_path ,os.path.join(goal_path_name, filename))
        
def save_parameters_to_txt(parameters_dict, filename):
    with open(filename, 'w') as f:
        print(parameters_dict, file=f)
        
def read_parameters_from_txt(filename):
    file = open(filename, "r")
    contents = file.read()
    dictionary = ast.literal_eval(contents)
    file.close()
    return dictionary
  
def update_parameters(old_parameters_dict):
    new_parameters_dict = {}
    
    new_parameters_dict['weight_large_range_exc_exc'] = old_parameters_dict['weight_large_range_exc_exc'] + (-1)**np.random.randint(0,2) * np.random.uniform(0,0.001)
    while new_parameters_dict['weight_large_range_exc_exc'] < 0.0 or new_parameters_dict['weight_large_range_exc_exc'] > 0.05:
        new_parameters_dict['weight_large_range_exc_exc'] = old_parameters_dict['weight_large_range_exc_exc'] + (-1)**np.random.randint(0,2) * np.random.uniform(0,0.001)
     
    new_parameters_dict['weight_large_range_exc_inh'] = old_parameters_dict['weight_large_range_exc_inh'] + (-1)**np.random.randint(0,2) * np.random.uniform(0,0.005)
    while new_parameters_dict['weight_large_range_exc_inh'] < 0.0 or new_parameters_dict['weight_large_range_exc_inh'] > 0.1:
        new_parameters_dict['weight_large_range_exc_inh'] = old_parameters_dict['weight_large_range_exc_inh'] + (-1)**np.random.randint(0,2) * np.random.uniform(0,0.005)
    
    new_parameters_dict['weight_inh_exc_'] = old_parameters_dict['weight_inh_exc_'] + (-1)**np.random.randint(0,2) * np.random.uniform(0,0.1)
    while new_parameters_dict['weight_inh_exc_'] < 0.01 or new_parameters_dict['weight_inh_exc_'] > 1.0:
        new_parameters_dict['weight_inh_exc_'] = old_parameters_dict['weight_inh_exc_'] + (-1)**np.random.randint(0,2) * np.random.uniform(0,0.1)
    
    new_parameters_dict['weight_inh_inh_'] = old_parameters_dict['weight_inh_inh_'] + (-1)**np.random.randint(0,2) * np.random.uniform(0,0.1)
    while new_parameters_dict['weight_inh_inh_'] < 0.01 or new_parameters_dict['weight_inh_inh_'] > 1.0:
        new_parameters_dict['weight_inh_inh_'] = old_parameters_dict['weight_inh_inh_'] + (-1)**np.random.randint(0,2) * np.random.uniform(0,0.1)
    
    new_parameters_dict['weight_exc_exc'] = old_parameters_dict['weight_exc_exc'] + (-1)**np.random.randint(0,2) * np.random.uniform(0,0.1)
    while new_parameters_dict['weight_exc_exc'] < 0.01 or new_parameters_dict['weight_exc_exc'] > 1.0:
        new_parameters_dict['weight_exc_exc'] = old_parameters_dict['weight_exc_exc'] + (-1)**np.random.randint(0,2) * np.random.uniform(0,0.1)
    
    new_parameters_dict['weight_exc_inh'] = old_parameters_dict['weight_exc_inh'] + (-1)**np.random.randint(0,2) * np.random.uniform(0,0.1)
    while new_parameters_dict['weight_exc_inh'] < 0.01 or new_parameters_dict['weight_exc_inh'] > 1.0:
        new_parameters_dict['weight_exc_inh'] = old_parameters_dict['weight_exc_inh'] + (-1)**np.random.randint(0,2) * np.random.uniform(0,0.1)
    
    new_parameters_dict['input_weight_poiss_inh'] = old_parameters_dict['input_weight_poiss_inh'] + (-1)**np.random.randint(0,2) * np.random.uniform(0,0.1)
    while new_parameters_dict['input_weight_poiss_inh'] < 0.1 or new_parameters_dict['input_weight_poiss_inh'] > 1.0:
        new_parameters_dict['input_weight_poiss_inh'] = old_parameters_dict['input_weight_poiss_inh'] + (-1)**np.random.randint(0,2) * np.random.uniform(0,0.1)
    
    return new_parameters_dict



####### algoritmo
create_folder(best_params_path)
create_folder(last_params_path)

best_parameters = read_parameters_from_txt(best_params_path + '/best_parameters.txt')

with open('execution_time.txt', 'r') as f:
    last_execution_time = int(f.read())
if last_execution_time > 20.0:
    remove_contents(last_params_path)
    new_parameters = update_parameters(best_parameters)
    save_parameters_to_txt(new_parameters, last_params_path + '/last_parameters.txt')
    quit()

run_paths = glob.glob(last_params_path+'/results_*')
if run_paths != []:
    last_run_file = max(run_paths , key=os.path.getctime)
    with open(last_run_file, 'rb') as f:
         last_spikes = pickle.load(f)['exc_spikes_from']
    if last_spikes > 10000:
        remove_contents(last_params_path)
        new_parameters = update_parameters(best_parameters)
        quit()

path, dirs, files = next(os.walk(last_params_path)); num_runnings = len(files)
if num_runnings < 8: 
    quit()
print("FINAL IFS")

df_best_params = get_df(best_params_path)
df_last_params = get_df(last_params_path)

print("FINAL DFS")

best_score, best_score_spikes, best_score_gammas = get_score(df_best_params['exc_spikes_from'],df_best_params['gamma_power_exc'])
last_score, last_score_spikes, last_score_gammas = get_score(df_last_params['exc_spikes_from'],df_last_params['gamma_power_exc'])

print("FINAL SCORES")

if last_score > best_score and last_score_spikes < 0 and last_score_gammas > 0.0:
    remove_contents(best_params_path)
    save_parameters_to_txt(last_parameters, best_params_path + '/best_parameters.txt')
    os.remove(last_params_path + '/last_parameters.txt')
    move_contents(last_params_path,best_params_path)
    
print("FINAL SI ES MEJOR")
    
new_parameters = update_parameters(best_parameters)
save_parameters_to_txt(new_parameters, last_params_path + '/last_parameters.txt')
print("FINAL")

    
    
   

  
  






