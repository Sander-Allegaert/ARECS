# -*- coding: utf-8 -*-
"""
Created on Thu May 13 22:16:15 2021

@author: SAL
"""
#%%
import time
import copy
import math
import itertools
import numpy as np
import sympy as sym
import pandas as pd
import FeatureExtraction as ft
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import warnings
from sys import exit
import os
from pathlib import Path
import Tool_v5_2021_05_17_SRE as Tool

#warnings.filterwarnings('ignore')

#%%
"""
Input
"""

Tool_path = Path().absolute()

name_input = 'Input_EtOH_275_0.88_no_outlet.xlsx'

df_input = pd.read_excel(Tool_path / 'Input' / name_input)
col_names = df_input.columns.values.tolist()

# Input from user asked: Species_total and Overall_stoich | Important that the names used for the initial flowrates columns in the excel input file are of the following format: In_XXX, with XXX the species name used in Species_total


Species_total = ['EtOH','ACET','H2','H2O'] # Ethanol dehydrogenation case study
Overall_stoich = {'EtOH':{1:-1},'ACET':{1:1},'H2':{1:1},'H2O':{1:0}} # Ethanol dehydrogenation case study

# Extracting the initial flowrates, output flowrates and selectivities from the input excel file into a dictionary for later use
Initial_flowrates_overalldict = {}
Exp_Output_flowrates_overalldict = {}
Exp_Select_dict = {}
for species in Species_total:
    Initial_flowrates_overalldict[species] = df_input['IN_' + species].tolist()
    Exp_Select_dict[species] = df_input['SEL_' + species].tolist()
    try:
        Exp_Output_flowrates_overalldict[species] = df_input['OUT_' + species].tolist() # Assume for now that user has output flowrates or calculates them himself
    except:
        continue
    
    
# Extracting the rest of the information from the input excel file
Pressure = df_input['Pressure'].tolist()
Volume_CatWeight = df_input['Volume_CatWeight'].tolist()
SpaceTime = df_input['SpaceTime'].tolist()
Temperature = df_input['Temperature'].tolist()
Exp_Conversion = df_input['Exp_Conversion'].tolist()

# Input from user asked: Reactor_type, species on which conversion is based, Known_ads, Known_values, Single_surface_step
Conv_species = 'EtOH'
Reactor_type = 'PFR'
Known_ads = {'EtOH':[1],'ACET':[1],'H2':[1],'H2O':[0]}
Known_values= {}
Single_surface_step = True

# Total number of experiments
num_exp = len(df_input.index)

#%%
"""
Tool Execution
"""

with warnings.catch_warnings():                                                                                                                        
    warnings.simplefilter("ignore")
    df_complete, start_time, total_comb = Tool.overall_loop(num_exp, Species_total, Overall_stoich, Pressure, Volume_CatWeight, SpaceTime, Temperature, Exp_Conversion, Initial_flowrates_overalldict, Exp_Select_dict, Reactor_type, Conv_species, Known_ads, Known_values, Exp_Output_flowrates_overalldict, Single_surface_step)   

print("\n")
print("Constructing and solving the mass balances for every mechanism (and parameter combinations) took ", time.time() - start_time, " seconds")
print("\n")
print("Saving dataframe to Excel...")
print("\n")

# Saving dataframe (df_complete) of all the mechanisms (and parameter combinations) to Excel 

start_save_massbalance = time.time()

Output_path = Tool_path / 'Output'
if not os.path.exists(Output_path):
    os.mkdir(Output_path)

name_output = 'Output.xlsx'

with pd.ExcelWriter(Output_path / name_output) as writer:
    df_complete.to_excel(writer, sheet_name = 'complete') 

print('Saving solved mass balance dataframe to Excel took ', time.time() - start_save_massbalance, ' seconds') 

#%%
"""
Feature Extraction (+ ranking)
"""
warnings.filterwarnings('ignore')

Next_ftex = True
while Next_ftex:
    
    # Ask user what the independent and the dependent variables will be for the feature extraction.
    print('What are the independent and dependent variables considered for the feature extraction (use the column names of the input file)?')
    
    bad_input_1 = True
        
    while bad_input_1:
        
        indep_var = input('Independent variable: ')
        
        if indep_var in col_names:
            
            bad_input_1 = False
            
        else:
            
            print('Please insert correct column name of the independent variable')
    
    bad_input_2 = True
        
    while bad_input_2:
        
        dep_var = input('Dependent variable: ')
        
        if dep_var in col_names:
            
            bad_input_2 = False
            
        else:
            
            print('Please insert correct column name of the dependent variable')
    
    #indep_var = 'Pressure'
    #dep_var = 'Conversion'
    indep_list = df_input[indep_var]
    dep_list = df_input[dep_var]
    
    # Ask user which datapoints need to be considered, in other words, what are the values of the experimental conditions considered constant
    # To be implemented
    
    start_ftex = time.time()

    df_ftex, Exp_features = Tool.ftex(df_complete, num_exp, total_comb, indep_var, dep_var, indep_list, dep_list, Pressure, Volume_CatWeight, SpaceTime, Temperature, Exp_Conversion, Exp_Select_dict)
    
    print('\n')
    print("Performing the feature extraction for every mechanism (and parameter combinations) took ", time.time() - start_ftex, " seconds")
    print('\n')
    
    start_ftex_save = time.time()
    
    # Saving df_ftex dataframe to Excel
    if len(dep_var) <= 7 and dep_var[:4] == 'SEL_':
        dep_var_short = dep_var
    elif dep_var[:4] == 'SEL_':
        dep_var_short = dep_var[:9]
    else:
        dep_var_short = dep_var[:3] 
    
    with pd.ExcelWriter(Output_path / name_output, engine='openpyxl', mode='a') as writer:
        df_ftex.to_excel(writer, sheet_name = 'ftex|%sVS%s' %(dep_var_short, indep_var[:3]))
    
    print('Saving feature extraction dataframe to Excel took ', time.time() - start_ftex_save, ' seconds')
    
    #%%%
    """
    Ranking
    """

    Next_rank = True
    while Next_rank:
        
        input_rank_false = True
        while input_rank_false:
            
            try:
                # Ask user for tolerance (0 < tol < 1) value for to determine if theoretical positions of transitions between features differ too much from the feature transition in the experimental data
                tol = float(input('Please input a tolerance value (between 0 and 1) to determine if theoretical positions of transitions between features differ too much from the feature transitions in the experimental data: '))
                input_rank_false = False
                
            except:
                
                print('Please insert correct input (number)')
        
        start_rank = time.time()
        
        df_rank = Tool.rank(df_ftex, Exp_features, tol)
        
        print('\n')
        print("Performing the ranking took ", time.time() - start_rank, " seconds")
        print('\n')
        
        start_rank_save = time.time()
        
        # Saving df_rank dataframe of all the mechanisms (and parameter combinations) to Excel 
        with pd.ExcelWriter(Output_path / name_output, engine='openpyxl', mode='a') as writer:
            df_rank.to_excel(writer, sheet_name = 'rank|tol=%.2f|%sVS%s' %(tol, dep_var_short, indep_var[:3]))
        
        print('Saving ranked dataframe to Excel took ', time.time() - start_rank_save, ' seconds')
        
        ans_rank_false = True
        while ans_rank_false:
            ans_rank = input('''Currently %d mechanisms are ranked. Increasing the tolerance could increase this number, decreasing the tolerance could decrease this number. 
Do you want to rank the mechanisms using a different tolerance value? (y/n) '''%(len(df_rank)))
            
            if ans_rank in ['n', 'N', 'no', 'NO']:
                
                Next_rank = False
                ans_rank_false = False
            
            elif ans_rank in ['y', 'Y', 'yes', 'YES']:
                
                ans_rank_false = False
            
            else:
                
                print('Please insert correct input')
    
    # The following part asks the user if he wants to perform another feature extraction based on a different independent and dependent variable
    ans_ftex_false = True
    while ans_ftex_false:
        
        ans_ftex = input('Do you want to rank the mechanisms based on feature extraction with a different combination of independent and dependent variables? (y/n) ')
    
        if ans_ftex in ['n', 'N', 'no', 'NO']:
            
            Next_ftex = False
            ans_ftex_false = False
            
        elif ans_ftex in ['y', 'Y', 'yes', 'YES']:
            
            ans_ftex_false = False    
            
        else:
                
            print('Please insert correct input')

#%%
"""
Final
"""

print('\n')
print('Tool finished')