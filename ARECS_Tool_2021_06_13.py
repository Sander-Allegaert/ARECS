# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 17:48:52 2021

@author: Sander Allegaert
"""

"""
Imported modules
"""
import time
import copy
import math
import itertools
import numpy as np
import sympy as sym
import pandas as pd
import FeatureExtraction_2021_06_13 as ft
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import warnings
from sys import exit
from sympy.interactive import printing
sym.init_session()
plt.ioff()

#%%
def rate_construct(Species_total, Reaction_number, Stoich, Ads_dict, RDS, Single_surface_step=True, Reversible=True): 
    """
    This function focusses on transforming a mechanism into the assiocated rate equation.

    Creation of rate equation r associated with a specific reaction. The general shape of the rate equation is given below,
    it consists of a kinetic, potential and adsorption term. 

        (Kinetic term)*(Potential term)
    r = -------------------------------
              (Adsorption term)
        
    ----------
    PARAMETERS
    ----------
    Species_total: List 
        List of species present in the reaction mixture.
        
    Reaction_number: Integer
        Represents which global reaction is considered.

    Stoich: Dictionary
        Dictionary which contains the stoichiometric numbers of the reactants and products of the specific reaction.
        Species that do not participate in the specific reaction are not included.
        Convention used: stoichiometric number of reactants negative, of products positive.
        
    Ads_dict: Dictionary
        Dictionary which contains the adsoroption behaviour of all the species.
        Convention: 0 = species does not adsorb
                    1 = species adsorbs molecularly
                    2 = species adsorbs dissociatively

    RDS: String
        The string represents the rate determining elementary step of the specific reaction.
        Convention: SRi = surface reaction i is RDS
                    ADS_i = adsorption of reactant i is RDS
                    DES_i = desorption of product i is RDS
                    UNCAT = uncatalyzed reaction (1-step gas phase reaction)
        
    Single_surface_step: Boolian
        True: one surface reaction assumed (default)
        False: multiple surface reactions assumed
        
    Reversible: Boolian
        True: reversible reaction (default)
        False: irreversible reaction (only used for uncatalyzed reactions, adsorption/desorption and surface reactions are always considered reversible)
        
    -------
    RETURNS
    -------
    Reaction_rate_symbolic: Symbolic expression
        Symbolic expression of the reaction rate (sympy module used for symbolic expression)
        
    Eq_ads_dict: Dictionary
        Dictionary which contains the equilibrium adsorption constants of every species.
        Constants expressed in symbol format.
        
    Eq_sr : Symbol
        Symbol which represents the equilibrium constant of the surface reaction.
    
    Eq_glob: Symbol
        Symbol which represents the global equilibrium constant of the specific reaction.
    
    Eq_glob_expr: Symbolic expression
        Symbolic expression of the global equilibrium constant in function of the adsorption equibilibrium 
        constants and the surface reaction equilibrium constant.
    
    Rate_RDS: Symbol
        Rate coefficient of the RDS.
        Convention: 'k_+' is forward rate coefficient of RDS, 'k_-' is reverse rate coefficient of RDS.
        Coefficient expressed in symbol format.
    
    Partial_pressure_dict: Dictionary
        Dictionary which contains the partial pressures of the corresponding species.
        Partial pressures expressed in symbol format.
    """

# Defining Eq_ads_dict in a symbolic way    
    Eq_ads_dict = {}
    
    for species in Species_total:
        
        Eq_ads_dict[species] = sym.symbols('K_' + str(species))
    
# Defining Eq_r
    Eq_sr = sym.symbols('K_sr%d'%Reaction_number)
        
# Defining Eq_glob
    Eq_glob = sym.symbols('K_glob%d'%Reaction_number)
  
# Defining Rate_RDS
    Rate_RDS = sym.symbols('k_-%d'%Reaction_number) if 'DES' in RDS else sym.symbols('k_+%d'%Reaction_number)
    
# Defining active site density
    Site_density = sym.symbols('L')
        
# Defining Partial_pressure_dict
    Partial_pressure_dict = {}
    
    for species in Species_total:
        
        Partial_pressure_dict[species] = sym.symbols('p_' + str(species))

# Actual start construction of the rate equation        
    if Single_surface_step == False:
        
        print("Multiple surface reactions are not yet included in the tool.")
        exit()
    
    else:
        
        Ads_term = 1
        Kin_term = 1
        Pot_term = 0
        Eq_glob_expr = 1
        
# Uncatalyzed reaction
        if RDS == 'UNCAT':
                
            Reaction_rate_symbolic_reac = 1
            Reaction_rate_symbolic_prod = 1
            
            for species in Stoich.keys():
                    
                if Stoich[species] < 0:
                        
                    Reaction_rate_symbolic_reac *= Partial_pressure_dict[species]**(-Stoich[species])
                
                elif Stoich[species] > 0 and Reversible == True:
                    
                    Reaction_rate_symbolic_prod *= Partial_pressure_dict[species]**(Stoich[species])
                
                else:
                    
                    Reaction_rate_symbolic_prod = 0
                
            Reaction_rate_symbolic = Rate_RDS * (Reaction_rate_symbolic_reac - Reaction_rate_symbolic_prod/Eq_glob)
            
            Eq_glob_expr *= Eq_sr
            
            return Reaction_rate_symbolic, Eq_ads_dict, Eq_sr, Eq_glob, Eq_glob_expr, Rate_RDS, Site_density, Partial_pressure_dict

# Surface reaction is RDS   
        elif RDS == 'SR1':
            
            #Determining Adsorption term        
    
            for species in Species_total:
                
                if Ads_dict[species] == 0:
                    
                    continue
                
                elif Ads_dict[species] == 1:
                    
                    Ads_term += Eq_ads_dict[species]*Partial_pressure_dict[species]
                    
                elif Ads_dict[species] == 2: 
                    
                    Ads_term += sym.sqrt(Eq_ads_dict[species]*Partial_pressure_dict[species])
            
            # Determining Kinetic and Potential term + exponent of Adsorption term
            
            stoich_reac_mol = 0
            stoich_reac_diss = 0
            stoich_prod_mol = 0
            stoich_prod_diss = 0
            
            diss_species = False
            
            Pot_term_reac = 1
            Pot_term_prod = 1
            
            for species in Stoich.keys():
                
                if Ads_dict[species] == 1:
                    
                    if Stoich[species] < 0:
                        
                        stoich_reac_mol += - Stoich[species]
                        
                        Pot_term_reac *= Partial_pressure_dict[species]
                        
                        Kin_term *= Eq_ads_dict[species]
                    
                    else:
                        
                        stoich_prod_mol += Stoich[species]
                        
                        Pot_term_prod *= Partial_pressure_dict[species]
                        
                elif Ads_dict[species] == 2:
                    
                    diss_species = True
                    
                    if Stoich[species] < 0:
                        
                        stoich_reac_diss += - Stoich[species]
                        
                        Pot_term_reac *= sym.sqrt(Partial_pressure_dict[species])
                        
                        Kin_term *= sym.sqrt(Eq_ads_dict[species])
                    
                    else:
                        
                        stoich_prod_diss += Stoich[species]
                        
                        Pot_term_prod *= sym.sqrt(Partial_pressure_dict[species])
                
                elif Ads_dict[species] == 0:
                    
                    if Stoich[species] < 0:
                        
                        Pot_term_reac *= Partial_pressure_dict[species]
                        
                    else:
                        
                        Pot_term_prod *= Partial_pressure_dict[species]
                        
                else:   
                    
                    continue
            
            if diss_species == True:
                
                stoich_reac_mol /= 2
                stoich_prod_mol /= 2
                
                Pot_term_prod /= sym.sqrt(Eq_glob)
                
                Kin_term *= 0.5
                
            else:
                
                Pot_term_prod /= Eq_glob
            
            n = round(max(stoich_reac_mol + stoich_reac_diss, stoich_prod_mol + stoich_prod_diss))
            Ads_term **= n
            
            Pot_term = Pot_term_reac - Pot_term_prod
            
            Kin_term *= Rate_RDS * Site_density
            
            # Constructing Eq_glob_expr
            if diss_species == True:
                
                Eq_glob_expr *= Eq_sr**2
                
                for species in Stoich.keys():
                        
                    if Ads_dict[species] == 1:
                        
                        Eq_glob_expr *= Eq_ads_dict[species]**(-2*np.sign(Stoich[species]))
                    
                    elif Ads_dict[species] == 2:
                        
                        Eq_glob_expr *= Eq_ads_dict[species]**(-np.sign(Stoich[species]))
                    
                    else:
                        
                        continue
            else:
                
                Eq_glob_expr *= Eq_sr
                
                for species in Stoich.keys():
                        
                    if Ads_dict[species] == 0:
                        
                        continue
                        
                    else:
                        
                        Eq_glob_expr *= Eq_ads_dict[species]**(-np.sign(Stoich[species]))             
                
# Adsorption of a reactant is RDS      
        elif 'ADS' in RDS:
            
            RDS_species = RDS.split("_")[1]
            
            diss_species = False
            diss_species_RDS = False
            
            Pot_term_reac = Partial_pressure_dict[RDS_species]
            
            n = 1 # Exponent of the adsorption term
            
            for species in Species_total:
                
                if species == RDS_species:
                    
                    if Ads_dict[species] == 2:
                        
                        diss_species_RDS = True
                        diss_species = True
                        n = 2
                        
                    else:
                        
                        continue
                    
                elif Ads_dict[species] == 0:
                    
                    continue
                
                elif Ads_dict[species] == 1:
                    
                    Ads_term += Eq_ads_dict[species]*Partial_pressure_dict[species]
                    
                elif Ads_dict[species] == 2: 
                    
                    Ads_term += sym.sqrt(Eq_ads_dict[species]*Partial_pressure_dict[species])
                    
                    if species in Stoich.keys():
                        
                        diss_species = True
                
            if diss_species == True:
                    
                term = sym.sqrt(Eq_glob)**(-1)
                
                if diss_species_RDS == True:
                    
                    term *= sym.sqrt(Eq_ads_dict[RDS_species])
                    
                else:
                    
                    term *= Eq_ads_dict[RDS_species]
                
                for species in Stoich.keys():
                    
                    if species == RDS_species:
                        
                        continue
                    
                    elif Ads_dict[species] == 2:
                    
                        term *= sym.sqrt(Partial_pressure_dict[species])**np.sign(Stoich[species])
                    
                    else:
                        
                        term *= Partial_pressure_dict[species]**np.sign(Stoich[species])
                        
            else:
                
                term = Eq_ads_dict[RDS_species]/Eq_glob
                
                for species in Stoich.keys():
                    
                    if species == RDS_species:
                        
                        continue
                    
                    else:
                    
                        term *= Partial_pressure_dict[species]**np.sign(Stoich[species])
                        
            if diss_species_RDS == True:
                
                Pot_term_prod = 1/Eq_glob
                
                for species in Stoich.keys():
                    
                    if species == RDS_species:
                        
                        continue
                    
                    elif Ads_dict[species] == 2:
                        
                        Pot_term_prod *= Partial_pressure_dict[species]**np.sign(Stoich[species])
                    
                    else:
                        
                        Pot_term_prod *= Partial_pressure_dict[species]**(2*np.sign(Stoich[species]))
                        
            else:
                    
                    Pot_term_prod = 1
                    
                    for species in Stoich.keys():
                    
                        if species == RDS_species:
                        
                            continue
                    
                        elif Ads_dict[species] == 2:
                        
                            Pot_term_prod *= sym.sqrt(Partial_pressure_dict[species])**np.sign(Stoich[species])
                    
                        else:
                        
                            Pot_term_prod *= Partial_pressure_dict[species]**(np.sign(Stoich[species]))
                    
                    if diss_species == True:
                        
                        Kin_term *= 0.5
                        
                        Pot_term_prod /= sym.sqrt(Eq_glob)
                        
                    else:
                        
                        Pot_term_prod /= Eq_glob
                
            
            Ads_term += term
            Ads_term **= n
            
            Pot_term = Pot_term_reac - Pot_term_prod
            
            Kin_term *= Rate_RDS * Site_density
            
            # Constructing Eq_glob_expr
            if diss_species == True:
                                  
                Eq_glob_expr *= Eq_sr**2
                
                for species in Stoich.keys():
                        
                    if Ads_dict[species] == 1:
                        
                        Eq_glob_expr *= Eq_ads_dict[species]**(-2*np.sign(Stoich[species]))
                    
                    elif Ads_dict[species] == 2:
                        
                        Eq_glob_expr *= Eq_ads_dict[species]**(-np.sign(Stoich[species]))
                    
                    else:
                        
                        continue
            else:
                
                Eq_glob_expr *= Eq_sr
                
                for species in Stoich.keys():
                        
                    if Ads_dict[species] == 0:
                        
                        continue
                        
                    else:
                        
                        Eq_glob_expr *= Eq_ads_dict[species]**(-np.sign(Stoich[species])) 

# Desorption of a product is RDS            
        elif 'DES' in RDS:
            
            RDS_species = RDS.split("_")[1]
            
            diss_species = False
            diss_species_RDS = False
            
            Pot_term_prod = Partial_pressure_dict[RDS_species]
            
            n = 1 # Exponent of the adsorption term
            
            for species in Species_total:
                
                if species == RDS_species:
                    
                    if Ads_dict[species] == 2:
                        
                        diss_species_RDS = True
                        diss_species = True
                        n = 2
                        
                    else:
                        
                        continue
                    
                elif Ads_dict[species] == 0:
                    
                    continue
                
                elif Ads_dict[species] == 1:
                    
                    Ads_term += Eq_ads_dict[species]*Partial_pressure_dict[species]
                    
                elif Ads_dict[species] == 2: 
                    
                    Ads_term += sym.sqrt(Eq_ads_dict[species]*Partial_pressure_dict[species])
                    
                    if species in Stoich.keys():
                        
                        diss_species = True
                
            if diss_species == True:
                    
                term = sym.sqrt(Eq_glob)
                
                if diss_species_RDS == True:
                    
                    term *= sym.sqrt(Eq_ads_dict[RDS_species])
                    
                else:
                    
                    term *= Eq_ads_dict[RDS_species]
                
                for species in Stoich.keys():
                    
                    if species == RDS_species:
                        
                        continue
                    
                    elif Ads_dict[species] == 2:
                    
                        term *= sym.sqrt(Partial_pressure_dict[species])**(-np.sign(Stoich[species]))
                    
                    else:
                        
                        term *= Partial_pressure_dict[species]**(-np.sign(Stoich[species]))
                        
            else:
                
                term = Eq_ads_dict[RDS_species]*Eq_glob
                
                for species in Stoich.keys():
                    
                    if species == RDS_species:
                        
                        continue
                    
                    else:

                        term *= Partial_pressure_dict[species]**(-np.sign(Stoich[species]))
            
            if diss_species_RDS == True:
                
                Pot_term_reac = Eq_glob
                
                for species in Stoich.keys():
                    
                    if species == RDS_species:
                        
                        continue
                    
                    elif Ads_dict[species] == 2:
                        
                        Pot_term_reac *= Partial_pressure_dict[species]**(-np.sign(Stoich[species]))
                    
                    else:
                        
                        Pot_term_reac *= Partial_pressure_dict[species]**(-2*np.sign(Stoich[species]))
                        
            else:
                    
                    Pot_term_reac = 1
                    
                    for species in Stoich.keys():
                    
                        if species == RDS_species:
                        
                            continue
                    
                        elif Ads_dict[species] == 2:
                        
                            Pot_term_reac *= sym.sqrt(Partial_pressure_dict[species])**(-np.sign(Stoich[species]))
                    
                        else:
                        
                            Pot_term_reac *= Partial_pressure_dict[species]**(-np.sign(Stoich[species]))
                    
                    if diss_species == True:
                        
                        Kin_term *= 0.5
                        
                        Pot_term_reac *= sym.sqrt(Eq_glob)
                        
                    else:
                        
                        Pot_term_reac *= Eq_glob
                
            
            Ads_term += term
            Ads_term **= n
            
            Pot_term = Pot_term_reac - Pot_term_prod
            
            Kin_term *= Rate_RDS * Site_density 

            # Constructing Eq_glob_expr
            if diss_species == True:
            
                Eq_glob_expr *= Eq_sr**2
                
                for species in Stoich.keys():
                        
                    if Ads_dict[species] == 1:
                        
                        Eq_glob_expr *= Eq_ads_dict[species]**(-2*np.sign(Stoich[species]))
                    
                    elif Ads_dict[species] == 2:
                        
                        Eq_glob_expr *= Eq_ads_dict[species]**(-np.sign(Stoich[species]))
                    
                    else:
                        
                        continue
            else:
                
                Eq_glob_expr *= Eq_sr
                
                for species in Stoich.keys():
                        
                        Eq_glob_expr *= Eq_ads_dict[species]**(-np.sign(Stoich[species]))                 

# Wrong input for RDS    
        else:
            
            print("Incorrect convention used for RDS")
            exit()
            
        Reaction_rate_symbolic = Kin_term*Pot_term/Ads_term
    
    return Reaction_rate_symbolic, Eq_ads_dict, Eq_sr, Eq_glob, Eq_glob_expr, Rate_RDS, Site_density, Partial_pressure_dict

#%%
def net_production(Reaction_rate_symbolic_dict, Net_production_stoich):
    """
    This function constructs the net production rates for all species.
    
    ----------
    PARAMETERS
    ----------
    Reaction_rate_symbolic_dict: Dictionary
        Dictionary containing the symbolic representation of every reaction in the reaction mixture.
        E.g.: {1:r1,2:r2} == r1 is the reaction rate of reaction 1, r2 is the reaction rate of reaction 2.
    
    Net_production_stoich: Dictionary
        The keys are the species present in the reaction mixture.
        The corresponding values are subdictionaries which represent the stoichiometric numbers 
        of all the species in the reaction mixture for every reaction.
        E.g.: {'A':{1:-1,2:0}, 'B':{1:1,2:-1}, 'C':{1:0,2:2}, 'I':{1:0,2:0}} == A <--reaction_1--> B <--reaction_2--> 2 C and I is inert
            
    -------
    RETURNS
    -------
    Net_production_dict: Dictionary
        Dictionary which contains the symbolic net production rates of all species.
    """
    
    Net_production_dict = {}
    
    for species in Net_production_stoich:
        
        Net_production_dict[species] = 0
        
        for reaction in Net_production_stoich[species]:
            
            Net_production_dict[species] += Net_production_stoich[species][reaction] * Reaction_rate_symbolic_dict[reaction]
    
    
    return Net_production_dict

#%%
def substitute(Net_production_dict, Eq_ads_symbol, Eq_ads_values, Eq_sr_symbol, Eq_sr_values, Eq_glob_symbol, Eq_glob_expr_dict, Rate_RDS_symbol, Rate_RDS_values, Site_density_symbol, Site_density_value, p_t, Partial_pressure_dict, Lump=False, Flowrates_values={}):
    """
    This function substitutes the equilibrium constants, rate coefficients and active site density,
    which are symbolically represented in the net production rates, with numeric values. The function
    also subsitutes the partial pressures (symbol) of the species with their respective flowrates (symbol).
    Furthermore, the global equilbrium constant is first substituted by its symbolic expression and subsequently
    substituted by numerical values.
    
    ----------
    PARAMETERS
    ----------
    Net_production_dict: Dictionary
        Dictionary which contains the symbolic net production rates of all species.
    
    Eq_ads_symbol: Dictionary
        Dictionary which contains the equilibrium adsorption constants of every species.
        Constants expressed in symbol format.
    
    Eq_ads_values: Dictionary
        Dictionary which contains the values of the equilibrium adsorption constants of every species.
        Constants expressed in Float format.
    
    Eq_sr_symbol: Dictionary
        Dictionary which contains the equilibrium constants of the surface steps of every reaction in the reaction mixture.
        Constants expressed in symbol format.
    
    Eq_sr_values: Dictionary
        Dictionary which contains the values of the equilibrium constants of the surface steps of every reaction in the reaction mixture.
        Constants expressed in Float format.
    
    Eq_glob_symbol: Dictionary
        Dictionary which contains the global equilibrium constant of every reaction in the reaction mixture.
        Constants expressed in symbol format.
    
    Eq_glob_expr_dict: Dictionary
        Dictionary which contains the symbolic expression of the global equilibrium constant of every
        reaction in the reaction mixture.
        
    Rate_RDS_symbol: Dictionary
        Dictionary which contains the rate coefficient of the RDS of every reaction in the reaction mixture.
        Convention: 'k_+' is forward rate coefficient of RDS, 'k_-' is reverse rate coefficient of RDS.
        Coefficients expressed in symbol format.    
        
    Rate_RDS_values: Dictionary
        Dictionary which contains the values of the rate coefficient of the RDS of every reaction in the reaction mixture.
        Coefficients expressed in Float format.
    
    Site_density_symbol: Symbol
        Symbol which respresents the active site density.
    
    Site_density_value: Float
        Density of the active sites, expressed in concentration units.
        
    p_t: Float
        Total pressure inside the reactor (constant).
    
    Partial_pressure_dict: Dictionary
        Dictionary which contains the partial pressures of the corresponding species.
        Partial pressures expressed in symbol format.
        
    -------
    RETURNS
    -------  
    Net_production_dict_values: Dictionary
        Dictionary which contains the net production rates of all species.
        The only parameters left are the flowrates of the species.
        
    Flowrates_dict: Dictionary
        Dictionary which contains the flowrates of all the species.
        Flowrates expressed in symbol format.  
    
    Eq_glob_values: Dictionary
        Dictionary which contains the global equilibrium constants of every reaction  in the reaction mixture.
        Equilibrium constants expressed in Float format.
    """
    
    Net_production_dict_values = copy.copy(Net_production_dict)
    Eq_glob_values = copy.copy(Eq_glob_expr_dict)
    Flowrates_dict = {}
    
    F_t = 0
    
    if Lump == True:
        
        for species in Partial_pressure_dict:
        
            F_t += Flowrates_values[species]
    
        for net_production in Net_production_dict:
            
            for reaction in Eq_glob_symbol:
                
                Net_production_dict_values[net_production] = Net_production_dict_values[net_production].subs([(Eq_glob_symbol[reaction], Eq_glob_expr_dict[reaction]),(Rate_RDS_symbol[reaction], 1)])
                Net_production_dict_values[net_production] = Net_production_dict_values[net_production].subs(Eq_sr_symbol[reaction], Eq_sr_values[reaction])
                
            for species in Eq_ads_values:
            
                Net_production_dict_values[net_production] = Net_production_dict_values[net_production].subs([(Eq_ads_symbol[species], Eq_ads_values[species]),(Partial_pressure_dict[species], Flowrates_values[species]*p_t/F_t)])    
                
            Net_production_dict_values[net_production] = Net_production_dict_values[net_production].subs(Site_density_symbol, 1)
            
        return Net_production_dict_values
                
    else:
        
        for species in Partial_pressure_dict:
            
            Flowrates_dict[species] = sym.symbols('F_'+ str(species))
        
            F_t += Flowrates_dict[species]
    
        for net_production in Net_production_dict:
            
            for reaction in Eq_glob_symbol:
                
                Net_production_dict_values[net_production] = Net_production_dict_values[net_production].subs([(Eq_glob_symbol[reaction], Eq_glob_expr_dict[reaction]),(Rate_RDS_symbol[reaction], Rate_RDS_values[reaction])])
                Net_production_dict_values[net_production] = Net_production_dict_values[net_production].subs(Eq_sr_symbol[reaction], Eq_sr_values[reaction])
            
            for species in Eq_ads_values:
            
                Net_production_dict_values[net_production] = Net_production_dict_values[net_production].subs([(Eq_ads_symbol[species], Eq_ads_values[species]),(Partial_pressure_dict[species], Flowrates_dict[species]*p_t/F_t)])    
                
            Net_production_dict_values[net_production] = Net_production_dict_values[net_production].subs(Site_density_symbol, Site_density_value)
            
        for reaction in Eq_glob_expr_dict:
            
            Eq_glob_values[reaction] = Eq_glob_values[reaction].subs(Eq_sr_symbol[reaction], Eq_sr_values[reaction])
            
            for species in Eq_ads_values:
                
                Eq_glob_values[reaction] = Eq_glob_values[reaction].subs(Eq_ads_symbol[species], Eq_ads_values[species])
        
        return Net_production_dict_values, Flowrates_dict, Eq_glob_values

#%%
def mass_balance_solver(Net_production_dict_values, Flowrates_dict, Initial_flowrates_dict, Reactor_type, Volume_CatWeight, Conv_species, Species_total, Overall_stoich, inerts, Select_stoich):
    """
    This function solves the set of mass balances and returns the conversion and 
    flowrates of all the species at the end of the reactor.
    
    ----------
    PARAMETERS
    ----------
    Net_production_dict_values: Dictionary
        Dictionary which contains the net production rates of all species.
        The only parameters left are the flowrates of the species.
        
    Flowrates_dict: Dictionary
        Dictionary which contains the flowrates of all the species.
        Flowrates expressed in symbol format.
        
    Initial_flowrates_dict: Dictionary
        Dictionary which contains the initial flowrates of all species present in the reaction mixture.
        Initial flowrates expressed in Float format.
        
    Reactor_type: String
        Two ideal, isothermal, isobaric reactor types are considered: plug flow reactor and continuous stirred-tank reactor.
        Convention: plug flow reactor -> 'PFR'
                    continuous stirred-tank reactor -> 'CSTR'
        
    Reactor_size: Float
        The reactor size can either be expressed by stating the total reactor volume (volumetric units)
        or the total amount of catalyst present in te reactor (mass units)
        
    Conv_species: String
        The conversion will be calculated based on the species represented by this string.
        
    -------
    RETURNS
    -------
    Final_flowrates: Dictionary
        Dictionary which contains the flowrates of the species at the end of the reactor.
        Flowrates expressed in Float format.
        
    Conversion: Float
        Conversion at the end of the reactor.
    """
    
    if Reactor_type == 'CSTR':
        
        print('Reactor type CSTR not yet implemented')
        exit()
    
    elif Reactor_type == 'PFR':
        
        l = []
    
        for net_production in Net_production_dict_values:
        
            f = sym.lambdify([tuple(Flowrates_dict.values())], Net_production_dict_values[net_production],'numpy')
            l.append(f)
        
        #start_solver = time.time()   
        sol = odeint(ode_construct, list(Initial_flowrates_dict.values()), [0, Volume_CatWeight], args=(l,), rtol=None, mxstep=0)
        #print(time.time()- start_solver)
        
        Final_flowrates = {species:sol[-1][Species_total.index(species)] for species in Species_total}
        Conversion = (Initial_flowrates_dict[Conv_species] - Final_flowrates[Conv_species])/Initial_flowrates_dict[Conv_species]
        Select = {species:((Final_flowrates[species] - Initial_flowrates_dict[species]) / (Initial_flowrates_dict[Conv_species] - Final_flowrates[Conv_species]) * (Select_stoich[Conv_species] / Select_stoich[species]) if species not in inerts else 'INERT') for species in Species_total}
        
        return Final_flowrates, Conversion, Select
        
    else:
        
        print('Wrong reactor type, check convention')
        exit()
    
    return None

#%%
def ode_construct(y, t, l):
    """
    This is a support function used to solve the set of differential mass balances when a PFR reactor is considered.
    ode_construct is used as parameter for the odeint function.
    
    ----------
    PARAMETERS
    ----------
    y: ~
        
    t: ~
    
    l: ~    
        
    -------
    RETURNS
    -------
    dydt: ~
    """
    
    dydt = [l[i](y) for i in range(len(l))]    
        
    return dydt   

#%%
def loop_par(Species_total, Overall_stoich, inerts, Select_stoich, Net_production_dict, Eq_ads_symbol, Eq_sr_symbol, Eq_glob_symbol, Eq_glob_expr_dict, RDS_dict, Rate_RDS_symbol, Site_density_symbol, Partial_pressure_dict, Initial_flowrates_overalldict, Exp_Outlet_flowrates_overalldict, Exp_Select_dict, Pressure, Exp_Conversion, Volume_CatWeight, Conv_species, Reactor_type, total_comb, it, Known_values={}, Time=False):
    """
    This function loops trough all the possible combinations of adsorption equilibrium constants, surface reaction
    equilibrium constants, active site density and RDS rate coefficients of the different reaction in the reaction mixture.
    The function generates a DataFrame which contains the theoretical conversion and final flowrates for every total pressure considered
    during the experiments for every possible combination of the aforementioned parameters.
    Note: a backward rate coefficient 'k_-' will be replaced by 'k_+'/'K_ADS_X'. The column 'Rate_RDS' in the DataFrame contains the values
    of the 'k_+' of every reaction. If a reaction uses 'k_-', the column 'Rate_RDS_adap' will contain the values given by 'k_+'/'K_ADS_X'.
    'k_-' is only used if the RDS is the desorption of a product, the 'K_ADS' of that product must then be used to calculate 'k_-'.
    
    ----------
    Parameters
    ----------
    Exp_features: List
    
    
    Net_production_dict: Dictionary
        Dictionary which contains the symbolic net production rates of all species.
        
    Eq_ads_symbol: Dictionary
        Dictionary which contains the equilibrium adsorption constants of every species.
        Constants expressed in symbol format.
        
    Eq_sr_symbol: Dictionary
        Dictionary which contains the equilibrium constants of the surface steps of every reaction in the reaction mixture.
        Constants expressed in symbol format.
        
    Eq_glob_symbol: Dictionary
        Dictionary which contains the global equilibrium constant of every reaction in the reaction mixture.
        Constants expressed in symbol format.
        
    Eq_glob_expr_dict: Dictionary
        Dictionary which contains the symbolic expression of the global equilibrium constant of every
        reaction in the reaction mixture.
        
    RDS_dict: Dictionary    
        
    Rate_RDS_symbol: Dictionary
        Dictionary which contains the rate coefficient of the RDS of every reaction in the reaction mixture.
        Convention: 'k_+' is forward rate coefficient of RDS, 'k_-' is reverse rate coefficient of RDS.
        Coefficients expressed in symbol format.
        
    Site_density_symbol: Symbol
        Symbol which respresents the active site density.
        
    p_t_list: List
        List which contains the total pressure values considered during the experiments.
        
    exp_y: List
        
        
    Partial_pressure_dict: Dictionary
        Dictionary which contains the partial pressures of the corresponding species.
        Partial pressures expressed in symbol format.
        
    Initial_flowrates_overalldict: Dictionary
        Dictionary which contains the initial molar flowrates of all species present in the reaction mixture for every datapoint/experiment.
        Every key corresponds to a species, the associated values are list of the initial molar flowrates of this species for every datapoint/experiment (in same order of experiments as p_t_list and exp_y)
        Initial molar flowrates expressed in Float format.
        
    Outlet_flowrates_overalldict: Dictionary
    
    Reactor_type: String
        Two ideal, isothermal, isobaric reactor types are considered: plug flow reactor and continuous stirred-tank reactor.
        Convention: plug flow reactor -> 'PFR'
                    continuous stirred-tank reactor -> 'CSTR'
        
    Reactor_size: Float
        The reactor size can either be expressed by stating the total reactor volume (volumetric units)
        or the total amount of catalyst present in te reactor (mass units)
        
    Conv_species: String
        The conversion will be calculated based on the species represented by this string.
        
    Known_values: Dictionary
        Dictionary which contains lists with known values. If e.g. the adsorption equilibrium constants of all species are known, this can be used as input.
        Doing this drastically lowers the amount of combinations the tools needs to go trough.
        Default = {}
        Convention: Known_values={'Site_density':[value],
                                  'Eq_ads_values':[[value for each species (mind the order, must correspond to order in which species were inputted)]],
                                  'Eq_sr_values':[[value for each surface step of each reaction (mind the order, must correspond to order in which reactions were inputted)]],
                                  'Rate_RDS_values':[[value for each RDS of each reaction (mind the order, must correspond to order in which reactions were inputted)]]}
    
    Time: Boolian
        True: execution time of a single iteration is determined (first parameter loop)
        False (default): pass
    
    it: Integer
    
    -------
    Returns
    -------
    df: DataFrame
        DataFrame which contains the theoretical conversion and final flowrates for every total pressure considered
        during the experiments for every possible combination of the aforementioned parameters, which are also included in the DataFrame.
        The extracted features of every combination are also included. 
    """
    
    # Lists of default parameter values
    Eq_ads_values_list = [0.1, 0.5, 1, 10, 100]
    #Eq_ads_values_list = [0.5, 5]
    Eq_sr_values_list = [0.01, 0.1, 0.5, 1, 10, 100]
    #Eq_sr_values_list = [0.5, 5]
    Rate_RDS_values_list = [1]
    Site_density_values_list = [1]
    
    header = pd.DataFrame(columns=['Site_density', 'Eq_ads', 'Eq_sr', 'Eq_glob', 'Rate_RDS', 'Rate_RDS_adap', 'Lumped_rate_coefficient', 'Final_flowrates', 'Conversion', 'Selectivities'])
    num_species = len(Eq_ads_symbol)    
    num_reactions = len(Eq_glob_symbol)
    num_exp = len(Pressure)
    
    # If the uncatalyzed system is considered, no iteration over possible adsorption values is needed
    Uncat = True
    for RDS in list(RDS_dict.values()):
        if RDS != 'UNCAT':
            Uncat = False
        else:
            continue
        
    var_ads_values = itertools.product(Eq_ads_values_list, repeat=num_species)
    var_sr_values = itertools.product(Eq_sr_values_list, repeat=num_reactions)
    var_rate_values = itertools.product(Rate_RDS_values_list, repeat=num_reactions)
    
    counter = 0
    
    # If Site_density and Rate_RDS_values are given as input by the user, a lumped rate coefficient will NOT be used
    if 'Site_density' in Known_values and 'Rate_RDS_values' in Known_values:
                
        Site_density_values_list = Known_values.get('Site_density', Site_density_values_list)
        var_ads_values = Known_values.get('Eq_ads_values', var_ads_values) if Uncat == False else itertools.product([1], repeat=num_species)
        var_sr_values = Known_values.get('Eq_sr_values', var_sr_values)
        var_rate_values = Known_values.get('Rate_RDS_values', var_rate_values)
        
        rows = []
        for density in Site_density_values_list:
            Site_density_value = density
            
            var_ads_values = itertools.product(Eq_ads_values_list, repeat=num_species)
            var_ads_values = Known_values.get('Eq_ads_values', var_ads_values) if Uncat == False else itertools.product([1], repeat=num_species)
            for val_ads in var_ads_values:
                Eq_ads_values = {list(Eq_ads_symbol.keys())[i]:val_ads[i] for i in range(num_species)}
                
                var_sr_values = itertools.product(Eq_sr_values_list, repeat=num_reactions)
                var_sr_values = Known_values.get('Eq_sr_values', var_sr_values)
                for val_sr in var_sr_values:
                    Eq_sr_values = {list(Eq_glob_symbol.keys())[i]:val_sr[i] for i in range(num_reactions)}
                    
                    var_rate_values = itertools.product(Rate_RDS_values_list, repeat=num_reactions)
                    var_rate_values = Known_values.get('Rate_RDS_values', var_rate_values)
                    for val_rate in var_rate_values:
                        Rate_RDS_values_orig = {(list(Rate_RDS_symbol.keys())[i]):(val_rate[i]) for i in range(num_reactions)}
                        
                        # Rate_RDS_values_adap is calculated to correct for the fact that when desorption is the RDS, the rate coefficient of the backward RDS is used: k_- = k_+ / K_ADS_RDS 
                        Rate_RDS_values_adap = {}
                        for j in list(RDS_dict.keys()):
                            
                            if RDS_dict[j][0:3] == 'DES':
                                
                                Rate_RDS_values_adap[j] = val_rate[j-1]/Eq_ads_values[RDS_dict[j][4:]] 
                            
                            else:
                                                                                                                                           
                                Rate_RDS_values_adap[j] = val_rate[j-1]
                    
                        conv_list = [] 
                        final_flowrates_list = []
                        select_list = []
                        
                        for index in range(num_exp):
                            
                            # This part is used to calculate the execution time of a single iteration
                            if Time == True and counter == 0: 
                            
                                start_single_iteration = time.time()
                                Initial_flowrates_dict = {list(Initial_flowrates_overalldict.keys())[i]:Initial_flowrates_overalldict[list(Initial_flowrates_overalldict.keys())[i]][index] for i in range(num_species)}
                                Net_production_dict_values, Flowrates_dict, Eq_glob_values = substitute(Net_production_dict, Eq_ads_symbol, Eq_ads_values, Eq_sr_symbol, Eq_sr_values, Eq_glob_symbol, Eq_glob_expr_dict, Rate_RDS_symbol, Rate_RDS_values_adap, Site_density_symbol, Site_density_value, Pressure[index], Partial_pressure_dict)                             
                                Final_flowrates, Conversion, Select = mass_balance_solver(Net_production_dict_values, Flowrates_dict, Initial_flowrates_dict, Reactor_type, Volume_CatWeight[index], Conv_species, Species_total, Overall_stoich, inerts, Select_stoich)
                                conv_list.append(Conversion)
                                final_flowrates_list.append(Final_flowrates)
                                select_list.append(Select)
                                iteration_time = time.time() - start_single_iteration
                                counter += 1
                                return iteration_time
                            
                            Initial_flowrates_dict = {list(Initial_flowrates_overalldict.keys())[i]:Initial_flowrates_overalldict[list(Initial_flowrates_overalldict.keys())[i]][index] for i in range(num_species)}
                            Net_production_dict_values, Flowrates_dict, Eq_glob_values = substitute(Net_production_dict, Eq_ads_symbol, Eq_ads_values, Eq_sr_symbol, Eq_sr_values, Eq_glob_symbol, Eq_glob_expr_dict, Rate_RDS_symbol, Rate_RDS_values_adap, Site_density_symbol, Site_density_value, Pressure[index], Partial_pressure_dict)                             
                            Final_flowrates, Conversion, Select = mass_balance_solver(Net_production_dict_values, Flowrates_dict, Initial_flowrates_dict, Reactor_type, Volume_CatWeight[index], Conv_species, Species_total, Overall_stoich, inerts, Select_stoich)
                            
                            conv_list.append(Conversion)
                            final_flowrates_list.append(Final_flowrates)
                            select_list.append(Select)
                            
                            it += 1
                            
                            if it % 1000 == 0:
                                print('%d combinations of %d executed, %.1f %%'%(it, total_comb, it/total_comb*100))
                        
                        rows.append({'Site_density':copy.copy(density), 'Eq_ads':copy.copy(Eq_ads_values), 'Eq_sr':copy.copy(Eq_sr_values), 'Eq_glob':copy.copy(Eq_glob_values), 
                                     'Rate_RDS':copy.copy(Rate_RDS_values_orig), 'Rate_RDS_adap':copy.copy(Rate_RDS_values_adap), 'Lumped_rate_coefficient':'N/A', 
                                     'Final_flowrates':copy.copy(final_flowrates_list), 'Conversion':copy.copy(conv_list), 'Selectivities':copy.copy(select_list)})
                          
        df_par = header.append(rows, ignore_index=False)
    
    # If 'Rate_RDS_values' are given as input by the user, but 'Site_density' not, it will be assumed the user has given a lumped rate coefficient which will be further used      
    elif 'Rate_RDS_values' in Known_values:
         
         var_ads_values = Known_values.get('Eq_ads_values', var_ads_values) if Uncat == False else itertools.product([1], repeat=num_species)
         var_sr_values = Known_values.get('Eq_sr_values', var_sr_values)
         var_rate_values = Known_values.get('Rate_RDS_values', var_rate_values)
         
         rows = []
         Site_density_value = 1
             
         var_ads_values = itertools.product(Eq_ads_values_list, repeat=num_species)
         var_ads_values = Known_values.get('Eq_ads_values', var_ads_values) if Uncat == False else itertools.product([1], repeat=num_species)
         for val_ads in var_ads_values:
            Eq_ads_values = {list(Eq_ads_symbol.keys())[i]:val_ads[i] for i in range(num_species)}
            
            var_sr_values = itertools.product(Eq_sr_values_list, repeat=num_reactions)
            var_sr_values = Known_values.get('Eq_sr_values', var_sr_values)
            for val_sr in var_sr_values:
                Eq_sr_values = {list(Eq_glob_symbol.keys())[i]:val_sr[i] for i in range(num_reactions)}
                
                var_rate_values = itertools.product(Rate_RDS_values_list, repeat=num_reactions)
                var_rate_values = Known_values.get('Rate_RDS_values', var_rate_values)
                for val_rate in var_rate_values:
                    Rate_RDS_values_orig = {(list(Rate_RDS_symbol.keys())[i]):(val_rate[i]) for i in range(num_reactions)}
                    
                    # Rate_RDS_values_adap is calculated to correct for the fact that when desorption is the RDS, the rate coefficient of the backward RDS is used: k_- = k_+ / K_ADS_RDS 
                    Rate_RDS_values_adap = {}
                    for j in list(RDS_dict.keys()):
                        
                        if RDS_dict[j][0:3] == 'DES':
                            
                            Rate_RDS_values_adap[j] = val_rate[j-1]/Eq_ads_values[RDS_dict[j][4:]] 
                        
                        else:
                                                                                                                                       
                            Rate_RDS_values_adap[j] = val_rate[j-1]
                
                    conv_list = [] 
                    final_flowrates_list = [] 
                    select_list = []
                    
                    for index in range(num_exp):
                        
                        # This part is used to calculate the execution time of a single iteration
                        if Time == True and counter == 0: 
                        
                            start_single_iteration = time.time()
                            Initial_flowrates_dict = {list(Initial_flowrates_overalldict.keys())[i]:Initial_flowrates_overalldict[list(Initial_flowrates_overalldict.keys())[i]][index] for i in range(num_species)}
                            Net_production_dict_values, Flowrates_dict, Eq_glob_values = substitute(Net_production_dict, Eq_ads_symbol, Eq_ads_values, Eq_sr_symbol, Eq_sr_values, Eq_glob_symbol, Eq_glob_expr_dict, Rate_RDS_symbol, Rate_RDS_values_adap, Site_density_symbol, Site_density_value, Pressure[index], Partial_pressure_dict)                             
                            Final_flowrates, Conversion, Select = mass_balance_solver(Net_production_dict_values, Flowrates_dict, Initial_flowrates_dict, Reactor_type, Volume_CatWeight[index], Conv_species, Species_total, Overall_stoich, inerts, Select_stoich)
                            conv_list.append(Conversion)
                            final_flowrates_list.append(Final_flowrates)
                            select_list.append(Select)
                            iteration_time = time.time() - start_single_iteration
                            counter += 1
                            return iteration_time
                        
                        Initial_flowrates_dict = {list(Initial_flowrates_overalldict.keys())[i]:Initial_flowrates_overalldict[list(Initial_flowrates_overalldict.keys())[i]][index] for i in range(num_species)}
                        Net_production_dict_values, Flowrates_dict, Eq_glob_values = substitute(Net_production_dict, Eq_ads_symbol, Eq_ads_values, Eq_sr_symbol, Eq_sr_values, Eq_glob_symbol, Eq_glob_expr_dict, Rate_RDS_symbol, Rate_RDS_values_adap, Site_density_symbol, Site_density_value, Pressure[index], Partial_pressure_dict)                             
                        Final_flowrates, Conversion, Select = mass_balance_solver(Net_production_dict_values, Flowrates_dict, Initial_flowrates_dict, Reactor_type, Volume_CatWeight[index], Conv_species, Species_total, Overall_stoich, inerts, Select_stoich)
                        conv_list.append(Conversion)
                        final_flowrates_list.append(Final_flowrates)
                        
                        it += 1
                    
                        if it % 1000 == 0:
                            print('%d combinations of %d executed, %.1f %%'%(it, total_comb, it/total_comb*100))
                            
                    rows.append({'Site_density':'Lumped with Rate_RDS', 'Eq_ads':copy.copy(Eq_ads_values), 'Eq_sr':copy.copy(Eq_sr_values), 'Eq_glob':copy.copy(Eq_glob_values), 
                                 'Rate_RDS':copy.copy(Rate_RDS_values_orig), 'Rate_RDS_adap':copy.copy(Rate_RDS_values_adap), 'Lumped_rate_coefficient':'N/A', 
                                 'Final_flowrates':copy.copy(final_flowrates_list), 'Conversion':copy.copy(conv_list), 'Selectivities':copy.copy(select_list)})
                        
         df_par = header.append(rows, ignore_index=False)
    
    # If 'Rate_RDS_values' is not given as input by the user, the tool will calculate a LUMPED rate coefficient
    else:
     
         var_ads_values = Known_values.get('Eq_ads_values', var_ads_values) if Uncat == False else itertools.product([1], repeat=num_species)
         var_sr_values = Known_values.get('Eq_sr_values', var_sr_values)
         
         rows = []
         Site_density_value = 1
             
         var_ads_values = itertools.product(Eq_ads_values_list, repeat=num_species)
         var_ads_values = Known_values.get('Eq_ads_values', var_ads_values) if Uncat == False else itertools.product([1], repeat=num_species)
         for val_ads in var_ads_values:
            Eq_ads_values = {list(Eq_ads_symbol.keys())[i]:val_ads[i] for i in range(num_species)}
            
            var_sr_values = itertools.product(Eq_sr_values_list, repeat=num_reactions)
            var_sr_values = Known_values.get('Eq_sr_values', var_sr_values)
            for val_sr in var_sr_values:
                Eq_sr_values = {list(Eq_glob_symbol.keys())[i]:val_sr[i] for i in range(num_reactions)}
                
                #Lumped_k_dict = get_lumped_k()
                Lumped_k_dict = get_lumped_k(Pressure, Exp_Conversion, Initial_flowrates_overalldict, Exp_Outlet_flowrates_overalldict, Exp_Select_dict, Volume_CatWeight, num_reactions, num_species, num_exp, Conv_species, Net_production_dict, Eq_ads_symbol, Eq_ads_values, Eq_sr_symbol, Eq_sr_values, Eq_glob_symbol, Eq_glob_expr_dict, Rate_RDS_symbol, Site_density_symbol, Partial_pressure_dict)
                
                conv_list = [] 
                final_flowrates_list = []                     
                select_list = []
                    
                for index in range(num_exp):

                    # This part is used to calculate the execution time of a single iteration
                    if Time == True and counter == 0: 
                    
                        start_single_iteration = time.time()
                        Initial_flowrates_dict = {list(Initial_flowrates_overalldict.keys())[i]:Initial_flowrates_overalldict[list(Initial_flowrates_overalldict.keys())[i]][index] for i in range(num_species)}
                        Net_production_dict_values, Flowrates_dict, Eq_glob_values = substitute(Net_production_dict, Eq_ads_symbol, Eq_ads_values, Eq_sr_symbol, Eq_sr_values, Eq_glob_symbol, Eq_glob_expr_dict, Rate_RDS_symbol, Lumped_k_dict, Site_density_symbol, Site_density_value, Pressure[index], Partial_pressure_dict)                             
                        Final_flowrates, Conversion, Select = mass_balance_solver(Net_production_dict_values, Flowrates_dict, Initial_flowrates_dict, Reactor_type, Volume_CatWeight[index], Conv_species, Species_total, Overall_stoich, inerts, Select_stoich)
                        conv_list.append(Conversion)
                        final_flowrates_list.append(Final_flowrates)
                        select_list.append(Select)
                        iteration_time = time.time() - start_single_iteration
                        counter += 1
                        return iteration_time
                    
                    Initial_flowrates_dict = {list(Initial_flowrates_overalldict.keys())[i]:Initial_flowrates_overalldict[list(Initial_flowrates_overalldict.keys())[i]][index] for i in range(num_species)}
                    Net_production_dict_values, Flowrates_dict, Eq_glob_values = substitute(Net_production_dict, Eq_ads_symbol, Eq_ads_values, Eq_sr_symbol, Eq_sr_values, Eq_glob_symbol, Eq_glob_expr_dict, Rate_RDS_symbol, Lumped_k_dict, Site_density_symbol, Site_density_value, Pressure[index], Partial_pressure_dict)                             
                    Final_flowrates, Conversion, Select = mass_balance_solver(Net_production_dict_values, Flowrates_dict, Initial_flowrates_dict, Reactor_type, Volume_CatWeight[index], Conv_species, Species_total, Overall_stoich, inerts, Select_stoich)
                    conv_list.append(Conversion)
                    final_flowrates_list.append(Final_flowrates)
                    select_list.append(Select)
                    
                    it += 1
                    
                    if it % 1000 == 0:
                        print('%d combinations of %d executed, %.1f %%'%(it, total_comb, it/total_comb*100))
                        
                rows.append({'Site_density':'N/A', 'Eq_ads':copy.copy(Eq_ads_values), 'Eq_sr':copy.copy(Eq_sr_values), 'Eq_glob':copy.copy(Eq_glob_values), 
                             'Rate_RDS':'N/A', 'Rate_RDS_adap':'N/A', 'Lumped_rate_coefficient':copy.copy(Lumped_k_dict), 
                             'Final_flowrates':copy.copy(final_flowrates_list), 'Conversion':copy.copy(conv_list), 'Selectivities':copy.copy(select_list)})
              
         df_par = header.append(rows, ignore_index=False)
    
    it_new = it
    return df_par, it_new

#%%
def loop_mech(Species_total, Overall_stoich, Known_ads, Single_surface_step=True):

    num_reactions = len(Overall_stoich[Species_total[0]])    

    if Single_surface_step == False:
        
        print("Multiple surface reactions are not yet included in the tool.")
        exit()
    
    else: 
        
        # User wants to test all the possible combinations
        if Known_ads == 'ALLPOS':
        
            num_species = len(Species_total) 
            num_reactions = len(list(Overall_stoich.values())[0])
           
            rowsx = []
    
            inerts = []
        
            active_species = copy.copy(Species_total)
    
            for species in Species_total:
        
                if list(Overall_stoich[species].values()) == [0 for i in range(num_reactions)]:
            
                    inerts.append(species)
                    active_species.remove(species)
                    
            no_2 = True
        
            num_mol = 0 # Number of species that adsorb molecularly (or do not adsorb), used if no_2 == False
            mol_species = [] # Species that adsorb molecularly (or do not adsorb), used if no_2 == False
            diss_species = [] # Species that adsorb dissociatively (or do not adsorb), used if no_2 == False
            for species in active_species:
         
                if (2 in Overall_stoich[species].values()) or (-2 in Overall_stoich[species].values()):
            
                    no_2 = False
                    num_mol += 1
                    mol_species.append(species)
            
                else:
                
                    diss_species.append(species)
                
                    continue
        
            Ads_dict = {}
        
            headerx = pd.DataFrame(columns=['Net_production_dict', 'Ads_dict', 'RDS_dict', 'Reaction_rate_dict', 'Reversible_dict', 
                                            'Eq_ads_symbol_dict', 'Eq_sr_symbol_dict', 'Eq_glob_symbol_dict', 'Eq_glob_expr_dict',
                                            'Rate_RDS_dict', 'Site_density_symbol', 'Partial_pressure_dict']) 
            rowsx = []
        
            # All species (excluding inerts) have stoichiometric number of [-1, 0, 1] in all reactions
            if no_2 == True:
            
                # Every species taking part in at least one reaction adsorbs molecularly or does not adsorb: [0,1]
                # Inerts can adsorb both molecularly and dissociatively or not at all: [0,1,2]
                var_ads_type = itertools.product([0,1], repeat=(num_species-len(inerts)))
                for ads_type in var_ads_type:
                    Ads_dict = {active_species[i]:ads_type[i] for i in range(num_species-len(inerts))}
                
                    var_ads_type_inerts = itertools.product([0,1,2], repeat=(len(inerts)))
                    for ads_type_inerts in var_ads_type_inerts:
                        Ads_dict_inerts = {inerts[i]:ads_type_inerts[i] for i in range(len(inerts))}
                        
                        Ads_dict.update(Ads_dict_inerts)
                        Ads_dict_s = {species:Ads_dict[species] for species in Species_total}

                        header_reactions = pd.DataFrame(columns=['Reaction', 'Ads_dict', 'RDS', 'Rate_construct_output', 'Reversible'])
                        rows = []
                        for reaction in range(1,num_reactions+1):
                        
                            Stoich_o = {species:Overall_stoich[species][reaction] for species in Species_total}
                            Stoich = copy.copy(Stoich_o)
                        
                            for species in Stoich_o:
                            
                                if Stoich_o[species] == 0:
                                
                                    del Stoich[species]                       
                        
                            Uncat = True
                            for species in Stoich:
                                if Ads_dict_s[species] != 0:
                                    Uncat = False
                        
                            if Uncat == True:
                            
                                # Uncatalyzed reaction assumed 
                                RDS = 'UNCAT'
                                Rate_construct_output = rate_construct(Species_total, reaction, Stoich, Ads_dict_s, RDS) #(reversible)
                                rows.append({'Reaction':copy.copy(reaction), 'Ads_dict':copy.copy(Ads_dict_s), 'RDS':copy.copy(RDS),'Rate_construct_output':copy.copy(Rate_construct_output), 'Reversible': 'YES'})

                                #Rate_construct_output = rate_construct(Species_total, reaction, Stoich, Ads_dict_s, RDS, Single_surface_step=True, Reversible=False) #irreversible
                                #rows.append({'Reaction':copy.copy(reaction), 'Ads_dict':copy.copy(Ads_dict_s), 'RDS':copy.copy(RDS),'Rate_construct_output':copy.copy(Rate_construct_output), 'Reversible': 'NO'})
                            
                            elif Uncat == False:
                        
                                # Surface reaction RDS
                                RDS = 'SR1'
                                Rate_construct_output = rate_construct(Species_total, reaction, Stoich, Ads_dict_s, RDS)
                                rows.append({'Reaction':copy.copy(reaction), 'Ads_dict':copy.copy(Ads_dict_s), 'RDS':copy.copy(RDS),'Rate_construct_output':copy.copy(Rate_construct_output), 'Reversible': 'YES'})
                                
                                # If there are multiple reactions, only the surface reaction can be rate determining, so adsorption and desorption are skipped when this is the case!
                                if num_reactions == 1: 
                                
                                    # Adsorption of reactants or desorption of products RDS
                                    for species in Stoich:
                                
                                        if Ads_dict_s[species] == 0:
                                    
                                            continue
                                
                                        elif Stoich[species] < 0: # Adsorption reactants
                                    
                                            RDS = 'ADS_' + species
                                            Rate_construct_output = rate_construct(Species_total, reaction, Stoich, Ads_dict_s, RDS)
                                            rows.append({'Reaction':copy.copy(reaction), 'Ads_dict':copy.copy(Ads_dict_s), 'RDS':copy.copy(RDS),'Rate_construct_output':copy.copy(Rate_construct_output), 'Reversible': 'YES'})
                                    
                                        elif Stoich[species] > 0: # Desorption products
                                    
                                            RDS = 'DES_' + species
                                            Rate_construct_output = rate_construct(Species_total, reaction, Stoich, Ads_dict_s, RDS)
                                            rows.append({'Reaction':copy.copy(reaction), 'Ads_dict':copy.copy(Ads_dict_s), 'RDS':copy.copy(RDS),'Rate_construct_output':copy.copy(Rate_construct_output), 'Reversible': 'YES'})
                                
                                    
                        df_reactions = header_reactions.append(rows, ignore_index=False)

                        reac_list = []
                        for i in range(1,num_reactions+1):
                            index_list = df_reactions[df_reactions['Reaction'] == i].index.tolist()
                            reac_list.append(index_list)

                        var_reac = itertools.product(*reac_list, repeat=1)
                        
                        for comb in var_reac:

                            reaction_rate_symbolic_dict = {i+1:df_reactions.loc[comb[i],'Rate_construct_output'][0] for i in range(num_reactions)}
                            RDS_dict = {i+1:df_reactions.loc[comb[i],'RDS'] for i in range(num_reactions)}
                            reversible_dict = {i+1:df_reactions.loc[comb[i],'Reversible'] for i in range(num_reactions)}
                            Ads_dict = df_reactions.loc[comb[0], 'Ads_dict']
                            Eq_ads_dict = df_reactions.loc[comb[0], 'Rate_construct_output'][1]
                            Eq_sr_dict = {i+1:df_reactions.loc[comb[i],'Rate_construct_output'][2] for i in range(num_reactions)}
                            Eq_glob_dict = {i+1:df_reactions.loc[comb[i],'Rate_construct_output'][3] for i in range(num_reactions)}
                            Eq_glob_expr_dict = {i+1:df_reactions.loc[comb[i],'Rate_construct_output'][4] for i in range(num_reactions)}
                            Rate_RDS_dict = {i+1:df_reactions.loc[comb[i],'Rate_construct_output'][5] for i in range(num_reactions)}
                            Site_density = df_reactions.loc[comb[0], 'Rate_construct_output'][6]
                            Partial_pressure_dict = df_reactions.loc[comb[0], 'Rate_construct_output'][7]
                        
                            Net_production_dict = net_production(reaction_rate_symbolic_dict, Overall_stoich)                
                        
                            rowsx.append({'Net_production_dict':copy.copy(Net_production_dict), 'Ads_dict':copy.copy(Ads_dict), 'RDS_dict':copy.copy(RDS_dict), 'Reaction_rate_dict':copy.copy(reaction_rate_symbolic_dict), 'Reversible_dict':copy.copy(reversible_dict), 
                                          'Eq_ads_symbol_dict':copy.copy(Eq_ads_dict), 'Eq_sr_symbol_dict':copy.copy(Eq_sr_dict), 'Eq_glob_symbol_dict':copy.copy(Eq_glob_dict), 'Eq_glob_expr_dict':copy.copy(Eq_glob_expr_dict),
                                          'Rate_RDS_dict':copy.copy(Rate_RDS_dict), 'Site_density_symbol':copy.copy(Site_density), 'Partial_pressure_dict':copy.copy(Partial_pressure_dict)})        
                            
                # Every species taking part in at least one reaction adsorbs dissociatively: [2]
                # Inerts can adsorb both molecularly and dissociatively or not at all: [0,1,2]
            
                var_ads_type = itertools.product([2], repeat=(num_species-len(inerts)))
                for ads_type in var_ads_type:
                    Ads_dict = {active_species[i]:ads_type[i] for i in range(num_species-len(inerts))}
                
                    var_ads_type_inerts = itertools.product([0,1,2], repeat=(len(inerts)))
                    for ads_type_inerts in var_ads_type_inerts:
                        Ads_dict_inerts = {inerts[i]:ads_type_inerts[i] for i in range(len(inerts))}
                    
                        Ads_dict.update(Ads_dict_inerts)
                        Ads_dict_s = {species:Ads_dict[species] for species in Species_total}

                        header_reactions = pd.DataFrame(columns=['Reaction', 'Ads_dict', 'RDS', 'Rate_construct_output', 'Reversible'])
                        rows = []
                        for reaction in range(1,num_reactions+1):
                        
                            Stoich_o = {species:Overall_stoich[species][reaction] for species in Species_total}
                            Stoich = copy.copy(Stoich_o)
                        
                            for species in Stoich_o:
                            
                                if Stoich_o[species] == 0:
                                    
                                    del Stoich[species]                       
                                              
                            # Surface reaction RDS
                            RDS = 'SR1'
                            Rate_construct_output = rate_construct(Species_total, reaction, Stoich, Ads_dict_s, RDS)
                            rows.append({'Reaction':copy.copy(reaction), 'Ads_dict':copy.copy(Ads_dict_s), 'RDS':copy.copy(RDS),'Rate_construct_output':copy.copy(Rate_construct_output), 'Reversible': 'YES'})
                            
                            # If there are multiple reactions, only the surface reaction can be rate determining, so adsorption and desorption are skipped when this is the case!
                            if num_reactions == 1:
                            
                                # Adsorption of reactants or desorption of products RDS
                                for species in Stoich:
                                
                                    if Ads_dict_s[species] == 0:
                                    
                                        continue
                                
                                    elif Stoich[species] < 0: # Adsorption reactants
                                    
                                        RDS = 'ADS_' + species
                                        Rate_construct_output = rate_construct(Species_total, reaction, Stoich, Ads_dict_s, RDS)
                                        rows.append({'Reaction':copy.copy(reaction), 'Ads_dict':copy.copy(Ads_dict_s), 'RDS':copy.copy(RDS),'Rate_construct_output':copy.copy(Rate_construct_output), 'Reversible': 'YES'})
                                    
                                    elif Stoich[species] > 0: # Desorption products
                                    
                                        RDS = 'DES_' + species
                                        Rate_construct_output = rate_construct(Species_total, reaction, Stoich, Ads_dict_s, RDS)
                                        rows.append({'Reaction':copy.copy(reaction), 'Ads_dict':copy.copy(Ads_dict_s), 'RDS':copy.copy(RDS),'Rate_construct_output':copy.copy(Rate_construct_output), 'Reversible': 'YES'})
                                    
                        df_reactions = header_reactions.append(rows, ignore_index=False)

                        reac_list = []
                        for i in range(1,num_reactions+1):
                            index_list = df_reactions[df_reactions['Reaction'] == i].index.tolist()
                            reac_list.append(index_list)

                        var_reac = itertools.product(*reac_list, repeat=1)
                        
                        for comb in var_reac:

                            reaction_rate_symbolic_dict = {i+1:df_reactions.loc[comb[i],'Rate_construct_output'][0] for i in range(num_reactions)}
                            RDS_dict = {i+1:df_reactions.loc[comb[i],'RDS'] for i in range(num_reactions)}
                            reversible_dict = {i+1:df_reactions.loc[comb[i],'Reversible'] for i in range(num_reactions)}
                            Ads_dict = df_reactions.loc[comb[0], 'Ads_dict']
                            Eq_ads_dict = df_reactions.loc[comb[0], 'Rate_construct_output'][1]
                            Eq_sr_dict = {i+1:df_reactions.loc[comb[i],'Rate_construct_output'][2] for i in range(num_reactions)}
                            Eq_glob_dict = {i+1:df_reactions.loc[comb[i],'Rate_construct_output'][3] for i in range(num_reactions)}
                            Eq_glob_expr_dict = {i+1:df_reactions.loc[comb[i],'Rate_construct_output'][4] for i in range(num_reactions)}
                            Rate_RDS_dict = {i+1:df_reactions.loc[comb[i],'Rate_construct_output'][5] for i in range(num_reactions)}
                            Site_density = df_reactions.loc[comb[0], 'Rate_construct_output'][6]
                            Partial_pressure_dict = df_reactions.loc[comb[0], 'Rate_construct_output'][7]
                        
                            Net_production_dict = net_production(reaction_rate_symbolic_dict, Overall_stoich)                
                        
                            rowsx.append({'Net_production_dict':copy.copy(Net_production_dict), 'Ads_dict':copy.copy(Ads_dict), 'RDS_dict':copy.copy(RDS_dict), 'Reaction_rate_dict':copy.copy(reaction_rate_symbolic_dict), 'Reversible_dict':copy.copy(reversible_dict), 
                                          'Eq_ads_symbol_dict':copy.copy(Eq_ads_dict), 'Eq_sr_symbol_dict':copy.copy(Eq_sr_dict), 'Eq_glob_symbol_dict':copy.copy(Eq_glob_dict), 'Eq_glob_expr_dict':copy.copy(Eq_glob_expr_dict),
                                          'Rate_RDS_dict':copy.copy(Rate_RDS_dict), 'Site_density_symbol':copy.copy(Site_density), 'Partial_pressure_dict':copy.copy(Partial_pressure_dict)}) 
        
            # At least one species (excluding inerts) has a stoichiometric number of [-2, 2] in at least 1 reaction
            # This means that species with stoichiometric number [-2,2] in a certain reactions must adsorb molecularly (or do not adsorb)
            # and species with stoichiometric number [-1,1] in certain reactions must adsorb dissociatively
            # ASSUMING ONLY ONE TYPE OF SURFACE REACTION
            if no_2 == False:
                
                Ads_dict = {diss_species[i]:2 for i in range(len(diss_species))}
                var_ads_type = itertools.product([0,1], repeat=(num_mol))
                for ads_type in var_ads_type:
                    Ads_dict_mol = {mol_species[i]:ads_type[i] for i in range(num_mol)}
                    
                    var_ads_type_inerts = itertools.product([0,1,2], repeat=(len(inerts)))
                    for ads_type_inerts in var_ads_type_inerts:
                        Ads_dict_inerts = {inerts[i]:ads_type_inerts[i] for i in range(len(inerts))}
                        
                        Ads_dict.update(Ads_dict_mol)
                        Ads_dict.update(Ads_dict_inerts)
                        Ads_dict_s = {species:Ads_dict[species] for species in Species_total}
                
                        header_reactions = pd.DataFrame(columns=['Reaction', 'Ads_dict', 'RDS', 'Rate_construct_output', 'Reversible'])
                        rows = []
                        for reaction in range(1,num_reactions+1):
                            
                            Stoich_o = {species:Overall_stoich[species][reaction] for species in Species_total}
                            Stoich = copy.copy(Stoich_o)
                            
                            for species in Stoich_o:
                                
                                if Stoich_o[species] == 0:
                                    
                                    del Stoich[species]                       
                                                  
                            RDS = 'UNCAT'
                            Rate_construct_output = rate_construct(Species_total, reaction, Stoich, Ads_dict_s, RDS) #(reversible)
                            rows.append({'Reaction':copy.copy(reaction), 'Ads_dict':copy.copy(Ads_dict_s), 'RDS':copy.copy(RDS),'Rate_construct_output':copy.copy(Rate_construct_output), 'Reversible': 'YES'})
    
                            #Rate_construct_output = rate_construct(Species_total, reaction, Stoich, Ads_dict_s, RDS, Single_surface_step=True, Reversible=False) #irreversible
                            #rows.append({'Reaction':copy.copy(reaction), 'Ads_dict':copy.copy(Ads_dict_s), 'RDS':copy.copy(RDS),'Rate_construct_output':copy.copy(Rate_construct_output), 'Reversible': 'NO'})
    
                            # Surface reaction RDS
                            RDS = 'SR1'
                            Rate_construct_output = rate_construct(Species_total, reaction, Stoich, Ads_dict_s, RDS)
                            rows.append({'Reaction':copy.copy(reaction), 'Ads_dict':copy.copy(Ads_dict_s), 'RDS':copy.copy(RDS),'Rate_construct_output':copy.copy(Rate_construct_output), 'Reversible': 'YES'})
                            
                            # If there are multiple reactions, only the surface reaction can be rate determining, so adsorption and desorption are skipped when this is the case!
                            if num_reactions == 1:
                            
                                # Adsorption of reactants or desorption of products RDS
                                for species in Stoich:
                                    
                                    if Ads_dict_s[species] == 0:
                                        
                                        continue
                                    
                                    elif Stoich[species] < 0: # Adsorption reactants
                                        
                                        RDS = 'ADS_' + species
                                        Rate_construct_output = rate_construct(Species_total, reaction, Stoich, Ads_dict_s, RDS)
                                        rows.append({'Reaction':copy.copy(reaction), 'Ads_dict':copy.copy(Ads_dict_s), 'RDS':copy.copy(RDS),'Rate_construct_output':copy.copy(Rate_construct_output), 'Reversible': 'YES'})
                                        
                                    elif Stoich[species] > 0: # Desorption products
                                        
                                        RDS = 'DES_' + species
                                        Rate_construct_output = rate_construct(Species_total, reaction, Stoich, Ads_dict_s, RDS)
                                        rows.append({'Reaction':copy.copy(reaction), 'Ads_dict':copy.copy(Ads_dict_s), 'RDS':copy.copy(RDS),'Rate_construct_output':copy.copy(Rate_construct_output), 'Reversible': 'YES'})
                                        
                        df_reactions = header_reactions.append(rows, ignore_index=False)
    
                        reac_list = []
                        for i in range(1,num_reactions+1):
                            index_list = df_reactions[df_reactions['Reaction'] == i].index.tolist()
                            reac_list.append(index_list)
    
                        var_reac = itertools.product(*reac_list, repeat=1)
                            
                        for comb in var_reac:
    
                            reaction_rate_symbolic_dict = {i+1:df_reactions.loc[comb[i],'Rate_construct_output'][0] for i in range(num_reactions)}
                            RDS_dict = {i+1:df_reactions.loc[comb[i],'RDS'] for i in range(num_reactions)}
                            reversible_dict = {i+1:df_reactions.loc[comb[i],'Reversible'] for i in range(num_reactions)}
                            Ads_dict = df_reactions.loc[comb[0], 'Ads_dict']
                            Eq_ads_dict = df_reactions.loc[comb[0], 'Rate_construct_output'][1]
                            Eq_sr_dict = {i+1:df_reactions.loc[comb[i],'Rate_construct_output'][2] for i in range(num_reactions)}
                            Eq_glob_dict = {i+1:df_reactions.loc[comb[i],'Rate_construct_output'][3] for i in range(num_reactions)}
                            Eq_glob_expr_dict = {i+1:df_reactions.loc[comb[i],'Rate_construct_output'][4] for i in range(num_reactions)}
                            Rate_RDS_dict = {i+1:df_reactions.loc[comb[i],'Rate_construct_output'][5] for i in range(num_reactions)}
                            Site_density = df_reactions.loc[comb[0], 'Rate_construct_output'][6]
                            Partial_pressure_dict = df_reactions.loc[comb[0], 'Rate_construct_output'][7]
                            
                            Net_production_dict = net_production(reaction_rate_symbolic_dict, Overall_stoich)                
                            
                            rowsx.append({'Net_production_dict':copy.copy(Net_production_dict), 'Ads_dict':copy.copy(Ads_dict), 'RDS_dict':copy.copy(RDS_dict), 'Reaction_rate_dict':copy.copy(reaction_rate_symbolic_dict), 'Reversible_dict':copy.copy(reversible_dict), 
                                          'Eq_ads_symbol_dict':copy.copy(Eq_ads_dict), 'Eq_sr_symbol_dict':copy.copy(Eq_sr_dict), 'Eq_glob_symbol_dict':copy.copy(Eq_glob_dict), 'Eq_glob_expr_dict':copy.copy(Eq_glob_expr_dict),
                                          'Rate_RDS_dict':copy.copy(Rate_RDS_dict), 'Site_density_symbol':copy.copy(Site_density), 'Partial_pressure_dict':copy.copy(Partial_pressure_dict)})
                
            
            dfx = headerx.append(rowsx, ignore_index=False)                    
            dfx['Reaction_rate_dict_str'] = dfx['Reaction_rate_dict'].astype(str)
            dfx.drop_duplicates('Reaction_rate_dict_str', inplace=True)
            df_mech = dfx.reset_index(drop=True)
            
        # User gives as input the adsorption types of the species, or default molecular adsorption is assumed (also dissociative adsorption if the stoichiometry defines it)
        else:
            
            num_species = len(Species_total) 
            num_reactions = len(list(Overall_stoich.values())[0])
           
            rowsx = []
    
            inerts = []
        
            active_species = copy.copy(Species_total)
            
            for species in Species_total:
        
                if list(Overall_stoich[species].values()) == [0 for i in range(num_reactions)]:
            
                    inerts.append(species)
                    active_species.remove(species)
            
            no_2 = True
        
            num_mol = 0 # Number of species that adsorb molecularly (or do not adsorb), used if no_2 == False
            mol_species = [] # Species that adsorb molecularly (or do not adsorb), used if no_2 == False
            diss_species = [] # Species that adsorb dissociatively (or do not adsorb), used if no_2 == False
            for species in active_species:
         
                if (2 in Overall_stoich[species].values()) or (-2 in Overall_stoich[species].values()):
            
                    no_2 = False
                    num_mol += 1
                    mol_species.append(species)
            
                else:
                
                    diss_species.append(species)
                
                    continue
            
            Ads_dict_overall = {}
            
            for species in Species_total: 
                
                if no_2 == False and species in diss_species:
                    
                    Ads_dict_overall[species] = Known_ads.get(species, [2]) # Default value of 2 is used for type of adsorption if species must adsorb dissociatively based on stoichiometry, assuming there is only one type of surface reaction
                
                else:
                    
                    Ads_dict_overall[species] = Known_ads.get(species, [1]) # Default value of 1 is used for type of adsorption
    
                
            headerx = pd.DataFrame(columns=['Net_production_dict', 'Ads_dict', 'RDS_dict', 'Reaction_rate_dict', 'Reversible_dict', 
                                      'Eq_ads_symbol_dict', 'Eq_sr_symbol_dict', 'Eq_glob_symbol_dict', 'Eq_glob_expr_dict',
                                      'Rate_RDS_dict', 'Site_density_symbol', 'Partial_pressure_dict']) 
            rowsx = []
            
            Ads_dict_s = {}
            
            list_ads_comb = list(itertools.product(*list(Ads_dict_overall.values())))
            list_ads_comb.append((0,)*num_species) # Uncatalyzed reaction is also considered
            for ads_comb in list_ads_comb:
                
                Ads_dict_s = {species:ads_comb[i] for i, species in enumerate(Species_total)}

                header_reactions = pd.DataFrame(columns=['Reaction', 'Ads_dict', 'RDS', 'Rate_construct_output', 'Reversible'])
                rows = []
                for reaction in range(1,num_reactions+1):
                
                    Stoich_o = {species:Overall_stoich[species][reaction] for species in Species_total}
                    Stoich = copy.copy(Stoich_o)
                
                    for species in Stoich_o:
                    
                        if Stoich_o[species] == 0:
                        
                            del Stoich[species]                       
                
                    Uncat = True
                    for species in Stoich:
                        if Ads_dict_s[species] != 0:
                            Uncat = False
                
                    if Uncat == True:
                    
                        # Uncatalyzed reaction assumed 
                        RDS = 'UNCAT'
                        Rate_construct_output = rate_construct(Species_total, reaction, Stoich, Ads_dict_s, RDS) #(reversible)
                        rows.append({'Reaction':copy.copy(reaction), 'Ads_dict':copy.copy(Ads_dict_s), 'RDS':copy.copy(RDS),'Rate_construct_output':copy.copy(Rate_construct_output), 'Reversible': 'YES'})

                        #Rate_construct_output = rate_construct(Species_total, reaction, Stoich, Ads_dict_s, RDS, Single_surface_step=True, Reversible=False) #irreversible
                        #rows.append({'Reaction':copy.copy(reaction), 'Ads_dict':copy.copy(Ads_dict_s), 'RDS':copy.copy(RDS),'Rate_construct_output':copy.copy(Rate_construct_output), 'Reversible': 'NO'})
                    
                    elif Uncat == False:
                
                        # Surface reaction RDS
                        RDS = 'SR1'
                        Rate_construct_output = rate_construct(Species_total, reaction, Stoich, Ads_dict_s, RDS)
                        rows.append({'Reaction':copy.copy(reaction), 'Ads_dict':copy.copy(Ads_dict_s), 'RDS':copy.copy(RDS),'Rate_construct_output':copy.copy(Rate_construct_output), 'Reversible': 'YES'})
                        
                        # If there are multiple reactions, only the surface reaction can be rate determining, so adsorption and desorption are skipped when this is the case!
                        if num_reactions == 1:
                        
                            # Adsorption of reactants or desorption of products RDS
                            for species in Stoich:
                        
                                if Ads_dict_s[species] == 0:
                            
                                    continue
                        
                                elif Stoich[species] < 0: # Adsorption reactants
                            
                                    RDS = 'ADS_' + species
                                    Rate_construct_output = rate_construct(Species_total, reaction, Stoich, Ads_dict_s, RDS)
                                    rows.append({'Reaction':copy.copy(reaction), 'Ads_dict':copy.copy(Ads_dict_s), 'RDS':copy.copy(RDS),'Rate_construct_output':copy.copy(Rate_construct_output), 'Reversible': 'YES'})
                            
                                elif Stoich[species] > 0: # Desorption products
                            
                                    RDS = 'DES_' + species
                                    Rate_construct_output = rate_construct(Species_total, reaction, Stoich, Ads_dict_s, RDS)
                                    rows.append({'Reaction':copy.copy(reaction), 'Ads_dict':copy.copy(Ads_dict_s), 'RDS':copy.copy(RDS),'Rate_construct_output':copy.copy(Rate_construct_output), 'Reversible': 'YES'})
                            
                df_reactions = header_reactions.append(rows, ignore_index=False)

                reac_list = []
                for i in range(1,num_reactions+1):
                    index_list = df_reactions[df_reactions['Reaction'] == i].index.tolist()
                    reac_list.append(index_list)

                var_reac = itertools.product(*reac_list, repeat=1)
                
                for comb in var_reac:

                    reaction_rate_symbolic_dict = {i+1:df_reactions.loc[comb[i],'Rate_construct_output'][0] for i in range(num_reactions)}
                    RDS_dict = {i+1:df_reactions.loc[comb[i],'RDS'] for i in range(num_reactions)}
                    reversible_dict = {i+1:df_reactions.loc[comb[i],'Reversible'] for i in range(num_reactions)}
                    Ads_dict = df_reactions.loc[comb[0], 'Ads_dict']
                    Eq_ads_dict = df_reactions.loc[comb[0], 'Rate_construct_output'][1]
                    Eq_sr_dict = {i+1:df_reactions.loc[comb[i],'Rate_construct_output'][2] for i in range(num_reactions)}
                    Eq_glob_dict = {i+1:df_reactions.loc[comb[i],'Rate_construct_output'][3] for i in range(num_reactions)}
                    Eq_glob_expr_dict = {i+1:df_reactions.loc[comb[i],'Rate_construct_output'][4] for i in range(num_reactions)}
                    Rate_RDS_dict = {i+1:df_reactions.loc[comb[i],'Rate_construct_output'][5] for i in range(num_reactions)}
                    Site_density = df_reactions.loc[comb[0], 'Rate_construct_output'][6]
                    Partial_pressure_dict = df_reactions.loc[comb[0], 'Rate_construct_output'][7]
                
                    Net_production_dict = net_production(reaction_rate_symbolic_dict, Overall_stoich)                
                
                    rowsx.append({'Net_production_dict':copy.copy(Net_production_dict), 'Ads_dict':copy.copy(Ads_dict), 'RDS_dict':copy.copy(RDS_dict), 'Reaction_rate_dict':copy.copy(reaction_rate_symbolic_dict), 'Reversible_dict':copy.copy(reversible_dict), 
                                  'Eq_ads_symbol_dict':copy.copy(Eq_ads_dict), 'Eq_sr_symbol_dict':copy.copy(Eq_sr_dict), 'Eq_glob_symbol_dict':copy.copy(Eq_glob_dict), 'Eq_glob_expr_dict':copy.copy(Eq_glob_expr_dict),
                                  'Rate_RDS_dict':copy.copy(Rate_RDS_dict), 'Site_density_symbol':copy.copy(Site_density), 'Partial_pressure_dict':copy.copy(Partial_pressure_dict)})        
            
            dfx = headerx.append(rowsx, ignore_index=False)                    
            dfx['Reaction_rate_dict_str'] = dfx['Reaction_rate_dict'].astype(str)
            dfx.drop_duplicates('Reaction_rate_dict_str', inplace=True)
            df_mech = dfx.reset_index(drop=True)
    
    return df_mech, inerts

#%%
def overall_loop(num_exp, Species_total, Overall_stoich, Pressure, Volume_CatWeight, SpaceTime, Temperature, Exp_Conversion, Initial_flowrates_overalldict, Exp_Select_dict, Reactor_type, Conv_species, Known_ads={}, Known_values={}, Exp_Outlet_flowrates_overalldict={}, Single_surface_step=True):          
    warnings.warn("runtime", RuntimeWarning)
    
    if Single_surface_step == True:
        df_mech, inerts = loop_mech(Species_total, Overall_stoich, Known_ads, Single_surface_step=True)
    else:
        df_mech, inerts = loop_mech(Species_total, Overall_stoich, Known_ads, Single_surface_step=False)
    
    Eq_ads_values_list = [0.1, 0.5, 1, 10, 100]
    #Eq_ads_values_list = [0.5, 5]
    Eq_sr_values_list = [0.01, 0.1, 0.5, 1, 10, 100]
    #Eq_sr_values_list = [0.5, 5]
    Rate_RDS_values_list = [1]
    Site_density_values_list = [1]
    
    num_species = len(df_mech['Ads_dict'][0])    
    num_reactions = len(df_mech['RDS_dict'][0]) 
    
    var_ads_values = itertools.product(Eq_ads_values_list, repeat=num_species)
    var_sr_values = itertools.product(Eq_sr_values_list, repeat=num_reactions)
    var_rate_values = itertools.product(Rate_RDS_values_list, repeat=num_reactions)
    
    if Known_values != {}:
        
        Site_density_values_list = Known_values.get('Site_density', Site_density_values_list)
        var_ads_values = Known_values.get('Eq_ads_values', var_ads_values)
        var_sr_values = Known_values.get('Eq_sr_values', var_sr_values)
        var_rate_values = Known_values.get('Rate_RDS_values', var_rate_values)
    
    total_mech = len(df_mech.index) - 1 # Exclude the fully uncatalyzed system 
    total_paracomb = len(Site_density_values_list)*len(list(copy.copy(var_ads_values)))*len(list(copy.copy(var_sr_values)))*len(list(copy.copy(var_rate_values)))
    total_paracomb_uncat = len(list(copy.copy(var_sr_values)))*len(list(copy.copy(var_rate_values)))
    total_comb = total_mech*total_paracomb*num_exp + total_paracomb_uncat*num_exp
    
    Select_stoich = {}
    if num_reactions > 1:
        for species in Species_total:
            
            if species == Conv_species:
                k = list(Overall_stoich[species].values())
                l = [i for i in k if i < 0]
                if all(i == l[0] for i in l):
                    Select_stoich[species] = abs(l[0])
                else:
                    print('Could not determine how to calculate the selectivity based on this stoichiometry')
                    Select_stoich = input('Give the stoichiometry needed to determine the selectivities (in dictionary format: {A:-1,B:1,C:2}): ')
            
            elif species in inerts:
                Select_stoich[species] = 0
            
            else:
                k = list(Overall_stoich[species].values())
                l = [i for i in k if i > 0]
                if all(i == l[0] for i in l):
                    Select_stoich[species] = abs(l[0])
                else:
                    print('Could not determine how to calculate the selectivity based on this stoichiometry')
                    Select_stoich = input('Give the stoichiometry needed to determine the selectivities (in dictionary format: {A:-1,B:1,C:2}): ')
                    
    else: # If there is only one reaction, calculating the selectivities is not important. However, it must be included otherwise error later one, so the values in the library are all set to 1
        Select_stoich = {species:1 for species in Species_total}
        
    # Determine execution time of one iteration of solving mass balances (average of first parameter loop of first and last mechanism in df_mech)   
    iteration_time_1 = loop_par(Species_total, Overall_stoich, inerts, Select_stoich, df_mech['Net_production_dict'].iloc[0], df_mech['Eq_ads_symbol_dict'].iloc[0], df_mech['Eq_sr_symbol_dict'].iloc[0], df_mech['Eq_glob_symbol_dict'].iloc[0],
                                df_mech['Eq_glob_expr_dict'].iloc[0], df_mech['RDS_dict'][0], df_mech['Rate_RDS_dict'].iloc[0], df_mech['Site_density_symbol'].iloc[0], 
                                df_mech['Partial_pressure_dict'].iloc[0], Initial_flowrates_overalldict, Exp_Outlet_flowrates_overalldict, Exp_Select_dict, Pressure, Exp_Conversion, 
                                Volume_CatWeight, Conv_species, Reactor_type, total_comb, 0, Known_values, Time=True)
    
    iteration_time_2 = loop_par(Species_total, Overall_stoich, inerts, Select_stoich, df_mech['Net_production_dict'].iloc[-1], df_mech['Eq_ads_symbol_dict'].iloc[-1], df_mech['Eq_sr_symbol_dict'].iloc[-1], df_mech['Eq_glob_symbol_dict'].iloc[-1],
                                df_mech['Eq_glob_expr_dict'].iloc[-1], df_mech['RDS_dict'].iloc[-1], df_mech['Rate_RDS_dict'].iloc[-1], df_mech['Site_density_symbol'].iloc[-1], 
                                df_mech['Partial_pressure_dict'].iloc[-1],Initial_flowrates_overalldict, Exp_Outlet_flowrates_overalldict, Exp_Select_dict, Pressure, Exp_Conversion, 
                                Volume_CatWeight, Conv_species, Reactor_type, total_comb, 0, Known_values, Time=True)
    
    iteration_time = 0.5*(iteration_time_1 + iteration_time_2)
    
    total_sec = iteration_time*total_comb 
    total_min = total_sec/60
    total_hours = total_sec/3600
    total_days = total_hours/24
    ans = input('''The total number of mechanisms considered equals %d, the total number of parameter combinations for every mechanism equals %d. The uncatalyzed system is also considered with %d possible parameter combinations. The number of experimental datapoints considered is %d.
This amounts to a total of %d combinations.\n
Assuming a calculation time of %.3f s per combination, the execution will take approximately %d seconds or %.1f minutes or %.1f hours or %.1f days.\n
Do you wish to continue? (y/n) '''%(total_mech, total_paracomb, total_paracomb_uncat, num_exp, total_comb, iteration_time, round(total_sec), round(total_min,1), round(total_hours,1), round(total_days,1)))
    
    if ans in ['n', 'N', 'no', 'NO']:
        
        exit()
        
    print('\n')
    
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    
    print('Start time: ' + str(current_time))
    print('Calculating ...')
    
    header_complete = pd.DataFrame(columns=['Mechanism_number', 'Ads_dict', 'RDS_dict', 'Reaction_rate_dict', 'Net_production_dict', 'Eq_glob_expr_dict',
                                            'Final_flowrates', 'Conversion', 'Selectivities', 'Eq_ads', 'Eq_sr', 'Eq_glob','Lumped_rate_coefficient', 'Site_density', 
                                            'Rate_RDS', 'Rate_RDS_adap', 'Reversible_dict'])
    
    start_time = time.time()
    
    it = 0
    mech_it = 0
    rows_complete = []
    for index in df_mech.index: # Iterating over all the possible mechanisms listed in df_mech
    
        df_par, it_new = loop_par(Species_total, Overall_stoich, inerts, Select_stoich, df_mech['Net_production_dict'][index], df_mech['Eq_ads_symbol_dict'][index], df_mech['Eq_sr_symbol_dict'][index], 
                                  df_mech['Eq_glob_symbol_dict'][index], df_mech['Eq_glob_expr_dict'][index], df_mech['RDS_dict'][index], 
                                  df_mech['Rate_RDS_dict'][index], df_mech['Site_density_symbol'][index], df_mech['Partial_pressure_dict'][index],
                                  Initial_flowrates_overalldict, Exp_Outlet_flowrates_overalldict, Exp_Select_dict, Pressure, Exp_Conversion, Volume_CatWeight, Conv_species, Reactor_type, 
                                  total_comb, it, Known_values)
        
        it = it_new
        
        if df_par.empty == False:
            
            for index_2 in df_par.index:    
            
                rows_complete.append({'Mechanism_number':index, 'Ads_dict':df_mech['Ads_dict'][index], 'RDS_dict':df_mech['RDS_dict'][index], 'Reversible_dict':df_mech['Reversible_dict'][index],
                                      'Reaction_rate_dict':df_mech['Reaction_rate_dict'][index], 'Net_production_dict':df_mech['Net_production_dict'][index], 'Eq_glob_expr_dict':df_mech['Eq_glob_expr_dict'][index], 
                                      'Final_flowrates':df_par['Final_flowrates'][index_2], 'Conversion':df_par['Conversion'][index_2], 'Selectivities':df_par['Selectivities'][index_2],
                                      'Eq_ads':df_par['Eq_ads'][index_2], 'Eq_sr':df_par['Eq_sr'][index_2], 'Eq_glob':df_par['Eq_glob'][index_2], 'Lumped_rate_coefficient':df_par['Lumped_rate_coefficient'][index_2], 
                                      'Site_density':df_par['Site_density'][index_2] ,'Rate_RDS':df_par['Rate_RDS'][index_2], 'Rate_RDS_adap':df_par['Rate_RDS_adap'][index_2]})
            
        mech_it += 1 
        print('Mechanism #%d finished'%(mech_it))
        end_time = time.time() - start_time
        print('%f seconds have passed'%(end_time))
    
    df_complete = header_complete.append(rows_complete, ignore_index=False)

    
    return df_complete, start_time, total_comb

#%%
def ftex(df_complete, num_exp, total_comb, indep_var, dep_var, indep_list, dep_list, Pressure, Volume_CatWeight, SpaceTime, Temperature, Exp_Conversion, Exp_Select_dict):
    
    ziplist = list(zip(indep_list, dep_list))
    ziplist_sorted = sorted(ziplist, key = lambda x:x[0])
    unzip_sorted = list(zip(*ziplist_sorted))
    indep_list_s, dep_list_s = list(unzip_sorted[0]), list(unzip_sorted[1])
    
    Exp_features = ft.features(indep_list_s, dep_list_s)
    
    header_ftex = pd.DataFrame(columns=['Mechanism_number', 'Ads_dict', 'RDS_dict', 'Features', 'SRE', 'MSE', 'Reversible_dict', 'Reaction_rate_dict', 'Net_production_dict', 
                                        'Eq_glob_expr_dict', 'Final_flowrates', 'Conversion', 'Selectivities', 'Eq_ads', 'Eq_sr', 'Eq_glob','Lumped_rate_coefficient', 
                                        'Site_density', 'Rate_RDS', 'Features_extra', 'Rate_RDS_adap'])
    rows_ftex = []
    
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    start_ftex = time.time()
    for index in df_complete.index:
        
        # The following is done to make sure the independent variable is sorted from smalles to largest value in order to perform the feature extraction
        df_sort = pd.DataFrame(list(zip(indep_list, df_complete['Final_flowrates'][index], df_complete['Conversion'][index], df_complete['Selectivities'][index])), columns=[indep_var, 'Final_flowrates', 'Conversion', 'Selectivities'])
        df_sort = df_sort.sort_values(by=indep_var)
        df_sort = df_sort.reset_index(drop=True)
        
        try:
        
            if dep_var[:4] == 'SEL_':
                dep_list_theo = [df_sort['Selectivities'][j][dep_var[4:]] for j in df_sort.index]
                
            else:
                dep_list_theo = df_sort[dep_var].tolist()
    
            #start_solver = time.time()
            Theo_features = ft.features(indep_list_s, dep_list_theo)
            #print(time.time()- start_solver)

            
            # SRE can only be calculated for a model if the model has the same features as the experimental data, otherwise it does not make sense.
            
            if Theo_features[0] == Exp_features[0]:
                SRE = 0
            
                j=1
                
                while j<len(Theo_features[1])-1: # Because the first and last extremes are always the start and end of the dataset
                                
                    SRE += abs((Theo_features[1][j] - Exp_features[1][j])/Exp_features[1][j])
        
                    j+=1
            else:
                
                SRE = 'N/A'
                
            Sum_SE = 0
                
            for i in range(num_exp):
        
                Sum_SE += (dep_list_theo[i] - dep_list_s[i])**2 # Adding the squared errors of all datapoints
    
            MSE = Sum_SE/num_exp # Calculating the normalized mean squared error (MSE_norm)
        
        except:
            
            Theo_features = ['ERROR']
            SRE = 'ERROR'
            MSE = 'ERROR'
        
        rows_ftex.append({'Mechanism_number':df_complete['Mechanism_number'][index], 'Ads_dict':df_complete['Ads_dict'][index], 'RDS_dict':df_complete['RDS_dict'][index], 
                          'Features':Theo_features[0],'SRE':SRE, 'MSE':MSE, 'Reversible_dict':df_complete['Reversible_dict'][index], 'Reaction_rate_dict':df_complete['Reaction_rate_dict'][index], 
                          'Net_production_dict':df_complete['Net_production_dict'][index], 'Eq_glob_expr_dict':df_complete['Eq_glob_expr_dict'][index], 'Final_flowrates':df_sort['Final_flowrates'].tolist(), 
                          'Conversion':df_sort['Conversion'].tolist(), 'Selectivities':df_sort['Selectivities'].tolist(), 'Eq_ads':df_complete['Eq_ads'][index], 'Eq_sr':df_complete['Eq_sr'][index], 
                          'Eq_glob':df_complete['Eq_glob'][index],'Lumped_rate_coefficient':df_complete['Lumped_rate_coefficient'][index], 'Site_density':df_complete['Site_density'][index], 
                          'Rate_RDS':df_complete['Rate_RDS'][index], 'Features_extra':Theo_features, 'Rate_RDS_adap':df_complete['Rate_RDS_adap'][index]})
        
        end_ftex = time.time()
        
        if index == 0:
            
            time_ftex = end_ftex - start_ftex
            
            print('''Feature extraction takes approximately %.3f seconds and must be performed %d times. 
The estimated execution time is %d seconds or %.1f minutes or %.1f hours or %.1f days'''%(time_ftex, round(total_comb/num_exp), round(time_ftex*total_comb/num_exp), round((time_ftex*total_comb/num_exp)/60, 1), round((time_ftex*total_comb/num_exp)/3600, 1), round((time_ftex*total_comb/num_exp)/86400, 1)))
            
            print('\n')
            print('Start time feature extraction: ' + str(current_time))
            
            print('Extracting features ...')
            
        elif index % 1000 == 0 and index != 0:
            
            print('%d iterations of %d executed, %.1f %%'%(index, round(total_comb/num_exp), index*num_exp/total_comb*100))
        
    df_ftex = header_ftex.append(rows_ftex, ignore_index=False)
    
    return df_ftex, Exp_features

#%%
def rank(df_ftex, Exp_features, tol=0.5):
   
    mech_num = df_ftex.Mechanism_number.unique()
    
    header_rank = pd.DataFrame(columns=['Mechanism_number', 'Ads_dict', 'RDS_dict', 'Features', 'SRE', 'MSE', 'Reversible_dict', 'Reaction_rate_dict', 'Net_production_dict', 
                                        'Eq_glob_expr_dict', 'Final_flowrates', 'Conversion', 'Selectivities', 'Eq_ads', 'Eq_sr', 'Eq_glob','Lumped_rate_coefficient', 
                                        'Site_density', 'Rate_RDS', 'Features_extra', 'Rate_RDS_adap'])
    
    rows_rank = []
    
    for num in mech_num:
        
        header_rank_2 = pd.DataFrame(columns=['Mechanism_number', 'Ads_dict', 'RDS_dict', 'Features', 'SRE', 'MSE', 'Reversible_dict', 'Reaction_rate_dict', 'Net_production_dict', 
                                              'Eq_glob_expr_dict', 'Final_flowrates', 'Conversion', 'Selectivities', 'Eq_ads', 'Eq_sr', 'Eq_glob','Lumped_rate_coefficient', 
                                              'Site_density', 'Rate_RDS', 'Features_extra', 'Rate_RDS_adap'])
        rows_rank_2 = []
        
        df_mech_2 = df_ftex[df_ftex['Mechanism_number'] == num]
        
        for index in df_mech_2.index:
            
            if df_mech_2['Features_extra'][index][0] != Exp_features[0]:
                
                continue
            
            Theor_Extremes = df_mech_2['Features_extra'][index][1]
        
            Extremes_in_range = True    # Start assuming that all extremes are in range    
                
            j=1
            while j<len(Theor_Extremes)-1: # Because the first and last extremes are always the start and end of the dataset
                    
                if not (Exp_features[1][j]*(1-tol) <= Theor_Extremes[j] <= Exp_features[1][j]*(1+tol)):
                        
                    Extremes_in_range = False
                    break
                    
                else:
                        
                    j += 1
            
            if Extremes_in_range == True: # Means that all extremes are in range

                rows_rank_2.append({'Mechanism_number':df_mech_2['Mechanism_number'][index], 'Ads_dict':df_mech_2['Ads_dict'][index], 'RDS_dict':df_mech_2['RDS_dict'][index], 
                                    'Features':df_mech_2['Features'][index], 'SRE':df_mech_2['SRE'][index], 'MSE':df_mech_2['MSE'][index], 'Reversible_dict':df_mech_2['Reversible_dict'][index], 
                                    'Reaction_rate_dict':df_mech_2['Reaction_rate_dict'][index], 'Net_production_dict':df_mech_2['Net_production_dict'][index], 'Eq_glob_expr_dict':df_mech_2['Eq_glob_expr_dict'][index], 
                                    'Final_flowrates':df_mech_2['Final_flowrates'][index], 'Conversion':df_mech_2['Conversion'][index], 'Selectivities':df_mech_2['Selectivities'][index], 'Eq_ads':df_mech_2['Eq_ads'][index], 
                                    'Eq_sr':df_mech_2['Eq_sr'][index], 'Eq_glob':df_mech_2['Eq_glob'][index],'Lumped_rate_coefficient':df_mech_2['Lumped_rate_coefficient'][index], 
                                    'Site_density':df_mech_2['Site_density'][index], 'Rate_RDS':df_mech_2['Rate_RDS'][index], 'Features_extra':df_mech_2['Features_extra'][index], 
                                    'Rate_RDS_adap':df_mech_2['Rate_RDS_adap'][index]})
        
        df_comp = header_rank_2.append(rows_rank_2, ignore_index=False)
        
        df_comp_sort = df_comp.sort_values(by=['SRE', 'MSE'], ignore_index=True)
        
        if df_comp_sort.empty == False:
        
            rows_rank.append({'Mechanism_number':df_comp_sort['Mechanism_number'][0], 'Ads_dict':df_comp_sort['Ads_dict'][0], 'RDS_dict':df_comp_sort['RDS_dict'][0], 
                              'Features':df_comp_sort['Features'][0], 'SRE':df_comp_sort['SRE'][0], 'MSE':df_comp_sort['MSE'][0], 'Reversible_dict':df_comp_sort['Reversible_dict'][0], 
                              'Reaction_rate_dict':df_comp_sort['Reaction_rate_dict'][0], 'Net_production_dict':df_comp_sort['Net_production_dict'][0], 'Eq_glob_expr_dict':df_comp_sort['Eq_glob_expr_dict'][0], 
                              'Final_flowrates':df_comp_sort['Final_flowrates'][0], 'Conversion':df_comp_sort['Conversion'][0], 'Selectivities':df_comp_sort['Selectivities'][0], 'Eq_ads':df_comp_sort['Eq_ads'][0], 
                              'Eq_sr':df_comp_sort['Eq_sr'][0], 'Eq_glob':df_comp_sort['Eq_glob'][0], 'Lumped_rate_coefficient':df_comp_sort['Lumped_rate_coefficient'][0], 
                              'Site_density':df_comp_sort['Site_density'][0], 'Rate_RDS':df_comp_sort['Rate_RDS'][0], 'Features_extra':df_comp_sort['Features_extra'][0], 
                              'Rate_RDS_adap':df_comp_sort['Rate_RDS_adap'][0]})
    
    df_rank = header_rank.append(rows_rank, ignore_index=False)
    
    df_rank_sort = df_rank.sort_values(by=['SRE', 'MSE'], ignore_index=True)   
        
    return df_rank_sort

#%%
def get_lumped_k(Pressure, Exp_Conversion, Initial_flowrates_overalldict, Exp_Outlet_flowrates_overalldict, Exp_Select_dict, Volume_CatWeight, num_reactions, num_species, num_exp, Conv_species, Net_production_dict, Eq_ads_symbol, Eq_ads_values, Eq_sr_symbol, Eq_sr_values, Eq_glob_symbol, Eq_glob_expr_dict, Rate_RDS_symbol, Site_density_symbol, Partial_pressure_dict):
    
    if num_reactions == 1: # Net production rates and reaction rates are equal (taking the stoichiometry into account)
        
        Rate_RDS_values = {1:1}
        Site_density_value = 1
        
        k_list = []
        
        for index in range(num_exp):
            
            if bool(Exp_Outlet_flowrates_overalldict):
            
                Initial_flowrates_dict = {list(Initial_flowrates_overalldict.keys())[i]:Initial_flowrates_overalldict[list(Initial_flowrates_overalldict.keys())[i]][index] for i in range(num_species)}
                Exp_Outlet_flowrates_dict = {list(Exp_Outlet_flowrates_overalldict.keys())[i]:Exp_Outlet_flowrates_overalldict[list(Exp_Outlet_flowrates_overalldict.keys())[i]][index] for i in range(num_species)}
            
                Net_production_dict_values_in = substitute(Net_production_dict, Eq_ads_symbol, Eq_ads_values, Eq_sr_symbol, Eq_sr_values, Eq_glob_symbol, Eq_glob_expr_dict, Rate_RDS_symbol, Rate_RDS_values, Site_density_symbol, Site_density_value, Pressure[index], Partial_pressure_dict, Lump=True, Flowrates_values=Initial_flowrates_dict)
                Net_production_dict_values_out = substitute(Net_production_dict, Eq_ads_symbol, Eq_ads_values, Eq_sr_symbol, Eq_sr_values, Eq_glob_symbol, Eq_glob_expr_dict, Rate_RDS_symbol, Rate_RDS_values, Site_density_symbol, Site_density_value, Pressure[index], Partial_pressure_dict, Lump=True, Flowrates_values=Exp_Outlet_flowrates_dict)
            
                k_in = (Exp_Conversion[index] * Initial_flowrates_dict[Conv_species] / Volume_CatWeight[index]) / (- Net_production_dict_values_in[Conv_species])
                k_out = (Exp_Conversion[index] * Exp_Outlet_flowrates_dict[Conv_species] / Volume_CatWeight[index]) / (- Net_production_dict_values_out[Conv_species])
                k = 0.5 * (k_in + k_out)
                k_list.append(k)
            
            else:
                
                Initial_flowrates_dict = {list(Initial_flowrates_overalldict.keys())[i]:Initial_flowrates_overalldict[list(Initial_flowrates_overalldict.keys())[i]][index] for i in range(num_species)}
                
                Net_production_dict_values_in = substitute(Net_production_dict, Eq_ads_symbol, Eq_ads_values, Eq_sr_symbol, Eq_sr_values, Eq_glob_symbol, Eq_glob_expr_dict, Rate_RDS_symbol, Rate_RDS_values, Site_density_symbol, Site_density_value, Pressure[index], Partial_pressure_dict, Lump=True, Flowrates_values=Initial_flowrates_dict)
                
                k_in = (Exp_Conversion[index] * Initial_flowrates_dict[Conv_species] / Volume_CatWeight[index]) / (-1 * Net_production_dict_values_in[Conv_species])         
                k_list.append(k_in)
        
        k_avg = sum(k_list) / len(k_list)
        
        Lumped_k_dict = {1:k_avg}            
            
    else:
        
        #print("Lumping the active site density with the RDS rate coefficients and backcalculating this lumped parameter is not yet implemented for multiple reactions")
        #exit()
        
        Rate_RDS_values = {i:1 for i in range(1,num_reactions+1)}
        Lumped_k_dict = {}
        
        for i in range(1,num_reactions+1):
            
            """
            if i == 1: # If the system comprises of only reactions in series, the first rate coefficient k1 can be calculated in the same way as if there was only one reaction (assuming of course the numbering of the reactions is logical and reaction 1 is the first reaction of the series)
                        # This is not a good method, better to use just k_lump = 1 for every reaction
                
                Site_density_value = 1
        
                k_list = []
                
                for index in range(num_exp):
                    
                    if bool(Exp_Outlet_flowrates_overalldict):
                    
                        Initial_flowrates_dict = {list(Initial_flowrates_overalldict.keys())[i]:Initial_flowrates_overalldict[list(Initial_flowrates_overalldict.keys())[i]][index] for i in range(num_species)}
                        Exp_Outlet_flowrates_dict = {list(Exp_Outlet_flowrates_overalldict.keys())[i]:Exp_Outlet_flowrates_overalldict[list(Exp_Outlet_flowrates_overalldict.keys())[i]][index] for i in range(num_species)}
                    
                        Net_production_dict_values_in = substitute(Net_production_dict, Eq_ads_symbol, Eq_ads_values, Eq_sr_symbol, Eq_sr_values, Eq_glob_symbol, Eq_glob_expr_dict, Rate_RDS_symbol, Rate_RDS_values, Site_density_symbol, Site_density_value, Pressure[index], Partial_pressure_dict, Lump=True, Flowrates_values=Initial_flowrates_dict)
                        Net_production_dict_values_out = substitute(Net_production_dict, Eq_ads_symbol, Eq_ads_values, Eq_sr_symbol, Eq_sr_values, Eq_glob_symbol, Eq_glob_expr_dict, Rate_RDS_symbol, Rate_RDS_values, Site_density_symbol, Site_density_value, Pressure[index], Partial_pressure_dict, Lump=True, Flowrates_values=Exp_Outlet_flowrates_dict)
                    
                        k_in = (Exp_Conversion[index] * Initial_flowrates_dict[Conv_species] / Volume_CatWeight[index]) / (- Net_production_dict_values_in[Conv_species])
                        k_out = (Exp_Conversion[index] * Exp_Outlet_flowrates_dict[Conv_species] / Volume_CatWeight[index]) / (- Net_production_dict_values_out[Conv_species])
                        k = 0.5 * (k_in + k_out)
                        k_list.append(k)
                    
                    else:
                        
                        Initial_flowrates_dict = {list(Initial_flowrates_overalldict.keys())[i]:Initial_flowrates_overalldict[list(Initial_flowrates_overalldict.keys())[i]][index] for i in range(num_species)}
                        
                        Net_production_dict_values_in = substitute(Net_production_dict, Eq_ads_symbol, Eq_ads_values, Eq_sr_symbol, Eq_sr_values, Eq_glob_symbol, Eq_glob_expr_dict, Rate_RDS_symbol, Rate_RDS_values, Site_density_symbol, Site_density_value, Pressure[index], Partial_pressure_dict, Lump=True, Flowrates_values=Initial_flowrates_dict)
                        
                        k_in = (Exp_Conversion[index] * Initial_flowrates_dict[Conv_species] / Volume_CatWeight[index]) / (-1 * Net_production_dict_values_in[Conv_species])         
                        k_list.append(k_in)
                
                k_avg = sum(k_list) / len(k_list)
                Lumped_k_dict[i] = k_avg
            
            else:

                Lumped_k_dict[i] = 1
            """
            Lumped_k_dict[i] = 1
    return Lumped_k_dict

#%%
"""
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
