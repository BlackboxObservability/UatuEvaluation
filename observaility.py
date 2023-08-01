#!/usr/bin/env python3

# Imports
import pandas as pd
try:
    from dd.cudd import BDD as _bdd
except ImportError:
    from dd.autoref import BDD as _bdd
import os
import time

# Own imports
from utils import *

class bc:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DARK = '\033[2;37m'

    def color_string(self, color, string):
        return color + string + self.ENDC


def read_csv(filename, delim):

    # Error handling if file does not exist
    if not os.path.exists(filename):
        print('File does not exist')
        exit(1)

    return pd.read_csv(filename, delimiter=delim)


def read_fm(filename):
    featuremodel = filename + '.fm'
    featurespace = filename + '.fs'
    if not os.path.isfile(featuremodel):
        print('FM for', featuremodel, 'does not exist')
        exit(1)
    if not os.path.isfile(featurespace):
        print('FS for', featurespace, 'does not exist')
        exit(1)
    
    with open(featuremodel, 'r') as f:
        lines = f.readlines()
        configurations = [line if line[0] != '#' else '' for line in lines]
        configurations = list(filter(None, [c.strip() for c in configurations]))
    
    with open(featurespace, 'r') as f:
        lines = f.readlines()
        features = [line.strip() if line[0] != '#' else '' for line in lines]
        features = list(filter(None, [f.strip() for f in features]))

    return features, configurations


def construct_BDD_from_FM(features, configurations):
    fm = _bdd()
    fm.configure(reordering=True)

    for feature in features:
        fm.declare(feature)
    
    v = fm.add_expr('False')

    for configuration in configurations:
        u = fm.add_expr(configuration)
        v = fm.apply('or', v, u)
    _bdd.reorder(fm)
    fm.dump('bdd.pdf', roots=[v])
    return fm, v


def construct_BDD_from_DF(configurations):
    """
        Function that gets a dataframe with configurations 
        columns are features and in each row there is a 1 if the feature is active and a 0 if the feature is inactive
        each line represents one valid configuration
        all lines together represent the valid configuration space
        the function returns a BDD that represents the valid configuration space
    """
    fm = _bdd()
    fm.configure(reordering=True)
    
    features = []

    for feature in configurations.columns:
        if feature == 'Unnamed: 0':
            continue
        fm.declare(feature)
        features.append(feature)

    v = fm.add_expr('False')

    for _, row in configurations.iterrows():
        if row[features[0]] == 1:
            v_config = fm.var(features[0])
        else:
            v_config = fm.apply('not', fm.var(features[0]))
        for feature in features[1:]:
            if row[feature] == 1:
                v_tmp = fm.var(feature)
            else:
                v_tmp = fm.apply('not', fm.var(feature))
            v_config = fm.apply('and', v_config, v_tmp)
        v = fm.apply('or', v, v_config)
    _bdd.reorder(fm)
    fm.dump('bdd.pdf', roots=[v])
    return fm, v


def get_one_wise_partial_feature_assignments(variables):
    """
        Function that gets a list of variables and returns a list of partial feature assignments
    """
    pfas = []
    for variable in variables:
        pfas.append([variable])
        pfas.append(['!' + variable])
    return pfas


def get_two_wise_partial_feature_assignments(variables):
    """
        Function that gets a list of variables and returns a list of partial feature assignments
    """
    pfas = []
    variables = list(variables)
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            pfas.append([variables[i], variables[j]])
            pfas.append([variables[i], '!' + variables[j]])
            pfas.append(['!' + variables[i], variables[j]])
            pfas.append(['!' + variables[i], '!' + variables[j]])
    return pfas


def get_three_wise_partial_feature_assignments(variables):
    """
        Function that gets a list of variables and returns a list of partial feature assignments
    """
    pfas = []
    variables = list(variables)
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            for k in range(j + 1, len(variables)):
                pfas.append([variables[i], variables[j], variables[k]])
                pfas.append([variables[i], variables[j], '!' + variables[k]])
                pfas.append([variables[i], '!' + variables[j], variables[k]])
                pfas.append([variables[i], '!' + variables[j], '!' + variables[k]])
                pfas.append(['!' + variables[i], variables[j], variables[k]])
                pfas.append(['!' + variables[i], variables[j], '!' + variables[k]])
                pfas.append(['!' + variables[i], '!' + variables[j], variables[k]])
                pfas.append(['!' + variables[i], '!' + variables[j], '!' + variables[k]])
    return pfas


def get_valid_configs(fm, v, pfa):
    """
        Function that returns a list of all valid configurations for a partial feature assignment
    """
    expr = ' & '.join(pfa)
    u = fm.add_expr(expr)
    w = fm.apply('and', v, u)
    return fm, w


def get_counterfactual_witnesses(fm, v, pfa):
    """
        Function that returns a list of counterfactual witnesses for a partial feature assignment
    """
    tmp_pfa = switch_pfa(pfa)
    expr = ' & '.join(tmp_pfa)

    u = fm.add_expr(expr)
    w = fm.apply('and', v, u)
    return fm, w


def count_valid_pfas(fm, v, pfas):
    """
        Function that counts the number of partial feature assignments with at least one valid configuration
        and returns a list of partial feature assignments without any valid configuration

        returns: number of valid partial feature assignments and a list of invalid partial feature assignments
    """

    count = 0
    invalid_pfas = []
    for pfa in pfas:
        _, univ = get_valid_configs(fm, v, pfa)
        if univ == fm.false:
            invalid_pfas.append(pfa)
        else:
            count += 1
    return count, invalid_pfas


def switch_pfa(pfa):
    """
        Function that switches the assignments of every feature in a partial feature assignment 
    """
    new_pfa = pfa.copy()
    for i in range(len(new_pfa)):
        if new_pfa[i][0] == '!':
            new_pfa[i] = new_pfa[i][1:]
        else:
            new_pfa[i] = '!' + new_pfa[i]

    return new_pfa


def support(pfa):
    """
        Function that returns the support of a partial feature assignment
    """
    return set([feature[1:] if feature[0] == '!' else feature for feature in pfa])


def count_counterfactual_witnesses(fm, v, pfas):
    """
        Function that counts the number of partial feature assignments with at least one counterfactual witness
        and returns a list of partial feature assignments without any counterfactual witness

        returns: number of partial feature assignments with at least one counterfactual witness and a list of partial feature assignments without any counterfactual witness
    """

    count = 0
    invalid_pfas = []
    for pfa in pfas:
        _, univ = get_counterfactual_witnesses(fm, v, pfa)
        if univ == fm.false:
            count += 1
        else:
            invalid_pfas.append(pfa)
    return count, invalid_pfas


def check_eta_bar(fm, witness_u, counterfactual_witness_u, pfa):
    supp = support(pfa)
    for witness in fm.pick_iter(witness_u, fm.vars):
        for counterfactual_witness in fm.pick_iter(counterfactual_witness_u, fm.vars):
            keys = witness_u.support.intersection(counterfactual_witness_u.support)
            if all(witness[key] == counterfactual_witness[key] for key in keys.difference(supp)):
                return True
            else:
                continue
    return False
    

def check_observability(fm, v, pfa):
    _, witness_u = get_valid_configs(fm, v, pfa)
    _, counterfactual_witness_u = get_counterfactual_witnesses(fm, v, pfa)

    is_observable = check_eta_bar(fm, witness_u, counterfactual_witness_u, pfa)
    return is_observable


def check_observability_pfas(fm, v, pfas, obs_pfa_univs, unobs_pfa_univs):
    direct_observable_pfas = []
    indirect_observable_pfas = []
    unobservable_pfas = []

    for pfa in pfas:

        size_pfa = len(pfa)
        u_obs_pfa = obs_pfa_univs[size_pfa]
        u_unobs_pfa = unobs_pfa_univs[size_pfa]

        u_pfa = fm.add_expr(' & '.join(pfa))

        # check if the partial feature assignment was already checked
        if u_obs_pfa != fm.false and (u_pfa | ~ u_obs_pfa) == fm.true:
            direct_observable_pfas.append(pfa)
            continue
        elif u_unobs_pfa != fm.false and (u_pfa | ~ u_unobs_pfa) == fm.true:
            print(bc.BOLD + 'Already checked:', pfa, 'is unobservable.' + bc.ENDC)
            continue

        # check if the partial feature assignment is direct observable
        if check_observability(fm, v, pfa):
            direct_observable_pfas.append(pfa)
            u_obs_pfa = fm.apply('or', u_obs_pfa, u_pfa)
            continue
        
        """
        Check if the partial feature assignment is indirect observable, iff it is not direct observable and 
        it is not already in the set of unobservable partial feature assignments and 
        the size of the partial feature assignment is greater than 1
        """
        if len(pfa) > 1:
            # partition the partial feature assignment into all possible partitions
            partitioned_pfas = partCombo(pfa)

            # check if one of the partitions is direct observable
            for partitioned_pfa in partitioned_pfas:

                for partial_pfa in partitioned_pfa:
                    u_tmp_unobs = unobs_pfa_univs[len(partial_pfa)]
                    u_tmp_obs = obs_pfa_univs[len(partial_pfa)]

                    u_partial_pfa = fm.add_expr(' & '.join(partial_pfa))
                    if u_tmp_unobs != fm.false and (u_partial_pfa | ~ u_tmp_unobs) == fm.true:
                        break
                    elif u_tmp_obs == fm.false or (u_partial_pfa | ~ u_tmp_obs) != fm.true:
                        if not check_observability(fm, v, partial_pfa):
                            u_tmp_unobs = fm.apply('or', u_tmp_unobs, fm.add_expr(' & '.join(partial_pfa)))
                            break
                        else:
                            u_tmp_obs = fm.apply('or', u_tmp_obs, fm.add_expr(' & '.join(partial_pfa)))
                    
                    unobs_pfa_univs[len(partial_pfa)] = u_tmp_unobs
                    obs_pfa_univs[len(partial_pfa)] = u_tmp_obs
                else:
                    indirect_observable_pfas.append(pfa)
                    u_obs_pfa = fm.apply('or', u_obs_pfa, u_pfa)
                    break
            else:
                unobservable_pfas.append(pfa)
                u_unobs_pfa = fm.apply('or', u_unobs_pfa, u_pfa)
        else:
            unobservable_pfas.append(pfa)
            u_unobs_pfa = fm.apply('or', u_unobs_pfa, u_pfa)
        
        obs_pfa_univs[size_pfa] = u_obs_pfa
        unobs_pfa_univs[size_pfa] = u_unobs_pfa

    return direct_observable_pfas, indirect_observable_pfas, unobservable_pfas, obs_pfa_univs, unobs_pfa_univs


"""
    Evaluation part of real world partial feature assignments
"""
def get_real_world_pfas(filename, excluded_columns, delimiter=';'):
    """
        Function that gets a csv file with partial feature assignments and returns a list of partial feature assignments
    """
    df = read_csv(filename, delimiter)
    columns = list(set(df.columns).difference(set(excluded_columns)))
    pfas = set() 
    for col, row in df.iterrows():
        for column in columns:
            if row[column] != 0:
                pfa = '&'.join([c.strip() for c in column.split('*') if c != ''])
                pfas.add(pfa)
    pfas = [pfa.split('&') for pfa in pfas]
    return pfas


def get_real_world_feature_model_DF(filename, excluded_columns, delimiter=';'):
    df = read_csv(filename, delimiter)
    df = df[df['revision'] == df['revision'].tolist()[0]]
    df = df.drop(columns=excluded_columns)

    bdd, univ = construct_BDD_from_DF(df)
    return bdd, univ