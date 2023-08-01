#! /usr/bin/env python3

# general imports
from dd.autoref import BDD as _bdd
import os
import pandas as pd
import time

# imports from other files
from bdd_parser import parse_DNF
from observaility import *

experiments = [
    'RWSPL/llvm/llvmPaper',
    'RWSPL/lrzip/lrzipPaper',
    'PCSimp/E1/Apache_P/Apache_PFW',
    'PCSimp/E1/Curl_MEM/Curl_MEMFW',
    'PCSimp/E1/EMail_P/EMail_PFW',
    'PCSimp/E1/h264_MEM/h264_MEMFW',
    'PCSimp/E1/LinkedList_BS/LinkedList_BSDW',
    'PCSimp/E1/PKJab_BS/PKJab_BSFW',
    'PCSimp/E1/Prevaylar_BS/Prevaylar_BSFW',
    'PCSimp/E1/ZipMe_BS/ZipMe_BSFW',
    'ProVeLines/cfdp/cfdp',
    'ProVeLines/elevator/elevator',
    'ProVeLines/minepump/minepump',
    'Prism/BSN/BSN',
    'Prism/aircraft/aircraft',
]

experiment_dir = './examples/'
results_dir = './results/'


performance_influence_model_examples = [
    ['FastDownward', ['revision', 'performance']],
    ['HSQLDB', ['revision', 'performance', 'cpu', 'benchmark-energy', 'fixed-energy', 'benchmark-power', 'fixed-power']],
    ['MariaDB', ['revision', 'performance', 'cpu']],
    ['MySQL', ['revision', 'performance', 'cpu']],
    ['PostgreSQL', ['revision', 'performance', 'cpu']],
    ['OpenVPN', ['revision', 'performance']],
    ['Opus', ['revision', 'performance']],
    ['VP8', ['revision', 'performance', 'size', 'energy', 'cpu']],
    ['z3', ['revision', 'performance', 'memory']],
    ['lrzip', ['revision', 'performance', 'size', 'cpu']],
    ['brotli', ['revision', 'performance', 'size', 'memory', 'energy']],
]

pim_base_dir = '../PerformanceEvolution_Website/PerformanceEvolution_Data/'
pim_model_suffix = 'models/models.csv'
fm_model_suffix = 'measurements.csv'

pim_model_exclude = ['revision', 'error']

class bc:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    def color_string(self, color, string):
        return color + string + self.ENDC


def check_experiments_exist():
    for experiment in experiments:
        if not os.path.isfile(experiment_dir + experiment + '.fm'):
            print(bc().color_string(bc.FAIL, 'Experiment ' + experiment + ' does not exist!'))
            exit(1)
    print(bc().color_string(bc.OKGREEN, 'All experiments exist!'))


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description='Run experiments')

    #Argument to check experiments exist
    parser.add_argument('--check', action='store_true', help='Check if experiments exist')
    #Argument to run all experiments
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    #Argument to run one specific experiment
    parser.add_argument('--exp', help='Run specific experiment')
    #Argument to specify results directory
    parser.add_argument('--results-dir', help='Specify results directory')
    #Argument to run experiments with performance influence models
    parser.add_argument('--pim', action='store_true', default=False, help='Run experiments with performance influence models')

    args = parser.parse_args()
    return args


def run_experiment(experiment, output):
    print(bc().color_string(bc.OKGREEN, 'Read feature model'))
    if 'SNW_BSFW' in experiment:
        bdd, univ = construct_BDD_from_FM(experiment)
    else:
        bdd, univ = parse_DNF(experiment)

    obs_pfas = [bdd.false] * len(bdd.vars)
    unobs_pfas = [bdd.false] * len(bdd.vars)

    start_time_one_wise = time.time()
    one_wise_pfas = get_one_wise_partial_feature_assignments(bdd.vars)
    number_valid_one_wise_pfas, invalid_one_wise_pfas = count_valid_pfas(bdd, univ, one_wise_pfas)
    direct_observable_one_wise_pfas, indirect_observable_one_wise_pfas, unobservable_one_wise_pfas, obs_pfas, unobs_pfas = check_observability_pfas(bdd, univ, one_wise_pfas, obs_pfas, unobs_pfas)
    end_time_one_wise = time.time()

    start_time_two_wise = time.time()
    two_wise_pfas = get_two_wise_partial_feature_assignments(bdd.vars)
    number_valid_two_wise_pfas, invalid_two_wise_pfas = count_valid_pfas(bdd, univ, two_wise_pfas)
    direct_observable_two_wise_pfas, indirect_observable_two_wise_pfas, unobservable_two_wise_pfas, obs_pfas, unobs_pfas= check_observability_pfas(bdd, univ, two_wise_pfas, obs_pfas, unobs_pfas)
    end_time_two_wise = time.time()

    start_time_three_wise = time.time()
    three_wise_pfas = get_three_wise_partial_feature_assignments(bdd.vars)
    number_valid_three_wise_pfas, invalid_three_wise_pfas = count_valid_pfas(bdd, univ, three_wise_pfas)
    direct_observable_three_wise_pfas, indirect_observable_three_wise_pfas, unobservable_three_wise_pfas, obs_pfas, unobs_pfas = check_observability_pfas(bdd, univ, three_wise_pfas, obs_pfas, unobs_pfas)
    end_time_three_wise = time.time()

    statistics = {'Experiment': [experiment.split('/')[-1]] * 3,
                  'PFA_size': [1, 2, 3],
                  'Number_valid_configurations': [univ.count(len(bdd.vars))] * 3,
                  'Number_features': [len(bdd.vars)] * 3,
                  'Number_PFAs': [len(one_wise_pfas), len(two_wise_pfas), len(three_wise_pfas)],
                  'Number_valid_PFAs': [number_valid_one_wise_pfas, number_valid_two_wise_pfas, number_valid_three_wise_pfas],
                  'Number_invalid_PFAs': [len(invalid_one_wise_pfas), len(invalid_two_wise_pfas), len(invalid_three_wise_pfas)],
                  'Number_direct_observable_PFAs': [len(direct_observable_one_wise_pfas), len(direct_observable_two_wise_pfas), len(direct_observable_three_wise_pfas)],
                  'Number_indirect_observable_PFAs': [len(indirect_observable_one_wise_pfas), len(indirect_observable_two_wise_pfas), len(indirect_observable_three_wise_pfas)],
                  'Number_non-observable_PFAs': [len(unobservable_one_wise_pfas) - len(invalid_one_wise_pfas), len(unobservable_two_wise_pfas) - len(invalid_two_wise_pfas), len(unobservable_three_wise_pfas) - len(invalid_three_wise_pfas)],
                  'Time': [end_time_one_wise - start_time_one_wise, end_time_two_wise - start_time_two_wise, end_time_three_wise - start_time_three_wise]}
    statistics = pd.DataFrame(statistics)
    print(statistics)
    return statistics


"""
    In this part we evaluate the observability of partial feature assignments used in 
    performance influence models of real-world case studies.
"""
def evaluate_real_world_pfas():
    """
        Function that evaluates the real world partial feature assignments
    """

    for example in performance_influence_model_examples:
        print(bc().color_string(bc.OKGREEN, 'Read feature model for ' + example[0]))
        fm, univ = get_real_world_feature_model_DF(pim_base_dir + example[0] + '/' + fm_model_suffix, example[1])
        statistics = {'Experiment': [],
                      'PFA_size': [],
                      'Number_valid_configurations': [],
                      'Number_features': [],
                      'Number_PFAs': [],
                      'Number_valid_PFAs': [],
                      'Number_invalid_PFAs': [],
                      'Number_direct_observable_PFAs': [],
                      'Number_indirect_observable_PFAs': [],
                      'Number_non-observable_PFAs': [],
                      'Time': []}
        print(bc().color_string(bc.OKGREEN, 'Construct partial feature assignments for ' + example[0]))
        pfas = get_real_world_pfas(pim_base_dir + example[0] + '/' + pim_model_suffix, pim_model_exclude)
        pfas = sorted(pfas, key=len)
        print('Number of PFAs: ' + str(len(pfas)))

        for i in range(1, len(pfas[-1]) + 1):
            pfas_size_i = [pfa for pfa in pfas if len(pfa) == i]
            print('Number of ' + str(i) + '-wise PFAs: ' + str(len(pfas_size_i)))

            obs_pfas = [fm.false] * len(fm.vars)
            unobs_pfas = [fm.false] * len(fm.vars)
            start_time = time.time()
            number_valid_pfas, invalid_pfas = count_valid_pfas(fm, univ, pfas_size_i)
            direct_observable_pfas, indirect_observable_pfas, unobservable_pfas, obs_pfas, unobs_pfas = check_observability_pfas(fm, univ, pfas_size_i, obs_pfas, unobs_pfas)
            end_time = time.time()
            statistics['Experiment'].append(example[0])
            statistics['PFA_size'].append(i)
            statistics['Number_valid_configurations'].append(univ.count(len(fm.vars)))
            statistics['Number_features'].append(len(fm.vars))
            statistics['Number_PFAs'].append(len(pfas_size_i))
            statistics['Number_valid_PFAs'].append(number_valid_pfas)
            statistics['Number_invalid_PFAs'].append(len(invalid_pfas))
            statistics['Number_direct_observable_PFAs'].append(len(direct_observable_pfas))
            statistics['Number_indirect_observable_PFAs'].append(len(indirect_observable_pfas))
            statistics['Number_non-observable_PFAs'].append(len(unobservable_pfas) - len(invalid_pfas))
            statistics['Time'].append(end_time - start_time)
        statistics = pd.DataFrame(statistics)
        statistics.to_csv(results_dir + example[0] + '_statistics.csv', index=False)
        

def main(args):
    if args.check:
        check_experiments_exist()
    if args.all:
        statistics = pd.DataFrame()
        for experiment in experiments:
            print(bc().color_string(bc.BOLD, 'Running ' + experiment))
            stat = run_experiment(experiment_dir + experiment, experiment_dir + experiment + '.csv')
            stat.to_csv(results_dir + experiment.split('/')[-1] + '_statistics.csv', index=False)
        print(bc().color_string(bc.OKGREEN, 'All experiments finished!'))
    if args.exp:
        stat = run_experiment(args.exp, args.exp + '.csv')
        if args.results_dir:
            stat.to_csv(args.results_dir + args.exp.split('/')[-1] + '_statistics.csv', index=False)
        else:
            stat.to_csv(results_dir + args.exp.split('/')[-1] + '_statistics.csv', index=False)
    if args.pim:
        evaluate_real_world_pfas()

        

if __name__ == '__main__':
    args = parse_arguments()
    main(args)