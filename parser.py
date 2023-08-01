#!/usr/bin/env python3

try:
    from dd.cudd import BDD as _bdd
except ImportError:
    print('\033[93m' + "CUDD was not installed, for better performance install CUDD bindings for python dd" + '\033[0m')
    from dd.autoref import BDD as _bdd


def parse_DNF(filename):
    features = []
    configurations = []
    parsed_configurations = []
    fm_bdd = _bdd()
    fm_universe = fm_bdd.false

    with open(filename + '.fs' , 'r') as f:
        features = [x.strip() for x in f.readlines() if x.startswith('#') is False and x.strip() != '']


    with open(filename + '.fm' , 'r') as f:
        configurations = f.readlines()
        configurations = [x.strip() for x in configurations if x.startswith('#') is False and x.strip() != '']    

    for configuration in configurations:
        tokens = configuration.split('&')
        tokens = [x.strip() for x in tokens]
        tokens = [x.replace('!', '~') for x in tokens]
        tokens = [x.replace('(' , '') for x in tokens]
        tokens = [x.replace(')' , '') for x in tokens]
        parsed_configurations.append(tokens)
    
    fm_bdd.declare(*features)

    for configuration in parsed_configurations:
        v = fm_bdd.true
        for token in configuration:
            if '|' in token:
                v_tmp = parse_OR_expr(token, fm_bdd)
            elif '^' in token:
                v_tmp = parse_XOR_expr(token, fm_bdd)
            elif token.startswith('~'):
                v_tmp = fm_bdd.apply('not', fm_bdd.var(token[1:]))
            else:
                v_tmp = fm_bdd.var(token)
            v = fm_bdd.apply('and', v, v_tmp)
        fm_universe = fm_bdd.apply('or', fm_universe, v)
        all_solutions = fm_bdd.pick_iter(fm_universe, care_vars=fm_bdd.vars)
        fm_bdd.reorder()

    return fm_bdd, fm_universe


def parse_OR_expr(expr, fm_bdd):
    tokens = expr.split('|')
    tokens = [x.strip() for x in tokens]
    print("OR expr:", tokens)
    
    token = tokens[0]
    if '^' in token:
        v = parse_XOR_expr(token, fm_bdd)
    elif token.startswith('~'):
        v = fm_bdd.apply('not', fm_bdd.var(token[1:]))
    else:
        v = fm_bdd.var(token)
    for token in tokens[1:]:
        if '^' in token:
            v_tmp = parse_XOR_expr(token, fm_bdd)
        elif token.startswith('~'):
            v_tmp = fm_bdd.apply('not', fm_bdd.var(token[1:]))
        else:
            v_tmp = fm_bdd.var(token)
        v = fm_bdd.apply('or', v, v_tmp)
    return v


def parse_XOR_expr(expr, fm_bdd):
    tokens = expr.split('^')
    tokens = [x.strip() for x in tokens]
    print("XOR expr:", tokens)
    
    token = tokens[0]
    if token.startswith('~'):
        v = fm_bdd.apply('not', fm_bdd.var(token[1:]))
    else:
        v = fm_bdd.var(token)
    for token in tokens[1:]:
        if token.startswith('~'):
            v_tmp = fm_bdd.apply('not', fm_bdd.var(token[1:]))
        else:
            v_tmp = fm_bdd.var(token)
        v = fm_bdd.apply('xor', v, v_tmp)
    return v