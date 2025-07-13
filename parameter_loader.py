"""
parameter_loader.py - General parameter loading module
Dynamically loads required parameters based on atom list
"""

import csv
import os
import itertools
from collections import defaultdict
import warnings

def load_atom_parameters(file_path: str, required_atoms: list) -> dict:
    """
    Load parameters for specified atoms from CSV file
    
    :param file_path: CSV file path
    :param required_atoms: List of atom symbols to load
    :return: Dictionary {atom symbol: parameter dictionary}
    """
    atom_params = {}
    found_atoms = set()
    required_set = set(required_atoms)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Atom parameter file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Check required fields
        required_fields = ['symbol']
        if not all(field in reader.fieldnames for field in required_fields):
            missing = [field for field in required_fields if field not in reader.fieldnames]
            raise ValueError(f"CSV file missing required fields: {', '.join(missing)}")
        
        for row in reader:
            symbol = row['symbol'].strip()
            
            # Only load required atoms
            if symbol not in required_set:
                continue
                
            # Ensure each atom is loaded only once
            if symbol in atom_params:
                raise ValueError(f"Atom '{symbol}' appears multiple times in file")
                
            params = {'symbol': symbol}
            
            # Process numerical parameters
            float_fields = [
                'charge', 'shannon_radius', 'mass', 'vdw_radius', 
                'covalent_radius', 'atomic_number'
            ]
            for field in float_fields:
                if field in row and row[field].strip():
                    try:
                        params[field] = float(row[field])
                    except ValueError:
                        warnings.warn(f"Atom '{symbol}' field '{field}' value '{row[field]}' cannot be converted to float")
            
            # Process string parameters
            str_fields = ['bonding_type', 'element_group', 'crystal_system']
            for field in str_fields:
                if field in row and row[field].strip():
                    params[field] = row[field].strip()
            
            atom_params[symbol] = params
            found_atoms.add(symbol)
    
    # Check if all required atoms were found
    missing_atoms = required_set - found_atoms
    if missing_atoms:
        warnings.warn(f"The following atoms are missing in the parameter file: {', '.join(missing_atoms)}")
    
    return atom_params

def load_atom_pair_parameters(file_path: str, required_atoms: list) -> dict:
    """
    Load atom pair parameters from CSV file
    
    :param file_path: CSV file path
    :param required_atoms: List of atom symbols to load
    :return: Dictionary {(element1, element2): parameter dictionary}
    """
    pair_params = {}
    required_set = set(required_atoms)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Atom pair parameter file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Check required fields
        required_fields = ['element1', 'element2']
        if not all(field in reader.fieldnames for field in required_fields):
            missing = [field for field in required_fields if field not in reader.fieldnames]
            raise ValueError(f"CSV file missing required fields: {', '.join(missing)}")
        
        for row in reader:
            elem1 = row['element1'].strip()
            elem2 = row['element2'].strip()
            
            # Only load pairs where both atoms are in required list
            if elem1 not in required_set or elem2 not in required_set:
                continue
                
            # Create normalized key (sorted tuple)
            key = tuple(sorted([elem1, elem2]))
            
            # Avoid duplicates
            if key in pair_params:
                warnings.warn(f"Atom pair '{elem1}-{elem2}' appears multiple times in file, using last occurrence")
            
            params = {}
            
            # Process Buckingham parameters
            if 'buckingham_A' in row and row['buckingham_A'].strip():
                buck_params = {
                    'A': float(row['buckingham_A']),
                    'rho': float(row.get('buckingham_rho', 0.3)),
                    'C': float(row.get('buckingham_C', 0.0))
                }
                params['buckingham'] = buck_params
            
            # Process electrostatic parameters
            if 'epsilon_r' in row and row['epsilon_r'].strip():
                elec_params = {'epsilon_r': float(row['epsilon_r'])}
                params['electrostatic'] = elec_params
            
            # Process bond order potential parameters
            if 'bond_model' in row and row['bond_model'].strip():
                bond_model = row['bond_model'].strip().lower()
                bond_params = {'model': bond_model}
                
                if bond_model == 'morse':
                    if 'morse_D' in row and row['morse_D'].strip():
                        bond_params.update({
                            'D': float(row['morse_D']),
                            'alpha': float(row.get('morse_alpha', 2.0)),
                            'r0': float(row.get('morse_r0', 2.0))
                        })
                
                elif bond_model == 'tersoff':
                    # Load Tersoff parameters
                    tersoff_params = {}
                    tersoff_fields = [
                        'tersoff_r0', 'tersoff_A', 'tersoff_B', 'tersoff_lambda1',
                        'tersoff_lambda2', 'tersoff_beta', 'tersoff_n', 'tersoff_c',
                        'tersoff_d', 'tersoff_h', 'tersoff_R', 'tersoff_S'
                    ]
                    for field in tersoff_fields:
                        base_field = field.replace('tersoff_', '')
                        if field in row and row[field].strip():
                            tersoff_params[base_field] = float(row[field])
                    
                    bond_params.update(tersoff_params)
                
                params['bond_order'] = bond_params
            
            # Process harmonic potential parameters
            if 'harmonic_k' in row and row['harmonic_k'].strip():
                harm_params = {
                    'k': float(row['harmonic_k']),
                    'r0': float(row.get('harmonic_r0', 2.0))
                }
                params['harmonic'] = harm_params
            
            # Process LJ parameters
            if 'lj_epsilon' in row and row['lj_epsilon'].strip():
                lj_params = {
                    'epsilon': float(row['lj_epsilon']),
                    'sigma': float(row.get('lj_sigma', 3.0))
                }
                params['lennard_jones'] = lj_params
            
            # Add general parameters
            if 'cutoff' in row and row['cutoff'].strip():
                params['cutoff'] = float(row['cutoff'])
            
            if 'comment' in row and row['comment'].strip():
                params['comment'] = row['comment'].strip()
            
            pair_params[key] = params
    
    return pair_params

def load_all_parameters(atom_list: list, data_dir: str = "data") -> tuple:
    """
    Load all relevant parameters for given atom list
    
    :param atom_list: List of atom symbols (e.g., ['Ga', 'N'])
    :param data_dir: Data file directory
    :return: (atom parameters, atom pair parameters)
    """
    atom_file = os.path.join(data_dir, "atom_parameters.csv")
    pair_file = os.path.join(data_dir, "atom_pair_parameters.csv")
    
    atom_params = load_atom_parameters(atom_file, atom_list)
    pair_params = load_atom_pair_parameters(pair_file, atom_list)
    
    # Ensure all possible atom pairs have parameters
    all_pairs = set(tuple(sorted(pair)) for pair in itertools.combinations_with_replacement(atom_list, 2))
    missing_pairs = all_pairs - set(pair_params.keys())
    
    if missing_pairs:
        warnings.warn(f"The following atom pairs are missing parameters: {', '.join(f'{p[0]}-{p[1]}' for p in missing_pairs)}")
    
    return atom_params, pair_params