"""
main.py - Main program for atom type assignment
Dynamically loads parameters based on atom list

Author: Jian Sun.
Recent Updated: Jul/14/2025
"""

import numpy as np
from lattice_grid import LatticeGrid, CrystalSystemFactory
from potentials import calculate_pair_energy
from parameter_loader import load_all_parameters
import time
import itertools
from collections import defaultdict

def assign_atom_types(positions: list, composition: dict) -> list:
    """
    Assign atom types based on composition ratio
    
    :param positions: List of grid point positions
    :param composition: Atomic composition dictionary {element: count}
    :return: List of atom types
    """
    total_sites = len(positions)
    total_atoms = sum(composition.values())
    
    # Validate grid points match atom count
    if total_sites < total_atoms:
        raise ValueError(f"Number of grid points ({total_sites}) is less than number of atoms ({total_atoms})")
    elif total_sites > total_atoms:
        print(f"Warning: Number of grid points ({total_sites}) exceeds number of atoms ({total_atoms}), vacancies will be created")
    
    # Create atom type list
    atom_types = []
    atom_counts = composition.copy()
    
    # Create all atoms to assign
    atoms_to_assign = []
    for element, count in composition.items():
        atoms_to_assign.extend([element] * count)
    
    # Add vacancies if grid points exceed atoms
    atoms_to_assign.extend([None] * (total_sites - total_atoms))
    
    # Randomize atom positions (should be optimized in actual applications)
    np.random.shuffle(atoms_to_assign)
    
    return atoms_to_assign

def main():
    try:
        # 1. Define crystal composition and atom list
        composition = {'Ga':2, 'N':2}  # Atom count for GaN crystal
        atom_list = list(composition.keys())  # Required atom types
        
        # 2. Load all relevant parameters
        atom_params, pair_params = load_all_parameters(atom_list)
        
        # 3. Create lattice (hexagonal GaN example)
        lattice = CrystalSystemFactory.hexagonal(
            a=3.19, c=5.19, divisions=(6, 6, 4)
        )
        
        # 4. Precompute all fractional coordinates
        positions = lattice.get_cartesian_sites()
        frac_positions = np.array([lattice.cartesian_to_fractional(pos) for pos in positions])
        
        # 5. Assign atom types
        atom_types = assign_atom_types(positions, composition)
        
        # 6. Efficiently compute total energy
        total_energy = 0.0
        count = 0
        interactions_count = defaultdict(int)
        
        start_time = time.time()
        for i in range(len(frac_positions)):
            pos_i = frac_positions[i]
            type_i = atom_types[i]
            
            # Skip if position is vacancy
            if type_i is None:
                continue
                
            for j in range(i + 1, len(frac_positions)):
                type_j = atom_types[j]
                
                # Skip if position is vacancy or same position
                if type_j is None or type_i == type_j and i == j:
                    continue
                
                # Calculate fractional coordinate difference
                delta = frac_positions[j] - pos_i
                delta -= np.round(delta)  # Minimum image convention
                
                # Calculate distance squared
                dist_sq = np.dot(delta, np.dot(lattice.metric_tensor, delta))
                
                # Get atom pair parameters
                pair_key = tuple(sorted([type_i, type_j]))
                pair_param = pair_params.get(pair_key, {})
                
                # Get cutoff distance (default 5.0Å)
                cutoff = pair_param.get('cutoff', 5.0)
                
                # Skip too distant pairs
                if dist_sq > cutoff ** 2:
                    continue
                    
                dist = np.sqrt(dist_sq)
                
                # Get atom parameters
                atom_i_param = atom_params[type_i]
                atom_j_param = atom_params[type_j]
                
                # Calculate potential energy
                energy = calculate_pair_energy(atom_i_param, atom_j_param, pair_param, dist)
                total_energy += energy
                count += 1
                interactions_count[pair_key] += 1
        
        # 7. Output results
        calc_time = time.time() - start_time
        print(f"\nCalculation completed, time taken: {calc_time:.2f}s")
        print(f"Total grid points: {len(positions)}")
        print(f"Assigned atoms: {sum(composition.values())}")
        print(f"Interactions calculated: {count}")
        print(f"Total potential energy: {total_energy:.2f} eV")
        
        # 8. Print interaction statistics
        print("\nInteraction statistics:")
        for pair, count in interactions_count.items():
            print(f"- {pair[0]}-{pair[1]}: {count} pairs")
        
        # 9. Validate typical bond energies
        print("\nTypical bond energy validation:")
        for pair in pair_params:
            if pair not in interactions_count:
                continue
                
            type1, type2 = pair
            atom1_param = atom_params[type1]
            atom2_param = atom_params[type2]
            pair_param = pair_params[pair]
            
            # Calculate energy at typical bond length
            typical_bond_length = 2.0  # Default value
            if 'bond_order' in pair_param and 'r0' in pair_param['bond_order']:
                typical_bond_length = pair_param['bond_order']['r0']
            elif 'harmonic' in pair_param and 'r0' in pair_param['harmonic']:
                typical_bond_length = pair_param['harmonic']['r0']
            elif 'buckingham' in pair_param:
                # For Buckingham potential, use rho as reference
                typical_bond_length = pair_param['buckingham'].get('rho', 0.3) * 2
            
            energy = calculate_pair_energy(atom1_param, atom2_param, pair_param, typical_bond_length)
            #print(f"{type1}-{type2} bond energy @{typical_bond_length:.3f}Å: {energy:.2f} eV")

        # Validation section in main.ipynb
        print(f"Ga-N bond energy @1.95Å: {calculate_pair_energy(atom_params['Ga'], atom_params['N'], pair_params[('Ga','N')], 1.95):.2f} eV")
        print(f"Ga-Ga bond energy @2.50Å: {calculate_pair_energy(atom_params['Ga'], atom_params['Ga'], pair_params[('Ga','Ga')], 2.50):.2f} eV")
        print(f"N-N bond energy @1.40Å: {calculate_pair_energy(atom_params['N'], atom_params['N'], pair_params[('N','N')], 1.40):.2f} eV")    
    
    except Exception as e:
        print(f"Error occurred during execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()