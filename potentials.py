"""
potentials.py - General potential energy calculation module
Supports multiple potential models: Buckingham, electrostatic, bond-order (Morse/Tersoff), etc.
"""

import numpy as np
from math import exp, sqrt
import warnings

# Physical constants
KE_EV_ANG = 14.3996  # eV·Å / e² (Coulomb constant)
BOLTZMANN_CONST = 8.617333262145e-5  # eV/K (Boltzmann constant)

def buckingham_energy(r: float, A: float, rho: float, C: float) -> float:
    """
    Calculate Buckingham potential energy (repulsion + dispersion terms)
    
    :param r: Interatomic distance (Å)
    :param A: Repulsion parameter (eV)
    :param rho: Repulsion length scale (Å)
    :param C: Attraction parameter (eV·Å⁶)
    :return: Potential energy value (eV)
    """
    if r < 1e-6:  # Avoid self-interaction
        return 0
    
    # Check parameter validity
    if A < 0 or rho < 1e-6:
        warnings.warn(f"Invalid Buckingham parameters: A={A}, rho={rho}")
        return 0
    
    return A * exp(-r / rho) - C / (r ** 6)

def electrostatic_energy(q1: float, q2: float, r: float, 
                         epsilon_r: float = 10.0) -> float:
    """
    Calculate electrostatic potential energy (Coulomb interaction)
    
    :param q1: Charge of first atom (e)
    :param q2: Charge of second atom (e)
    :param r: Interatomic distance (Å)
    :param epsilon_r: Relative permittivity (dimensionless)
    :return: Potential energy value (eV)
    """
    if r < 1e-6:  # Avoid self-interaction
        return 0
    
    # Handle zero charge cases
    if q1 == 0 or q2 == 0:
        return 0
    
    return KE_EV_ANG * q1 * q2 / (epsilon_r * r)

def tersoff_bond_order(r: float, params: dict) -> float:
    """
    Tersoff bond-order potential energy function
    
    :param r: Interatomic distance (Å)
    :param params: Tersoff parameter dictionary, should contain:
        - r0: Equilibrium bond length (Å)
        - A: Repulsion parameter (eV)
        - B: Attraction parameter (eV)
        - lambda1: Repulsion decay coefficient (Å⁻¹)
        - lambda2: Attraction decay coefficient (Å⁻¹)
        - beta: Bond-order parameter
        - n: Bond-order exponent parameter
        - c: Angle term parameter
        - d: Angle term parameter
        - h: Angle term parameter
        - R: Lower cutoff distance (Å)
        - S: Upper cutoff distance (Å)
    :return: Potential energy value (eV)
    """
    # Check required parameters
    required = ['r0', 'A', 'B', 'lambda1', 'lambda2', 'beta', 'n', 'c', 'd', 'h', 'R', 'S']
    if not all(key in params for key in required):
        missing = [key for key in required if key not in params]
        warnings.warn(f"Missing Tersoff parameters: {', '.join(missing)}")
        return 0
    
    r0 = params['r0']
    A = params['A']
    B = params['B']
    lambda1 = params['lambda1']
    lambda2 = params['lambda2']
    beta = params['beta']
    n = params['n']
    c = params['c']
    d = params['d']
    h = params['h']
    R = params['R']
    S = params['S']
    
    # Cutoff function
    if r < R:
        f_c = 1.0
    elif r < S:
        f_c = 0.5 * (1 + np.cos(np.pi * (r - R) / (S - R)))
    else:
        f_c = 0.0
    
    # Repulsion term
    f_R = A * exp(-lambda1 * r)
    
    # Attraction term
    f_A = B * exp(-lambda2 * r)
    
    # Bond-order function (simplified to 1, full implementation requires local environment)
    b_ij = 1.0  # Simplified version ignores angular dependence
    
    # Combine energy terms
    return f_c * (f_R - b_ij * f_A)

def morse_potential(r: float, D: float, alpha: float, r0: float) -> float:
    """
    Morse potential energy function
    
    :param r: Interatomic distance (Å)
    :param D: Dissociation energy (eV)
    :param alpha: Potential well width parameter (Å⁻¹)
    :param r0: Equilibrium bond length (Å)
    :return: Potential energy value (eV)
    """
    if r < 1e-6:
        return 0
    
    exp_term = exp(-alpha * (r - r0))
    return D * (exp_term ** 2 - 2 * exp_term)

def harmonic_bond_energy(r: float, r0: float, k: float) -> float:
    """
    Harmonic oscillator bond potential energy
    
    :param r: Interatomic distance (Å)
    :param r0: Equilibrium bond length (Å)
    :param k: Force constant (eV/Å²)
    :return: Potential energy value (eV)
    """
    return 0.5 * k * (r - r0) ** 2

def lennard_jones_energy(r: float, epsilon: float, sigma: float) -> float:
    """
    Lennard-Jones potential energy
    
    :param r: Interatomic distance (Å)
    :param epsilon: Potential well depth (eV)
    :param sigma: Zero-energy distance (Å)
    :return: Potential energy value (eV)
    """
    if r < 1e-6:
        return 0
    
    sig_r = sigma / r
    sig_r6 = sig_r ** 6
    sig_r12 = sig_r6 ** 2
    return 4 * epsilon * (sig_r12 - sig_r6)

def calculate_pair_energy(atom1_params: dict, atom2_params: dict, 
                         pair_params: dict, r: float) -> float:
    """
    Calculate potential energy between a pair of atoms (intelligently selects potential model)
    
    :param atom1_params: Parameters for atom 1
    :param atom2_params: Parameters for atom 2
    :param pair_params: Atom pair parameters
    :param r: Interatomic distance (Å)
    :return: Potential energy value (eV)
    """
    energy = 0.0
    
    # 1. Always handle electrostatic energy if present
    if 'electrostatic' in pair_params:
        elec_params = pair_params['electrostatic']
        q1 = atom1_params.get('charge', 0)
        q2 = atom2_params.get('charge', 0)
        epsilon_r = elec_params.get('epsilon_r', 10.0)
        energy += electrostatic_energy(q1, q2, r, epsilon_r)
    
    # 2. Intelligently select one bonding potential model (priority: bond_order > Buckingham > others)
    if 'bond_order' in pair_params:
        # Prioritize bond order potential
        bond_params = pair_params['bond_order']
        model = bond_params.get('model', 'morse').lower()
        
        if model == 'morse':
            D = bond_params.get('D')
            alpha = bond_params.get('alpha')
            r0 = bond_params.get('r0')
            if D is not None and alpha is not None and r0 is not None:
                energy += morse_potential(r, D, alpha, r0)
        
        elif model == 'tersoff':
            energy += tersoff_bond_order(r, bond_params)
    
    elif 'buckingham' in pair_params:
        # Next use Buckingham potential
        buck_params = pair_params['buckingham']
        A = buck_params.get('A', 0)
        rho = buck_params.get('rho', 0)
        C = buck_params.get('C', 0)
        energy += buckingham_energy(r, A, rho, C)
    
    elif 'harmonic' in pair_params:
        # Then use harmonic potential
        harm_params = pair_params['harmonic']
        r0 = harm_params.get('r0')
        k = harm_params.get('k')
        if r0 is not None and k is not None:
            energy += harmonic_bond_energy(r, r0, k)
    
    elif 'lennard_jones' in pair_params:
        # Finally use Lennard-Jones potential
        lj_params = pair_params['lennard_jones']
        epsilon = lj_params.get('epsilon')
        sigma = lj_params.get('sigma')
        if epsilon is not None and sigma is not None:
            energy += lennard_jones_energy(r, epsilon, sigma)
    
    return energy

def validate_potential_parameters(pair_params: dict):
    """
    Validate potential energy parameters
    
    :param pair_params: Atom pair parameters
    :return: Whether valid
    """
    valid = True
    
    if 'buckingham' in pair_params:
        buck = pair_params['buckingham']
        if 'A' not in buck or 'rho' not in buck:
            warnings.warn("Buckingham potential missing required parameters A or rho")
            valid = False
    
    if 'electrostatic' in pair_params:
        elec = pair_params['electrostatic']
        if 'epsilon_r' not in elec:
            warnings.warn("Electrostatic potential missing permittivity epsilon_r, using default 10.0")
    
    if 'bond_order' in pair_params:
        bond = pair_params['bond_order']
        model = bond.get('model', 'morse').lower()
        
        if model == 'morse':
            if 'D' not in bond or 'alpha' not in bond or 'r0' not in bond:
                warnings.warn("Morse potential missing required parameters D, alpha, or r0")
                valid = False
        
        elif model == 'tersoff':
            required = ['r0', 'A', 'B', 'lambda1', 'lambda2', 'beta', 'n', 'c', 'd', 'h', 'R', 'S']
            if not all(key in bond for key in required):
                missing = [key for key in required if key not in bond]
                warnings.warn(f"Tersoff potential missing parameters: {', '.join(missing)}")
                valid = False
    
    return valid

if __name__ == "__main__":
    # Module self-test
    print("Testing potentials module...")
    
    # Test Buckingham energy
    buck_energy = buckingham_energy(2.0, 1000.0, 0.2, 10.0)
    print(f"Buckingham energy at 2.0 Å: {buck_energy:.4f} eV")
    
    # Test electrostatic energy
    elec_energy = electrostatic_energy(1.0, -1.0, 2.0)
    print(f"Electrostatic energy at 2.0 Å: {elec_energy:.4f} eV")
    
    # Test Morse energy
    morse_energy = morse_potential(1.95, 5.0, 2.5, 1.95)
    print(f"Morse energy at 1.95 Å: {morse_energy:.4f} eV")
    
    # Test combined energy
    atom1 = {'charge': 1.5}
    atom2 = {'charge': -1.5}
    pair_params = {
        'buckingham': {'A': 4500.0, 'rho': 0.25, 'C': 0.0},
        'electrostatic': {'epsilon_r': 10.0},
        'bond_order': {
            'model': 'morse',
            'D': 5.0,
            'alpha': 2.5,
            'r0': 1.95
        }
    }
    combined_energy = calculate_pair_energy(atom1, atom2, pair_params, 1.95)
    print(f"Combined energy at 1.95 Å: {combined_energy:.4f} eV")
    
    print("Potentials module tests complete.")