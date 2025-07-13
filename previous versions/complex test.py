import numpy as np
import pulp
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import itertools
import math
import csv

@dataclass
class AtomType:
    symbol: str
    charge: float
    A: float
    rho: float
    C: float
    shannon_radius: float

@dataclass
class GridSite:
    position: np.ndarray
    site_id: int

class CubicGrid:
    def __init__(self, lattice_param: float, grid_divisions: int):
        self.a = lattice_param
        self.divisions = grid_divisions
        self.spacing = lattice_param / grid_divisions
        self.sites = self._generate_sites()
        
    def _generate_sites(self) -> List[GridSite]:
        """Generate discrete grid sites"""
        sites = []
        idx = 0
        for i, j, k in itertools.product(range(self.divisions), repeat=3):
            pos = np.array([i, j, k]) * self.spacing
            sites.append(GridSite(pos, idx))
            idx += 1
        return sites
    
    def periodic_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate minimum image distance with periodic boundary conditions"""
        delta = pos1 - pos2
        for dim in range(3):
            if delta[dim] > self.a / 2:
                delta[dim] -= self.a
            elif delta[dim] < -self.a / 2:
                delta[dim] += self.a
        return np.linalg.norm(delta)

def buckingham_energy(r: float, A: float, rho: float, C: float) -> float:
    """Calculate Buckingham potential energy"""
    if r < 1e-6:
        return 0
    return A * np.exp(-r / rho) - C / r**6

KE_EV_ANG = 14.3996  # eV·Å / e²
eps_r = 10.0          # Relative permittivity

def ewald_energy(q1: float, q2: float, r: float) -> float:
    """Simplified electrostatic energy"""
    if r < 1e-6:
        return 0
    return KE_EV_ANG * q1 * q2 / (eps_r * r)

def compute_interaction_matrix(
        grid: CubicGrid,
        atom_types: Dict[str, AtomType],
        pair_params: Dict[Tuple[str, str], Tuple[float, float, float]],
        r_cutoff: float = 6.0
) -> Dict[Tuple[int, str, int, str], float]:
    """Compute pairwise interaction energies"""
    interaction = {}
    n_sites = len(grid.sites)
    positions = [site.position for site in grid.sites]
    
    for i, j in itertools.combinations(range(n_sites), 2):
        r = grid.periodic_distance(positions[i], positions[j])
        if r > r_cutoff:
            continue
            
        for e1, e2 in itertools.product(atom_types.keys(), repeat=2):
            # Get potential parameters
            if (e1, e2) in pair_params:
                A, rho, C = pair_params[(e1, e2)]
            elif (e2, e1) in pair_params:
                A, rho, C = pair_params[(e2, e1)]
            else:
                at1, at2 = atom_types[e1], atom_types[e2]
                A   = (at1.A + at2.A) / 2
                rho = (at1.rho + at2.rho) / 2
                C   = (at1.C + at2.C) / 2

            # Calculate energies
            e_buck  = buckingham_energy(r, A, rho, C)
            q1, q2  = atom_types[e1].charge, atom_types[e2].charge
            e_ewald = ewald_energy(q1, q2, r)
            
            # Store interaction
            interaction[(i, e1, j, e2)] = e_buck + e_ewald
                
    return interaction

def add_proximity_constraints(
        prob: pulp.LpProblem,
        grid: CubicGrid,
        x: Dict[Tuple[int, str], pulp.LpVariable],
        atom_types: Dict[str, AtomType],
        min_distance_factor: float = 0.7
):
    """Prevent atom overlap"""
    positions = [site.position for site in grid.sites]
    n_sites = len(positions)
    
    for i, j in itertools.combinations(range(n_sites), 2):
        r = grid.periodic_distance(positions[i], positions[j])
        for e1 in atom_types:
            r1 = atom_types[e1].shannon_radius
            for e2 in atom_types:
                r2 = atom_types[e2].shannon_radius
                min_dist = min_distance_factor * (r1 + r2)
                if r < min_dist:
                    prob += x[(i, e1)] + x[(j, e2)] <= 1

def add_coordination_constraints(
        prob: pulp.LpProblem,
        grid: CubicGrid,
        x: Dict[Tuple[int, str], pulp.LpVariable],
        atom_types: Dict[str, AtomType],
        coordination_rules: Dict[str, Tuple[str, int, float, float]]
):
    """
    Add coordination constraints for specific atom types
    :param coordination_rules: Dict of {atom_type: (neighbor_type, coordination_number, bond_length, tolerance)}
    """
    positions = [site.position for site in grid.sites]
    n_sites = len(positions)
    
    for atom_type, (neighbor_type, coord_num, bond_len, tolerance) in coordination_rules.items():
        for i in range(n_sites):
            # Create indicator variable for current atom type
            is_atom = pulp.LpVariable(f"is_{atom_type}_{i}", cat='Binary')
            prob += is_atom == x[(i, atom_type)]
            
            # Count neighbors
            neighbor_count = pulp.LpAffineExpression()
            for j in range(n_sites):
                if i == j:
                    continue
                    
                r = grid.periodic_distance(positions[i], positions[j])
                if abs(r - bond_len) < tolerance:
                    neighbor_count += x[(j, neighbor_type)]
            
            # Apply coordination constraint
            prob += neighbor_count >= coord_num * is_atom - 0.5
            prob += neighbor_count <= coord_num * is_atom + 0.5

def solve_global_ip(
        grid: CubicGrid, 
        atom_types: Dict[str, AtomType], 
        composition: Dict[str, int],
        pair_params: Dict[Tuple[str, str], Tuple[float, float, float]],
        coordination_rules: Optional[Dict[str, Tuple[str, int, float, float]]] = None,
        r_cutoff: float = 4.0,
        time_limit: int = 600
) -> List[Tuple[str, np.ndarray]]:
    """
    Solve global optimization problem for crystal structure
    :param grid: CubicGrid instance
    :param atom_types: Dictionary of atom types
    :param composition: Dictionary of element counts
    :param pair_params: Buckingham potential parameters for atom pairs
    :param coordination_rules: Coordination constraints (optional)
    :param r_cutoff: Cutoff distance for interactions
    :param time_limit: Solver time limit in seconds
    :return: List of (element, position) tuples
    """
    prob = pulp.LpProblem("GaN_Structure_Optimization", pulp.LpMinimize)
    n_sites = len(grid.sites)
    
    # Create decision variables
    x = pulp.LpVariable.dicts("x", 
        ((i, e) for i in range(n_sites) for e in atom_types),
        cat='Binary'
    )
    
    # Basic constraints
    for i in range(n_sites):
        prob += pulp.lpSum(x[(i, e)] for e in atom_types) <= 1

    for e, count in composition.items():
        prob += pulp.lpSum(x[(i, e)] for i in range(n_sites)) == count

    # Compute interaction matrix
    print("Computing interaction matrix...")
    interaction = compute_interaction_matrix(grid, atom_types, pair_params, r_cutoff)
    
    # Add proximity constraints
    print("Adding proximity constraints...")
    add_proximity_constraints(prob, grid, x, atom_types, min_distance_factor=0.75)
    
    # Add coordination constraints if provided
    if coordination_rules:
        print("Adding coordination constraints...")
        add_coordination_constraints(prob, grid, x, atom_types, coordination_rules)
    
    # Linearize objective function
    print("Linearizing objective function...")
    z = pulp.LpVariable.dicts("z", 
        ((i, e1, j, e2) for (i, e1, j, e2) in interaction if i < j),
        cat='Binary'
    )
    
    for (i, e1, j, e2) in z:
        prob += z[(i, e1, j, e2)] <= x[(i, e1)]
        prob += z[(i, e1, j, e2)] <= x[(j, e2)]
        prob += z[(i, e1, j, e2)] >= x[(i, e1)] + x[(j, e2)] - 1

    prob += pulp.lpSum(
        interaction[key] * z[key] 
        for key in z if key in interaction
    )

    # Configure solver
    print(f"Problem size: {n_sites} sites, {len(atom_types)} atom types, {len(z)} interactions")
    solver = pulp.GUROBI(
        msg=True,
        timeLimit=time_limit,
        mipFocus=1,       # Focus on finding feasible solutions
        presolve=2,        # Aggressive preprocessing
        heuristics=0.8,    # Higher heuristic intensity
        threads=4          # Use multiple cores
    )
    
    # Solve the problem
    print("Solving integer program...")
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    print(f"Solver status: {status}")
    
    # Extract results only if solution was found
    selected = []
    if status == "Optimal" or status == "Feasible":
        for (i, e) in x:
            var_value = pulp.value(x[(i, e)])
            if var_value is not None and var_value > 0.5:
                selected.append((e, grid.sites[i].position))
        print(f"Found solution with {len(selected)} atoms")
    else:
        print("No feasible solution found")
    
    return selected

def load_atom_parameters(csv_file: str) -> Dict[str, AtomType]:
    """Load atom parameters from CSV file"""
    atom_types = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            atom_types[row['symbol']] = AtomType(
                symbol=row['symbol'],
                charge=float(row['charge']),
                A=float(row['A']),
                rho=float(row['rho']),
                C=float(row['C']),
                shannon_radius=float(row['shannon_radius'])
            )
    return atom_types

def load_pair_parameters(csv_file: str) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
    """Load pair potential parameters from CSV file"""
    pair_params = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair = (row['element1'], row['element2'])
            pair_params[pair] = (
                float(row['A']), 
                float(row['rho']), 
                float(row['C']))
    return pair_params

def save_structure_to_xyz(positions: List[Tuple[str, np.ndarray]], filename: str):
    """Save predicted structure to XYZ file"""
    with open(filename, 'w') as f:
        f.write(f"{len(positions)}\n")
        f.write("Generated GaN structure\n")
        for element, pos in positions:
            f.write(f"{element} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

# ===== GaN PARAMETERS =====
# Lattice parameter (Å) - calculated from wurtzite volume
# Original wurtzite volume: a=3.19Å, c=5.19Å → V = a² * c * √3/2 ≈ 45.6 Å³
# Cubic equivalent: a_cubic = ∛V ≈ 3.57 Å
# For 2×2×2 supercell: a = 7.14 Å
LATTICE_PARAM = 7.14  # Å

# Grid divisions - optimized for wurtzite positions
# Divisions should be multiple of 3 for hexagonal symmetry
GRID_DIVISIONS = 12   # Results in grid spacing of 0.595 Å

# Atom composition
COMPOSITION = {'Ga': 16, 'N': 16}  # For 2×2×2 wurtzite supercell

# Coordination constraints for wurtzite structure
COORDINATION_RULES = {
    'Ga': ('N', 4, 1.95, 0.3),  # Ga should have 4 N neighbors at ~1.95 Å
    'N': ('Ga', 4, 1.95, 0.3)   # N should have 4 Ga neighbors
}

# Cutoff distance for interactions (Å)
R_CUTOFF = 5.0

# Solver time limit (seconds)
TIME_LIMIT = 12000  #

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("Initializing GaN structure optimization...")
    
    # Load parameters from CSV files
    try:
        atom_types = load_atom_parameters("atom_parameters.csv")
        pair_params = load_pair_parameters("pair_parameters.csv")
    except FileNotFoundError:
        print("Using default parameters")
        # Fallback to default parameters if CSV files not found
        atom_types = {
            'Ga': AtomType('Ga', 3.0, 3500.0, 0.27, 0.0, 0.62),
            'N':  AtomType('N', -3.0, 8000.0, 0.22, 30.0, 1.46)
        }
        pair_params = {
            ('Ga','N'): (4500.0, 0.25, 0.0),
            ('N','N') : (10000.0, 0.18, 35.0),
            ('Ga','Ga'): (3000.0, 0.30, 0.0)
        }
    
    # Create cubic grid
    print(f"Creating cubic grid: a={LATTICE_PARAM} Å, divisions={GRID_DIVISIONS}")
    grid = CubicGrid(LATTICE_PARAM, GRID_DIVISIONS)
    print(f"Generated {len(grid.sites)} grid sites")
    
    # Solve for optimal structure
    print("Solving for optimal GaN structure...")
    selected_atoms = solve_global_ip(
        grid=grid,
        atom_types=atom_types,
        composition=COMPOSITION,
        pair_params=pair_params,
        coordination_rules=COORDINATION_RULES,
        r_cutoff=R_CUTOFF,
        time_limit=TIME_LIMIT
    )
    
    # Save and analyze results
    if selected_atoms:
        print("\nPredicted atomic positions:")
        for i, (element, pos) in enumerate(selected_atoms[:5]):  # Print first 5 positions
            print(f"{i+1}. {element}: {np.round(pos, 4)}")
        print(f"... and {len(selected_atoms)-5} more positions")
        
        # Save to XYZ file
        save_structure_to_xyz(selected_atoms, "gan_structure.xyz")
        print("Structure saved to gan_structure.xyz")
        
        # Count atoms
        ga_count = sum(1 for e, _ in selected_atoms if e == 'Ga')
        n_count = sum(1 for e, _ in selected_atoms if e == 'N')
        print(f"\nGa atoms: {ga_count}, N atoms: {n_count}")
        
        # Calculate average bond length
        ga_positions = [pos for e, pos in selected_atoms if e == 'Ga']
        n_positions = [pos for e, pos in selected_atoms if e == 'N']
        bond_lengths = []
        
        for ga_pos in ga_positions:
            for n_pos in n_positions:
                dist = np.linalg.norm(ga_pos - n_pos)
                if 1.7 < dist < 2.2:  # Reasonable bond length range
                    bond_lengths.append(dist)
        
        if bond_lengths:
            avg_bond = np.mean(bond_lengths)
            print(f"Average Ga-N bond length: {avg_bond:.4f} Å")
            print(f"Expected bond length: ~1.95 Å")
    else:
        print("No solution found. Consider adjusting parameters or increasing time limit.")