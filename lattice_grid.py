"""
lattice_grid.py - Build basic coordinates sturctures for 7 different kind of crystal systems.

Author: Jian Sun.
Recent Updated: Jul/13/2025
"""

import numpy as np
from typing import Tuple, List, Optional
import itertools

class LatticeGrid:
    """General lattice grid system supporting all crystal systems"""
    def __init__(self, 
                 lattice_params: Tuple[float, float, float],
                 angles: Tuple[float, float, float],
                 grid_divisions: Tuple[int, int, int]):
        """
        Initialize lattice grid
        
        :param lattice_params: Lattice constants (a, b, c) (Å)
        :param angles: Lattice angles (α, β, γ) (degrees)
        :param grid_divisions: Number of grid divisions in each direction (nx, ny, nz)
        """
        self.a, self.b, self.c = lattice_params
        self.alpha, self.beta, self.gamma = np.radians(angles)
        self.grid_divisions = grid_divisions
        self.metric_tensor = self._compute_metric_tensor()
        self.sites = self._generate_sites()
    
    def _compute_metric_tensor(self) -> np.ndarray:
        """Compute the metric tensor of the lattice"""
        a, b, c = self.a, self.b, self.c
        alpha, beta, gamma = self.alpha, self.beta, self.gamma
        
        # Calculate metric tensor components
        g11 = a**2
        g22 = b**2
        g33 = c**2
        g12 = a * b * np.cos(gamma)
        g13 = a * c * np.cos(beta)
        g23 = b * c * np.cos(alpha)
        
        return np.array([
            [g11, g12, g13],
            [g12, g22, g23],
            [g13, g23, g33]
        ])
    
    def _generate_sites(self) -> List[Tuple[float, float, float]]:
        """Generate grid point coordinates (fractional coordinates)"""
        sites = []
        nx, ny, nz = self.grid_divisions
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    u = i / nx
                    v = j / ny
                    w = k / nz
                    sites.append((u, v, w))
        return sites
    
    def fractional_to_cartesian(self, frac_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Convert fractional coordinates to Cartesian coordinates
        
        :param frac_pos: Fractional coordinates (u, v, w)
        :return: Cartesian coordinates (x, y, z) (Å)
        """
        u, v, w = frac_pos
        
        # Calculate basis vectors based on crystal system
        a = self.a
        b = self.b
        c = self.c
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        
        # Calculate basis vectors
        ax = a
        ay = 0.0
        az = 0.0
        
        bx = b * np.cos(gamma)
        by = b * np.sin(gamma)
        bz = 0.0
        
        cx = c * np.cos(beta)
        cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        cz = c * np.sqrt(1 - np.cos(beta)**2 - cy**2/c**2)
        
        # Calculate Cartesian coordinates
        x = u * ax + v * bx + w * cx
        y = u * ay + v * by + w * cy
        z = u * az + v * bz + w * cz
        
        return (x, y, z)
    
    def cartesian_to_fractional(self, cart_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Convert Cartesian coordinates to fractional coordinates
        
        :param cart_pos: Cartesian coordinates (x, y, z)
        :return: Fractional coordinates (u, v, w)
        """
        x, y, z = cart_pos
        
        # Calculate reciprocal lattice vectors
        volume = self._cell_volume()
        
        # Calculate reciprocal basis vectors
        a1 = np.array([self.a, 0, 0])
        a2 = np.array([self.b * np.cos(self.gamma), self.b * np.sin(self.gamma), 0])
        a3 = np.array([
            self.c * np.cos(self.beta),
            self.c * (np.cos(self.alpha) - np.cos(self.beta) * np.cos(self.gamma)) / np.sin(self.gamma),
            volume / (self.a * self.b * np.sin(self.gamma))
        ])
        
        # Calculate fractional coordinates
        b1 = 2 * np.pi * np.cross(a2, a3) / volume
        b2 = 2 * np.pi * np.cross(a3, a1) / volume
        b3 = 2 * np.pi * np.cross(a1, a2) / volume
        
        # Create position vector
        r = np.array([x, y, z])
        
        # Calculate fractional coordinates
        u = np.dot(r, b1) / (2 * np.pi)
        v = np.dot(r, b2) / (2 * np.pi)
        w = np.dot(r, b3) / (2 * np.pi)
        
        return (u, v, w)
    
    def periodic_distance(self, 
                         frac_pos1: Tuple[float, float, float], 
                         frac_pos2: Tuple[float, float, float]) -> float:
        """
        Calculate minimum distance between two points considering periodic boundary conditions
        
        :param frac_pos1: Fractional coordinates of first point
        :param frac_pos2: Fractional coordinates of second point
        :return: Minimum image distance (Å)
        """
        u1, v1, w1 = frac_pos1
        u2, v2, w2 = frac_pos2
        
        # Calculate fractional coordinate differences
        du = u1 - u2
        dv = v1 - v2
        dw = w1 - w2
        
        # Apply periodic boundary conditions (nearest image)
        du = du - round(du)
        dv = dv - round(dv)
        dw = dw - round(dw)
        
        # Convert to Cartesian coordinate differences
        dr = np.array([du, dv, dw])
        
        # Calculate distance using metric tensor
        distance_sq = np.dot(dr, np.dot(self.metric_tensor, dr))
        
        return np.sqrt(distance_sq)
    
    def _cell_volume(self) -> float:
        """Calculate unit cell volume"""
        a, b, c = self.a, self.b, self.c
        alpha, beta, gamma = self.alpha, self.beta, self.gamma
        
        return a * b * c * np.sqrt(
            1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 
            + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        )
    
    @property
    def num_sites(self) -> int:
        """Return total number of grid points"""
        nx, ny, nz = self.grid_divisions
        return nx * ny * nz
    
    def get_cartesian_sites(self) -> List[Tuple[float, float, float]]:
        """Get Cartesian coordinates of all grid points"""
        return [self.fractional_to_cartesian(frac) for frac in self.sites]


class CrystalSystemFactory:
    """Crystal system factory class for simplified creation of common crystal systems"""
    @staticmethod
    def cubic(a: float, divisions: Tuple[int, int, int]) -> LatticeGrid:
        """Create cubic crystal system"""
        return LatticeGrid(
            lattice_params=(a, a, a),
            angles=(90.0, 90.0, 90.0),
            grid_divisions=divisions
        )
    
    @staticmethod
    def hexagonal(a: float, c: float, divisions: Tuple[int, int, int]) -> LatticeGrid:
        """Create hexagonal crystal system"""
        return LatticeGrid(
            lattice_params=(a, a, c),
            angles=(90.0, 90.0, 120.0),
            grid_divisions=divisions
        )
    
    @staticmethod
    def tetragonal(a: float, c: float, divisions: Tuple[int, int, int]) -> LatticeGrid:
        """Create tetragonal crystal system"""
        return LatticeGrid(
            lattice_params=(a, a, c),
            angles=(90.0, 90.0, 90.0),
            grid_divisions=divisions
        )
    
    @staticmethod
    def orthorhombic(a: float, b: float, c: float, divisions: Tuple[int, int, int]) -> LatticeGrid:
        """Create orthorhombic crystal system"""
        return LatticeGrid(
            lattice_params=(a, b, c),
            angles=(90.0, 90.0, 90.0),
            grid_divisions=divisions
        )
    
    @staticmethod
    def rhombohedral(a: float, alpha: float, divisions: Tuple[int, int, int]) -> LatticeGrid:
        """Create rhombohedral crystal system"""
        return LatticeGrid(
            lattice_params=(a, a, a),
            angles=(alpha, alpha, alpha),
            grid_divisions=divisions
        )
    
    @staticmethod
    def monoclinic(a: float, b: float, c: float, beta: float, divisions: Tuple[int, int, int]) -> LatticeGrid:
        """Create monoclinic crystal system"""
        return LatticeGrid(
            lattice_params=(a, b, c),
            angles=(90.0, beta, 90.0),
            grid_divisions=divisions
        )
    
    @staticmethod
    def triclinic(a: float, b: float, c: float, 
                 alpha: float, beta: float, gamma: float, 
                 divisions: Tuple[int, int, int]) -> LatticeGrid:
        """Create triclinic crystal system"""
        return LatticeGrid(
            lattice_params=(a, b, c),
            angles=(alpha, beta, gamma),
            grid_divisions=divisions
        )