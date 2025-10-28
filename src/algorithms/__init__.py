"""
Algoritmos de optimización para control de semáforos.

Este módulo contiene las implementaciones de los cuatro algoritmos principales:
- Programación Dinámica: Optimización de un semáforo individual
- Branch & Bound: Coordinación entre semáforos (ondas verdes)
- Algoritmos Genéticos: Optimización global de toda la red
- Max-Pressure: Heurística baseline en tiempo real
"""

from .max_pressure import MaxPressureController
from .dynamic_programming import DynamicProgrammingOptimizer
from .branch_and_bound import BranchAndBoundCoordinator, BranchAndBoundNode
from .genetic_algorithm import GeneticOptimizer, Individual

__all__ = [
    'MaxPressureController',
    'DynamicProgrammingOptimizer',
    'BranchAndBoundCoordinator',
    'BranchAndBoundNode',
    'GeneticOptimizer',
    'Individual'
]
