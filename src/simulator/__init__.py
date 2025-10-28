"""
Simulador de tráfico vehicular.

Este módulo contiene el motor de simulación que modela:
- Red vial como grafo dirigido
- Movimiento de vehículos
- Estados de semáforos
- Generación de tráfico según patrones
"""

from .traffic_network import TrafficNetwork, Intersection, RoadSegment
from .traffic_light import TrafficLight, TrafficLightPhase, LightState
from .vehicle import Vehicle, VehicleState

__all__ = [
    'TrafficNetwork',
    'Intersection',
    'RoadSegment',
    'TrafficLight',
    'TrafficLightPhase',
    'LightState',
    'Vehicle',
    'VehicleState'
]
