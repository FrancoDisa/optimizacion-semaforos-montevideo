"""
Tests para el módulo de vehículos (Vehicle).
"""

import pytest
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulator.vehicle import Vehicle, VehicleState
from src.simulator.traffic_network import RoadSegment


class TestVehicle:
    """Tests para la clase Vehicle."""

    def test_vehicle_creation(self):
        """Test de creación básica de vehículo."""
        route = [1, 2, 3, 4, 5]
        vehicle = Vehicle(
            origin=1,
            destination=5,
            spawn_time=0.0,
            route=route
        )

        assert vehicle.origin == 1
        assert vehicle.destination == 5
        assert vehicle.route == route
        assert vehicle.spawn_time == 0.0
        assert vehicle.state == VehicleState.MOVING

    def test_unique_ids(self):
        """Test de IDs únicos para vehículos."""
        v1 = Vehicle(1, 2, 0.0, [1, 2])
        v2 = Vehicle(1, 2, 0.0, [1, 2])

        assert v1.id != v2.id

    def test_direction_calculation(self):
        """Test de cálculo de dirección."""
        route = [1, 2, 3]
        vehicle = Vehicle(1, 3, 0.0, route)

        # De 1 a 2 (aumenta) = oeste
        direction = vehicle.get_direction()
        assert direction in ["east", "west", "north", "south"]

    def test_vehicle_movement(self):
        """Test de movimiento básico."""
        route = [1, 2]
        vehicle = Vehicle(1, 2, 0.0, route, max_speed_ms=10.0)

        # Asignar segmento
        segment = RoadSegment(1, 2, 100, 10, 2)
        vehicle.current_segment = segment

        initial_pos = vehicle.position_on_edge

        # Actualizar durante 5 segundos
        for t in range(5):
            vehicle.update(dt=1.0, current_time=t)

        # Debería haberse movido
        assert vehicle.position_on_edge > initial_pos
        assert vehicle.distance_traveled > 0

    def test_stopping_distance(self):
        """Test de cálculo de distancia de frenado."""
        vehicle = Vehicle(1, 2, 0.0, [1, 2])

        # A velocidad 0, distancia de frenado = 0
        vehicle.current_speed = 0.0
        assert vehicle._calculate_stopping_distance() == 0.0

        # A velocidad mayor, distancia > 0
        vehicle.current_speed = 10.0  # m/s
        stopping_dist = vehicle._calculate_stopping_distance()
        assert stopping_dist > 0

    def test_acceleration_and_braking(self):
        """Test de aceleración y frenado."""
        vehicle = Vehicle(1, 2, 0.0, [1, 2], max_speed_ms=10.0)

        # Inicialmente detenido
        assert vehicle.current_speed == 0.0

        # Ajustar hacia velocidad objetivo
        vehicle._adjust_speed(target_speed=10.0, dt=1.0)

        # Debería haber acelerado
        assert vehicle.current_speed > 0
        assert vehicle.state == VehicleState.ACCELERATING

        # Ahora frenar
        vehicle._adjust_speed(target_speed=0.0, dt=1.0)

        # Debería estar frenando
        assert vehicle.state == VehicleState.BRAKING

    def test_segment_transition(self):
        """Test de transición entre segmentos."""
        route = [1, 2, 3]
        vehicle = Vehicle(1, 3, 0.0, route)

        # Segmento inicial
        segment1 = RoadSegment(1, 2, 50, 5, 2)
        vehicle.current_segment = segment1

        assert vehicle.current_edge_index == 0

        # Mover más allá del final del segmento
        vehicle.position_on_edge = 60  # Más de 50m
        vehicle._move_to_next_segment()

        # Debería estar en siguiente segmento
        assert vehicle.current_edge_index == 1
        assert vehicle.position_on_edge == 0.0

    def test_arrival_detection(self):
        """Test de detección de llegada."""
        route = [1, 2]
        vehicle = Vehicle(1, 2, 0.0, route)

        assert not vehicle.has_arrived()

        # Avanzar al último segmento y más allá
        vehicle.current_edge_index = len(route) - 1
        vehicle._move_to_next_segment()

        assert vehicle.has_arrived()
        assert vehicle.state == VehicleState.ARRIVED

    def test_statistics_collection(self):
        """Test de recolección de estadísticas."""
        route = [1, 2]
        vehicle = Vehicle(1, 2, 0.0, route)

        # Simular algunos pasos detenido
        vehicle.current_speed = 0.0
        vehicle.state = VehicleState.STOPPED_AT_LIGHT

        for t in range(10):
            vehicle._update_statistics(dt=1.0, current_time=t)

        # Debería haber acumulado tiempo de espera
        assert vehicle.total_waiting_time > 0
        assert vehicle.num_stops > 0

    def test_travel_time_calculation(self):
        """Test de cálculo de tiempo de viaje."""
        vehicle = Vehicle(1, 2, spawn_time=10.0, route=[1, 2])

        # Sin llegar, tiempo = current_time - spawn_time
        travel_time = vehicle.get_travel_time(current_time=25.0)
        assert travel_time == 15.0

        # Con llegada
        vehicle.arrival_time = 30.0
        vehicle.state = VehicleState.ARRIVED
        travel_time = vehicle.get_travel_time(current_time=35.0)
        assert travel_time == 20.0  # 30 - 10

    def test_delay_calculation(self):
        """Test de cálculo de retraso."""
        vehicle = Vehicle(1, 2, spawn_time=0.0, route=[1, 2])
        vehicle.arrival_time = 50.0
        vehicle.state = VehicleState.ARRIVED

        # Tiempo ideal = 30s, real = 50s, retraso = 20s
        delay = vehicle.get_delay(ideal_travel_time=30.0)
        assert delay == 20.0

    def test_statistics_dict(self):
        """Test de diccionario de estadísticas."""
        vehicle = Vehicle(1, 2, 0.0, [1, 2])

        stats = vehicle.get_statistics()

        assert 'vehicle_id' in stats
        assert 'origin' in stats
        assert 'destination' in stats
        assert 'travel_time' in stats
        assert 'num_stops' in stats
        assert stats['origin'] == 1
        assert stats['destination'] == 2

    def test_average_speed(self):
        """Test de velocidad promedio."""
        vehicle = Vehicle(1, 2, 0.0, [1, 2])

        # Sin movimiento, velocidad = 0
        assert vehicle.get_average_speed_kmh() == 0.0

        # Simular viaje
        vehicle.distance_traveled = 1000  # 1 km
        vehicle.arrival_time = 100.0  # 100 segundos
        vehicle.state = VehicleState.ARRIVED

        avg_speed = vehicle.get_average_speed_kmh()

        # 1000m en 100s = 10 m/s = 36 km/h
        assert abs(avg_speed - 36.0) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
