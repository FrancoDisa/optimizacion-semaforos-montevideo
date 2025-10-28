"""
Tests para el módulo de red vial (TrafficNetwork).
"""

import pytest
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulator.traffic_network import TrafficNetwork, Intersection, RoadSegment
from src.utils.config import INTERSECTIONS_FILE


class TestIntersection:
    """Tests para la clase Intersection."""

    def test_intersection_creation(self):
        """Test de creación básica de intersección."""
        intersection = Intersection(
            intersection_id=1,
            name="Av. Brasil y Rambla",
            coordinates=(-34.9089, -56.1536)
        )

        assert intersection.id == 1
        assert intersection.name == "Av. Brasil y Rambla"
        assert intersection.lat == -34.9089
        assert intersection.lon == -56.1536

    def test_queue_operations(self):
        """Test de operaciones de cola."""
        intersection = Intersection(1, "Test", (0, 0))

        # Inicialmente vacías
        assert intersection.get_queue_length("north_south") == 0
        assert intersection.get_total_queue_length() == 0

        # Agregar vehículos ficticios
        intersection.add_vehicle_to_queue("vehicle1", "north_south")
        intersection.add_vehicle_to_queue("vehicle2", "north_south")

        assert intersection.get_queue_length("north_south") == 2
        assert intersection.get_total_queue_length() == 2

        # Remover vehículo
        intersection.remove_vehicle_from_queue("vehicle1", "north_south")
        assert intersection.get_queue_length("north_south") == 1

        # Limpiar colas
        intersection.clear_queues()
        assert intersection.get_total_queue_length() == 0


class TestRoadSegment:
    """Tests para la clase RoadSegment."""

    def test_segment_creation(self):
        """Test de creación de segmento."""
        segment = RoadSegment(
            from_id=1,
            to_id=2,
            length_m=280,
            travel_time_s=22,
            lanes=2
        )

        assert segment.from_id == 1
        assert segment.to_id == 2
        assert segment.length_m == 280
        assert segment.lanes == 2

    def test_speed_calculation(self):
        """Test de cálculo de velocidad."""
        segment = RoadSegment(1, 2, 280, 22, 2)

        # Velocidad = 280m / 22s = ~12.7 m/s
        assert abs(segment.avg_speed_ms - 12.7) < 0.1

        # ~45.8 km/h
        assert abs(segment.avg_speed_kmh - 45.8) < 1.0

    def test_capacity_calculation(self):
        """Test de cálculo de capacidad."""
        segment = RoadSegment(1, 2, 280, 22, lanes=2)

        # 280m / 7m por vehículo * 2 carriles = 80 vehículos
        assert segment.get_capacity() == 80

    def test_occupancy(self):
        """Test de ocupación del segmento."""
        segment = RoadSegment(1, 2, 280, 22, lanes=2)

        assert not segment.is_full()
        assert segment.get_occupancy_rate() == 0.0

        # Agregar vehículos hasta llenar
        for i in range(segment.get_capacity()):
            segment.add_vehicle(f"vehicle{i}")

        assert segment.is_full()
        assert segment.get_occupancy_rate() == 1.0


class TestTrafficNetwork:
    """Tests para la clase TrafficNetwork."""

    def test_empty_network_creation(self):
        """Test de creación de red vacía."""
        network = TrafficNetwork()

        assert len(network.intersections) == 0
        assert len(network.segments) == 0

    def test_network_from_file(self):
        """Test de carga de red desde archivo."""
        network = TrafficNetwork(str(INTERSECTIONS_FILE))

        # Verificar que cargó las 5 intersecciones
        assert len(network.intersections) == 5

        # Verificar nombres
        assert 1 in network.intersections
        assert "Rambla" in network.intersections[1].name

    def test_add_intersection(self):
        """Test de agregar intersección."""
        network = TrafficNetwork()

        network.add_intersection(
            intersection_id=1,
            name="Test Intersection",
            coordinates=(-34.9, -56.15)
        )

        assert 1 in network.intersections
        assert network.get_intersection(1).name == "Test Intersection"

    def test_add_segment(self):
        """Test de agregar segmento."""
        network = TrafficNetwork()

        # Agregar dos intersecciones
        network.add_intersection(1, "A", (0, 0))
        network.add_intersection(2, "B", (0, 1))

        # Agregar segmento
        network.add_segment(1, 2, length_m=100, travel_time_s=10)

        assert (1, 2) in network.segments
        segment = network.get_segment(1, 2)
        assert segment.length_m == 100

    def test_shortest_path(self):
        """Test de cálculo de ruta más corta."""
        network = TrafficNetwork(str(INTERSECTIONS_FILE))

        # Ruta de intersección 1 a 5
        path = network.get_shortest_path(1, 5)

        assert path is not None
        assert path[0] == 1
        assert path[-1] == 5
        assert len(path) >= 2

    def test_path_metrics(self):
        """Test de métricas de ruta."""
        network = TrafficNetwork(str(INTERSECTIONS_FILE))

        path = network.get_shortest_path(1, 5)
        assert path is not None

        # Longitud de ruta
        length = network.get_path_length(path)
        assert length > 0

        # Tiempo de viaje
        travel_time = network.get_path_travel_time(path)
        assert travel_time > 0

    def test_network_stats(self):
        """Test de estadísticas de la red."""
        network = TrafficNetwork(str(INTERSECTIONS_FILE))

        stats = network.get_network_stats()

        assert stats['num_intersections'] == 5
        assert stats['num_segments'] > 0
        assert stats['total_length_km'] > 0
        assert 'network_name' in stats

    def test_neighbors(self):
        """Test de vecinos de una intersección."""
        network = TrafficNetwork(str(INTERSECTIONS_FILE))

        # La intersección 1 debería tener vecinos
        neighbors = network.get_neighbors(1)
        assert len(neighbors) > 0

        # Vecinos entrantes
        incoming = network.get_incoming_neighbors(2)
        assert len(incoming) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
