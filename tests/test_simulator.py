"""
Tests para el simulador de tráfico.
"""

import pytest
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulator import (
    TrafficNetwork, TrafficSimulator, TrafficScenario,
    TrafficGenerator
)
from src.utils.config import INTERSECTIONS_FILE, MEDIUM_FLOW_FILE


class TestTrafficScenario:
    """Tests para la clase TrafficScenario."""

    def test_scenario_loading(self):
        """Test de carga de escenario desde archivo."""
        scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))

        assert scenario.name is not None
        assert scenario.traffic_level in ['low', 'medium', 'high']
        assert scenario.vehicle_spawn_rate > 0

    def test_scenario_parameters(self):
        """Test de parámetros del escenario."""
        scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))

        assert scenario.simulation_duration > 0
        assert 0 <= scenario.congestion_level <= 1
        assert scenario.lambda_per_minute > 0

    def test_intersection_flows(self):
        """Test de flujos por intersección."""
        scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))

        # Debe tener flujos definidos para las intersecciones
        assert len(scenario.intersection_flows) > 0

        # Verificar estructura de flujos
        for int_id, flow in scenario.intersection_flows.items():
            assert isinstance(int_id, int)
            assert 'flow_east_west' in flow or 'flow_north_south' in flow


class TestTrafficGenerator:
    """Tests para la clase TrafficGenerator."""

    def test_generator_creation(self):
        """Test de creación del generador."""
        network = TrafficNetwork(str(INTERSECTIONS_FILE))
        scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))

        generator = TrafficGenerator(network, scenario)

        assert generator.network == network
        assert generator.scenario == scenario
        assert generator.total_vehicles_generated == 0

    def test_random_seed(self):
        """Test de reproducibilidad con semilla."""
        network = TrafficNetwork(str(INTERSECTIONS_FILE))
        scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))

        gen1 = TrafficGenerator(network, scenario)
        gen1.set_random_seed(42)

        gen2 = TrafficGenerator(network, scenario)
        gen2.set_random_seed(42)

        # Generar secuencia de origen-destino
        pairs1 = [gen1.generate_origin_destination() for _ in range(10)]
        gen2.current_time = 0
        pairs2 = [gen2.generate_origin_destination() for _ in range(10)]

        # Deben ser iguales con la misma semilla
        assert pairs1 == pairs2

    def test_vehicle_generation(self):
        """Test de generación de vehículos."""
        network = TrafficNetwork(str(INTERSECTIONS_FILE))
        scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))

        generator = TrafficGenerator(network, scenario)
        generator.set_random_seed(42)

        vehicle = generator.generate_vehicle(current_time=0.0)

        assert vehicle is not None
        assert vehicle.origin != vehicle.destination
        assert len(vehicle.route) >= 2

    def test_spawn_rate(self):
        """Test de tasa de generación."""
        network = TrafficNetwork(str(INTERSECTIONS_FILE))
        scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))

        generator = TrafficGenerator(network, scenario)
        generator.set_random_seed(42)

        # Simular 60 segundos
        spawned_count = 0
        for t in range(60):
            if generator.should_spawn_vehicle(t, 1.0):
                spawned_count += 1

        # Debe haber generado algunos vehículos
        assert spawned_count > 0

        # Tasa debe ser razonable (no exacta por aleatoriedad)
        expected_per_minute = scenario.lambda_per_minute
        actual_per_minute = spawned_count

        # Permitir variación del 50%
        assert 0.5 * expected_per_minute < actual_per_minute < 1.5 * expected_per_minute


class TestTrafficSimulator:
    """Tests para la clase TrafficSimulator."""

    def test_simulator_creation(self):
        """Test de creación del simulador."""
        network = TrafficNetwork(str(INTERSECTIONS_FILE))
        scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))

        simulator = TrafficSimulator(network, scenario)

        assert simulator.network == network
        assert simulator.scenario == scenario
        assert len(simulator.traffic_lights) == len(network.intersections)
        assert simulator.current_time == 0.0

    def test_traffic_light_initialization(self):
        """Test de inicialización de semáforos."""
        network = TrafficNetwork(str(INTERSECTIONS_FILE))
        scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))

        simulator = TrafficSimulator(network, scenario)

        # Debe haber un semáforo por intersección
        for int_id in network.get_all_intersection_ids():
            assert int_id in simulator.traffic_lights
            light = simulator.traffic_lights[int_id]
            assert light.intersection_id == int_id

    def test_traffic_light_configuration(self):
        """Test de configuración de semáforos."""
        network = TrafficNetwork(str(INTERSECTIONS_FILE))
        scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))

        simulator = TrafficSimulator(network, scenario)

        # Configurar semáforos
        config = {
            1: {'north_south': 35, 'east_west': 40, 'offset': 5},
            2: {'north_south': 30, 'east_west': 35, 'offset': 10}
        }

        simulator.configure_traffic_lights(config)

        # Verificar configuración
        assert simulator.traffic_lights[1].offset == 5
        assert simulator.traffic_lights[2].offset == 10

    def test_simulation_step(self):
        """Test de un paso de simulación."""
        network = TrafficNetwork(str(INTERSECTIONS_FILE))
        scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))

        simulator = TrafficSimulator(network, scenario)
        simulator.reset()

        initial_time = simulator.current_time

        # Ejecutar un paso
        simulator.step()

        # El tiempo debe haber avanzado
        assert simulator.current_time == initial_time + simulator.dt

    def test_short_simulation(self):
        """Test de simulación corta (30 segundos)."""
        network = TrafficNetwork(str(INTERSECTIONS_FILE))
        scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))

        simulator = TrafficSimulator(network, scenario)

        # Ejecutar simulación corta
        metrics = simulator.run(duration=30, verbose=False)

        # Verificar que retorna métricas
        assert 'avg_delay' in metrics
        assert 'throughput' in metrics
        assert 'vehicles_generated' in metrics
        assert metrics['simulation_time'] == 30.0

    def test_reset_functionality(self):
        """Test de reinicio del simulador."""
        network = TrafficNetwork(str(INTERSECTIONS_FILE))
        scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))

        simulator = TrafficSimulator(network, scenario)

        # Ejecutar simulación
        simulator.run(duration=10, verbose=False)

        assert simulator.current_time > 0
        assert len(simulator.completed_vehicles) >= 0 or len(simulator.active_vehicles) >= 0

        # Reiniciar
        simulator.reset()

        assert simulator.current_time == 0.0
        assert len(simulator.active_vehicles) == 0
        assert len(simulator.completed_vehicles) == 0

    def test_metrics_calculation(self):
        """Test de cálculo de métricas."""
        network = TrafficNetwork(str(INTERSECTIONS_FILE))
        scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))

        simulator = TrafficSimulator(network, scenario)
        simulator.run(duration=60, verbose=False)

        metrics = simulator.calculate_final_metrics()

        # Verificar estructura de métricas
        required_metrics = [
            'avg_delay', 'avg_queue_length', 'throughput',
            'avg_stops', 'vehicles_completed', 'computation_time'
        ]

        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))

    def test_current_state(self):
        """Test de obtención de estado actual."""
        network = TrafficNetwork(str(INTERSECTIONS_FILE))
        scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))

        simulator = TrafficSimulator(network, scenario)

        # Ejecutar algunos pasos
        for _ in range(10):
            simulator.step()

        state = simulator.get_current_state()

        assert 'time' in state
        assert 'active_vehicles' in state
        assert 'traffic_lights' in state
        assert 'queues' in state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
