"""
Generador de tráfico vehicular según patrones y escenarios.

Este módulo implementa la lógica para generar vehículos de manera realista
siguiendo distribuciones de Poisson y patrones de flujo configurables.
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class TrafficScenario:
    """
    Representa un escenario de tráfico con patrones de flujo específicos.

    Un escenario define cómo se genera el tráfico: tasas de llegada,
    distribuciones origen-destino, y niveles de congestión.
    """

    def __init__(self, scenario_file: str):
        """
        Carga un escenario desde archivo JSON.

        Args:
            scenario_file: Ruta al archivo JSON con datos del escenario
        """
        self.scenario_file = scenario_file
        self._load_scenario()

    def _load_scenario(self):
        """Carga los datos del escenario desde el archivo."""
        path = Path(self.scenario_file)
        if not path.exists():
            raise FileNotFoundError(f"Escenario no encontrado: {self.scenario_file}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Metadata del escenario
        self.name = data.get('scenario_name', 'Unknown')
        self.description = data.get('description', '')
        self.traffic_level = data.get('traffic_level', 'medium')
        self.time_period = data.get('time_period', '')

        # Parámetros globales
        params = data.get('global_parameters', {})
        self.simulation_duration = params.get('simulation_duration_s', 3600)
        self.vehicle_spawn_rate = params.get('vehicle_spawn_rate_veh_per_hour', 300)
        self.avg_speed_kmh = params.get('avg_speed_kmh', 35)
        self.congestion_level = params.get('congestion_level', 0.5)

        # Generación de tráfico
        traffic_gen = data.get('traffic_generation', {})
        self.distribution = traffic_gen.get('distribution', 'poisson')
        self.lambda_per_minute = traffic_gen.get('lambda_vehicles_per_minute', 5.0)

        # Matriz origen-destino
        od_matrix = traffic_gen.get('origin_destination_matrix', {})
        self.prob_rambla_to_interior = od_matrix.get('rambla_to_interior', 0.5)
        self.prob_interior_to_rambla = od_matrix.get('interior_to_rambla', 0.5)

        # Flujos por intersección
        self.intersection_flows = {
            flow['intersection_id']: flow
            for flow in data.get('intersection_flows', [])
        }

        print(f"✓ Escenario cargado: {self.name}")
        print(f"  Nivel de tráfico: {self.traffic_level}")
        print(f"  Tasa de generación: {self.vehicle_spawn_rate} veh/hora")

    def get_flow_for_intersection(self, intersection_id: int) -> Dict:
        """
        Obtiene los flujos de tráfico para una intersección específica.

        Args:
            intersection_id: ID de la intersección

        Returns:
            dict: Flujos por dirección
        """
        return self.intersection_flows.get(intersection_id, {})


class TrafficGenerator:
    """
    Genera vehículos según patrones de tráfico realistas.

    Usa distribuciones de Poisson para modelar llegadas de vehículos
    y matrices origen-destino para rutas.
    """

    def __init__(self, network, scenario: TrafficScenario):
        """
        Inicializa el generador de tráfico.

        Args:
            network: Instancia de TrafficNetwork
            scenario: Escenario de tráfico a usar
        """
        self.network = network
        self.scenario = scenario

        # Control de generación
        self.current_time = 0.0
        self.next_spawn_time = 0.0
        self.total_vehicles_generated = 0

        # Para distribución de Poisson
        self.lambda_per_second = scenario.lambda_per_minute / 60.0

        # Semilla para reproducibilidad (opcional)
        self.random_seed = None

    def set_random_seed(self, seed: int):
        """
        Establece semilla para reproducibilidad.

        Args:
            seed: Semilla para generador aleatorio
        """
        self.random_seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def should_spawn_vehicle(self, current_time: float, dt: float) -> bool:
        """
        Determina si debe generarse un vehículo en este paso.

        Usa proceso de Poisson: llegadas siguen distribución exponencial.

        Args:
            current_time: Tiempo actual de simulación (segundos)
            dt: Paso de tiempo (segundos)

        Returns:
            bool: True si debe generarse un vehículo
        """
        self.current_time = current_time

        # Si aún no es momento de generar el próximo vehículo
        if current_time < self.next_spawn_time:
            return False

        # Calcular tiempo hasta próximo vehículo (distribución exponencial)
        # Tiempo entre llegadas ~ Exp(λ)
        inter_arrival_time = np.random.exponential(1.0 / self.lambda_per_second)
        self.next_spawn_time = current_time + inter_arrival_time

        return True

    def generate_origin_destination(self) -> Tuple[int, int]:
        """
        Genera un par origen-destino según patrones del escenario.

        Returns:
            tuple: (origen_id, destino_id)
        """
        intersection_ids = self.network.get_all_intersection_ids()

        if len(intersection_ids) < 2:
            raise ValueError("Red necesita al menos 2 intersecciones")

        # Ordenar IDs (asumimos que menor ID = más cerca de rambla)
        sorted_ids = sorted(intersection_ids)
        min_id = sorted_ids[0]   # Rambla (ID más bajo)
        max_id = sorted_ids[-1]  # Interior (ID más alto)

        # Decidir dirección según probabilidades del escenario
        if random.random() < self.scenario.prob_rambla_to_interior:
            # Rambla → Interior (aumenta ID)
            origin = random.choice(sorted_ids[:2])  # Desde rambla o cerca
            destination = random.choice(sorted_ids[-2:])  # Hacia interior
        else:
            # Interior → Rambla (disminuye ID)
            origin = random.choice(sorted_ids[-2:])  # Desde interior
            destination = random.choice(sorted_ids[:2])  # Hacia rambla

        # Asegurar que origen ≠ destino
        if origin == destination:
            all_except_origin = [i for i in sorted_ids if i != origin]
            if all_except_origin:
                destination = random.choice(all_except_origin)

        return origin, destination

    def generate_vehicle(self, current_time: float):
        """
        Genera un nuevo vehículo con ruta calculada.

        Args:
            current_time: Tiempo actual de simulación

        Returns:
            Vehicle: Nuevo vehículo generado, o None si no se pudo crear ruta
        """
        from .vehicle import Vehicle

        # Generar origen y destino
        origin, destination = self.generate_origin_destination()

        # Calcular ruta
        route = self.network.get_shortest_path(origin, destination)

        if route is None or len(route) < 2:
            # No hay ruta válida, no generar vehículo
            return None

        # Crear vehículo con velocidad según escenario
        max_speed_ms = self.scenario.avg_speed_kmh / 3.6

        vehicle = Vehicle(
            origin=origin,
            destination=destination,
            spawn_time=current_time,
            route=route,
            max_speed_ms=max_speed_ms
        )

        self.total_vehicles_generated += 1

        return vehicle

    def get_spawn_statistics(self) -> Dict:
        """
        Retorna estadísticas de generación de vehículos.

        Returns:
            dict: Estadísticas de spawn
        """
        if self.current_time > 0:
            actual_rate = (self.total_vehicles_generated / self.current_time) * 3600
        else:
            actual_rate = 0

        return {
            'total_generated': self.total_vehicles_generated,
            'target_rate_per_hour': self.scenario.vehicle_spawn_rate,
            'actual_rate_per_hour': actual_rate,
            'lambda_per_second': self.lambda_per_second,
            'scenario_name': self.scenario.name
        }

    def reset(self):
        """Reinicia el generador."""
        self.current_time = 0.0
        self.next_spawn_time = 0.0
        self.total_vehicles_generated = 0


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.simulator.traffic_network import TrafficNetwork
    from src.utils.config import INTERSECTIONS_FILE, HIGH_FLOW_FILE

    print("="*70)
    print("EJEMPLO: Generador de Tráfico")
    print("="*70)

    # Cargar red y escenario
    network = TrafficNetwork(str(INTERSECTIONS_FILE))
    scenario = TrafficScenario(str(HIGH_FLOW_FILE))

    # Crear generador
    generator = TrafficGenerator(network, scenario)
    generator.set_random_seed(42)

    print("\n" + "="*70)
    print("SIMULANDO GENERACIÓN DE VEHÍCULOS")
    print("="*70)

    # Simular 60 segundos de generación
    vehicles_generated = []

    for t in range(60):
        if generator.should_spawn_vehicle(current_time=t, dt=1.0):
            vehicle = generator.generate_vehicle(current_time=t)
            if vehicle:
                vehicles_generated.append(vehicle)
                print(f"T={t:3d}s: Generado {vehicle} (ruta: {vehicle.route})")

    # Estadísticas
    print("\n" + "="*70)
    print("ESTADÍSTICAS DE GENERACIÓN")
    print("="*70)
    stats = generator.get_spawn_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print(f"\nVehículos generados en 60s: {len(vehicles_generated)}")
    print(f"Tasa proyectada: {len(vehicles_generated) * 60} veh/hora")
