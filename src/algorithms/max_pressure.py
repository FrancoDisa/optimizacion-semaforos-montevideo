"""
Algoritmo Max-Pressure para control de semáforos en tiempo real.

Este algoritmo es una heurística simple que toma decisiones basadas en la
"presión" de cada dirección, priorizando las direcciones con más vehículos
esperando. Sirve como baseline para comparación con algoritmos más sofisticados.
"""

from typing import Dict, List
import numpy as np


class MaxPressureController:
    """
    Controlador heurístico Max-Pressure.

    Toma decisiones en tiempo real basándose en la presión de cada dirección.
    La presión se define como: vehículos_esperando - capacidad_downstream

    Este es un algoritmo simple pero efectivo que:
    - No requiere optimización offline
    - Reacciona dinámicamente al tráfico
    - Es usado como baseline para comparación
    """

    def __init__(self, network, min_phase_duration: int = 10,
                 pressure_threshold: float = 5.0):
        """
        Inicializa el controlador Max-Pressure.

        Args:
            network: Instancia de TrafficNetwork
            min_phase_duration: Duración mínima de cada fase en segundos
            pressure_threshold: Umbral de presión para cambiar de fase
        """
        self.network = network
        self.min_phase_duration = min_phase_duration
        self.pressure_threshold = pressure_threshold

        # Estado de cada semáforo
        self.current_phases = {}  # {intersection_id: current_phase_name}
        self.phase_start_times = {}  # {intersection_id: start_time}

        # Inicializar estado
        for int_id in network.get_all_intersection_ids():
            self.current_phases[int_id] = "north_south"  # Fase inicial
            self.phase_start_times[int_id] = 0.0

        # Estadísticas
        self.phase_changes = 0
        self.decisions_made = 0

    def calculate_pressure(self, intersection_id: int, direction: str,
                          traffic_state: Dict) -> float:
        """
        Calcula la presión de una dirección en una intersección.

        Presión = vehículos_esperando - capacidad_downstream

        Args:
            intersection_id: ID de la intersección
            direction: Dirección a evaluar ("north_south", "east_west", etc.)
            traffic_state: Estado actual del tráfico

        Returns:
            float: Valor de presión (mayor = más urgente)
        """
        intersection = self.network.get_intersection(intersection_id)
        if not intersection:
            return 0.0

        # Mapeo de direcciones a colas
        direction_map = {
            'north_south': ['north_south', 'south_north'],
            'east_west': ['east_west', 'west_east'],
            'left_turns': ['left_north', 'left_south', 'left_east', 'left_west']
        }

        # Contar vehículos esperando en esta dirección
        vehicles_waiting = 0
        if direction in direction_map:
            for queue_dir in direction_map[direction]:
                queue_length = intersection.get_queue_length(queue_dir)
                vehicles_waiting += queue_length

        # Estimar capacidad downstream (simplificado)
        # En un modelo completo, esto consideraría el estado de las
        # intersecciones siguientes
        downstream_capacity = self._estimate_downstream_capacity(
            intersection_id, direction
        )

        pressure = vehicles_waiting - downstream_capacity

        return pressure

    def _estimate_downstream_capacity(self, intersection_id: int,
                                      direction: str) -> float:
        """
        Estima la capacidad de avance hacia downstream.

        Simplificación: asume capacidad constante basada en vecinos.

        Args:
            intersection_id: ID de la intersección
            direction: Dirección de flujo

        Returns:
            float: Capacidad estimada (vehículos que pueden avanzar)
        """
        # Simplificación: capacidad base de 5 vehículos por segmento
        base_capacity = 5.0

        # Obtener vecinos downstream
        neighbors = self.network.get_neighbors(intersection_id)

        if neighbors:
            # Si hay vecinos, la capacidad depende de cuántos pueden recibir
            return base_capacity * len(neighbors)
        else:
            # Nodo terminal, capacidad máxima (pueden salir libremente)
            return 100.0

    def decide_phase(self, intersection_id: int, current_time: float,
                    traffic_state: Dict) -> str:
        """
        Decide qué fase activar según la presión máxima.

        Args:
            intersection_id: ID de la intersección
            current_time: Tiempo actual de simulación
            traffic_state: Estado actual del tráfico

        Returns:
            str: Nombre de la fase a activar
        """
        self.decisions_made += 1

        # Verificar tiempo mínimo en fase actual
        current_phase = self.current_phases[intersection_id]
        phase_start = self.phase_start_times[intersection_id]
        time_in_phase = current_time - phase_start

        if time_in_phase < self.min_phase_duration:
            # No cambiar aún, no ha pasado tiempo mínimo
            return current_phase

        # Calcular presión de cada dirección
        pressures = {}
        for direction in ['north_south', 'east_west', 'left_turns']:
            pressures[direction] = self.calculate_pressure(
                intersection_id, direction, traffic_state
            )

        # Encontrar dirección con mayor presión
        max_pressure_direction = max(pressures, key=pressures.get)
        max_pressure = pressures[max_pressure_direction]

        # Cambiar solo si:
        # 1. La presión máxima es mayor que la actual
        # 2. La diferencia supera el umbral
        current_pressure = pressures[current_phase]

        if max_pressure > current_pressure + self.pressure_threshold:
            # Cambiar a la fase con mayor presión
            if max_pressure_direction != current_phase:
                self.current_phases[intersection_id] = max_pressure_direction
                self.phase_start_times[intersection_id] = current_time
                self.phase_changes += 1

            return max_pressure_direction
        else:
            # Mantener fase actual
            return current_phase

    def generate_configuration(self, traffic_state: Dict,
                              current_time: float = 0.0) -> Dict[int, Dict]:
        """
        Genera configuración de semáforos para un momento dado.

        Nota: Max-Pressure es reactivo, no genera configuraciones offline.
        Este método genera una configuración "snapshot" para un momento.

        Args:
            traffic_state: Estado del tráfico
            current_time: Tiempo actual

        Returns:
            dict: Configuración {intersection_id: {phases, offset}}
        """
        configuration = {}

        for int_id in self.network.get_all_intersection_ids():
            # Decidir fase para esta intersección
            phase = self.decide_phase(int_id, current_time, traffic_state)

            # Max-Pressure usa tiempos estándar, la inteligencia está
            # en cuándo cambiar de fase
            config = {
                'north_south': 30,
                'east_west': 30,
                'offset': 0  # Max-Pressure no usa coordinación
            }

            configuration[int_id] = config

        return configuration

    def get_statistics(self) -> Dict:
        """
        Retorna estadísticas del controlador.

        Returns:
            dict: Estadísticas de operación
        """
        return {
            'algorithm': 'Max-Pressure',
            'phase_changes': self.phase_changes,
            'decisions_made': self.decisions_made,
            'avg_decisions_per_change': (
                self.decisions_made / max(1, self.phase_changes)
            )
        }

    def reset(self):
        """Reinicia el controlador."""
        for int_id in self.network.get_all_intersection_ids():
            self.current_phases[int_id] = "north_south"
            self.phase_start_times[int_id] = 0.0

        self.phase_changes = 0
        self.decisions_made = 0


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.simulator import TrafficNetwork
    from src.utils.config import INTERSECTIONS_FILE

    print("="*70)
    print("EJEMPLO: Controlador Max-Pressure")
    print("="*70)

    # Cargar red
    network = TrafficNetwork(str(INTERSECTIONS_FILE))

    # Crear controlador
    controller = MaxPressureController(network, min_phase_duration=10)

    # Simular estado de tráfico ficticio
    traffic_state = {
        'queues': {
            1: {'north_south': 8, 'east_west': 3},
            2: {'north_south': 5, 'east_west': 10},
            3: {'north_south': 2, 'east_west': 7},
        }
    }

    print("\nEstado del tráfico (colas):")
    for int_id, queues in traffic_state['queues'].items():
        print(f"  Intersección {int_id}:")
        print(f"    Norte-Sur: {queues['north_south']} vehículos")
        print(f"    Este-Oeste: {queues['east_west']} vehículos")

    print("\n" + "="*70)
    print("DECISIONES DEL CONTROLADOR")
    print("="*70)

    # Tomar decisiones para cada intersección
    for int_id in [1, 2, 3]:
        print(f"\nIntersección {int_id}:")

        # Calcular presiones
        pressure_ns = controller.calculate_pressure(int_id, 'north_south', traffic_state)
        pressure_ew = controller.calculate_pressure(int_id, 'east_west', traffic_state)

        print(f"  Presión Norte-Sur: {pressure_ns:.2f}")
        print(f"  Presión Este-Oeste: {pressure_ew:.2f}")

        # Decidir fase
        phase = controller.decide_phase(int_id, current_time=15.0, traffic_state=traffic_state)
        print(f"  → Fase seleccionada: {phase}")

    # Estadísticas
    print("\n" + "="*70)
    print("ESTADÍSTICAS")
    print("="*70)
    stats = controller.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
