"""
Algoritmo de Programación Dinámica para optimización de semáforos.

Este algoritmo optimiza la distribución de tiempos de verde dentro de un
ÚNICO semáforo, encontrando la asignación óptima que minimiza el delay
total de los vehículos.
"""

from typing import Dict, List, Tuple
import numpy as np
from copy import deepcopy


class DynamicProgrammingOptimizer:
    """
    Optimizador basado en Programación Dinámica.

    Optimiza la distribución de tiempos de verde entre las fases de un
    semáforo individual usando programación dinámica.

    El problema se modela como:
    - Estados: (tiempo_restante, fases_asignadas)
    - Decisiones: duración de la próxima fase
    - Objetivo: minimizar delay total esperado
    """

    def __init__(self, network, cycle_length: int = 90,
                 time_granularity: int = 5, min_green: int = 10,
                 max_green: int = 60):
        """
        Inicializa el optimizador de Programación Dinámica.

        Args:
            network: Instancia de TrafficNetwork
            cycle_length: Duración total del ciclo en segundos
            time_granularity: Granularidad de tiempo para DP (segundos)
            min_green: Tiempo mínimo de verde por fase (segundos)
            max_green: Tiempo máximo de verde por fase (segundos)
        """
        self.network = network
        self.cycle_length = cycle_length
        self.time_granularity = time_granularity
        self.min_green = min_green
        self.max_green = max_green

        # Fases disponibles
        self.phases = ['north_south', 'east_west']

        # Overhead de cada fase (amarillo + todo-rojo)
        self.phase_overhead = 4  # 3s amarillo + 1s todo-rojo

        # Tabla de memoización
        self.dp_table = {}

        # Estadísticas
        self.states_evaluated = 0
        self.optimizations_performed = 0

    def estimate_delay(self, phase: str, green_duration: int,
                      traffic_data: Dict) -> float:
        """
        Estima el delay causado por una configuración de fase.

        Usa el modelo de Webster simplificado para calcular delay en colas.

        Args:
            phase: Nombre de la fase
            green_duration: Duración del verde en segundos
            traffic_data: Datos de flujo vehicular

        Returns:
            float: Delay estimado en vehículos·segundos
        """
        # Obtener tasa de llegada (vehículos/segundo)
        arrival_rate = traffic_data.get(f'{phase}_arrival_rate', 0.3)

        # Calcular capacidad (vehículos que pueden pasar durante verde)
        # Asumimos saturation flow rate de 0.5 veh/s (1800 veh/h por carril)
        saturation_flow = 0.5  # veh/s
        capacity_per_cycle = saturation_flow * green_duration

        # Tasa de servicio efectiva (considerando el ciclo completo)
        effective_green_ratio = green_duration / self.cycle_length
        service_rate = saturation_flow * effective_green_ratio

        # Factor de utilización
        rho = arrival_rate / service_rate if service_rate > 0 else 1.0

        # Evitar saturación (ρ ≥ 1)
        if rho >= 0.98:
            # Penalización alta si la fase no puede manejar el tráfico
            return 10000.0 * (1.0 + rho)

        # Modelo de Webster para delay promedio
        # d = C(1-λ)²/(2(1-ρ)) donde C es el ciclo
        try:
            delay = (self.cycle_length * (1 - effective_green_ratio)**2) / \
                    (2 * (1 - rho))
        except (ZeroDivisionError, ValueError):
            delay = 10000.0

        # Delay total = delay_promedio * tasa_llegada * ciclo
        total_delay = delay * arrival_rate * self.cycle_length

        return total_delay

    def optimize_single_light(self, intersection_id: int,
                             traffic_data: Dict) -> Dict[str, int]:
        """
        Optimiza un solo semáforo usando Programación Dinámica.

        Args:
            intersection_id: ID de la intersección a optimizar
            traffic_data: Datos de flujo vehicular para esta intersección

        Returns:
            dict: Configuración óptima {fase: duración_verde}
        """
        self.optimizations_performed += 1
        self.dp_table.clear()
        self.states_evaluated = 0

        # Tiempo disponible para verde (ciclo - overhead de todas las fases)
        available_time = self.cycle_length - (len(self.phases) * self.phase_overhead)

        # Resolver con DP
        optimal_cost, optimal_config = self._dp_solve(
            available_time, 0, [], traffic_data
        )

        # Convertir lista de duraciones a diccionario
        result = {}
        for i, phase in enumerate(self.phases):
            if i < len(optimal_config):
                result[phase] = optimal_config[i]
            else:
                result[phase] = self.min_green

        return result

    def _dp_solve(self, time_remaining: int, phase_index: int,
                  current_config: List[int],
                  traffic_data: Dict) -> Tuple[float, List[int]]:
        """
        Resuelve el problema usando programación dinámica.

        Args:
            time_remaining: Tiempo restante a asignar
            phase_index: Índice de la fase actual
            current_config: Configuración actual (lista de duraciones)
            traffic_data: Datos de tráfico

        Returns:
            tuple: (costo_mínimo, configuración_óptima)
        """
        self.states_evaluated += 1

        # Estado como tupla para memoización
        state_key = (time_remaining, phase_index, tuple(current_config))

        # Verificar si ya calculamos este estado
        if state_key in self.dp_table:
            return self.dp_table[state_key]

        # Caso base: todas las fases asignadas
        if phase_index >= len(self.phases):
            # Calcular costo total de esta configuración
            total_delay = 0.0
            for i, phase in enumerate(self.phases):
                if i < len(current_config):
                    delay = self.estimate_delay(
                        phase, current_config[i], traffic_data
                    )
                    total_delay += delay

            return (total_delay, current_config[:])

        # Caso recursivo: asignar tiempo a la fase actual
        min_cost = float('inf')
        best_config = current_config[:]

        # Probar diferentes duraciones para esta fase
        # Usamos time_granularity para reducir espacio de búsqueda
        min_duration = self.min_green
        max_duration = min(self.max_green, time_remaining)

        for duration in range(min_duration, max_duration + 1, self.time_granularity):
            # Crear nueva configuración con esta duración
            new_config = current_config + [duration]
            new_time_remaining = time_remaining - duration

            # Si es la última fase, debe usar todo el tiempo restante
            if phase_index == len(self.phases) - 1:
                if new_time_remaining >= self.min_green:
                    duration = new_time_remaining
                    new_config = current_config + [duration]
                    new_time_remaining = 0
                else:
                    # No hay suficiente tiempo, configuración inválida
                    continue

            # Resolver subproblema recursivamente
            cost, config = self._dp_solve(
                new_time_remaining, phase_index + 1,
                new_config, traffic_data
            )

            # Actualizar mejor solución
            if cost < min_cost:
                min_cost = cost
                best_config = config

        # Memoizar resultado
        self.dp_table[state_key] = (min_cost, best_config)

        return (min_cost, best_config)

    def optimize_network(self, traffic_data: Dict[int, Dict]) -> Dict[int, Dict]:
        """
        Optimiza todos los semáforos de la red independientemente.

        Args:
            traffic_data: Datos de tráfico por intersección
                         {intersection_id: {phase: arrival_rate}}

        Returns:
            dict: Configuración completa {intersection_id: {fase: duración}}
        """
        configuration = {}

        print(f"\nOptimizando {len(self.network.intersections)} semáforos con DP...")

        for int_id in self.network.get_all_intersection_ids():
            # Obtener datos de tráfico para esta intersección
            int_traffic = traffic_data.get(int_id, {})

            # Optimizar este semáforo
            optimal_config = self.optimize_single_light(int_id, int_traffic)

            # Agregar a configuración (sin offset, DP no coordina)
            optimal_config['offset'] = 0
            configuration[int_id] = optimal_config

            print(f"  Intersección {int_id}: NS={optimal_config['north_south']}s, "
                  f"EW={optimal_config['east_west']}s "
                  f"(estados evaluados: {self.states_evaluated})")

            # Resetear contador para siguiente intersección
            self.states_evaluated = 0

        return configuration

    def get_statistics(self) -> Dict:
        """
        Retorna estadísticas del optimizador.

        Returns:
            dict: Estadísticas de operación
        """
        return {
            'algorithm': 'Dynamic Programming',
            'optimizations_performed': self.optimizations_performed,
            'avg_states_per_optimization': len(self.dp_table) / max(1, self.optimizations_performed),
            'cycle_length': self.cycle_length,
            'time_granularity': self.time_granularity
        }

    def reset(self):
        """Reinicia el optimizador."""
        self.dp_table.clear()
        self.states_evaluated = 0
        self.optimizations_performed = 0


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.simulator import TrafficNetwork
    from src.utils.config import INTERSECTIONS_FILE

    print("="*70)
    print("EJEMPLO: Optimizador de Programación Dinámica")
    print("="*70)

    # Cargar red
    network = TrafficNetwork(str(INTERSECTIONS_FILE))

    # Crear optimizador
    optimizer = DynamicProgrammingOptimizer(
        network,
        cycle_length=90,
        time_granularity=5,
        min_green=10,
        max_green=60
    )

    # Datos de tráfico ficticios para ejemplo
    traffic_data = {
        1: {
            'north_south_arrival_rate': 0.25,  # 0.25 veh/s = 900 veh/h
            'east_west_arrival_rate': 0.40     # 0.40 veh/s = 1440 veh/h
        },
        2: {
            'north_south_arrival_rate': 0.30,
            'east_west_arrival_rate': 0.35
        },
        3: {
            'north_south_arrival_rate': 0.20,
            'east_west_arrival_rate': 0.45
        }
    }

    print("\nDatos de tráfico (tasas de llegada en veh/s):")
    for int_id, data in traffic_data.items():
        print(f"  Intersección {int_id}:")
        print(f"    Norte-Sur: {data['north_south_arrival_rate']} veh/s")
        print(f"    Este-Oeste: {data['east_west_arrival_rate']} veh/s")

    print("\n" + "="*70)
    print("OPTIMIZACIÓN")
    print("="*70)

    # Optimizar toda la red
    configuration = optimizer.optimize_network(traffic_data)

    print("\n" + "="*70)
    print("CONFIGURACIÓN ÓPTIMA")
    print("="*70)
    for int_id, config in configuration.items():
        cycle_time = config['north_south'] + config['east_west'] + 8  # +8 overhead
        print(f"\nIntersección {int_id}:")
        print(f"  Norte-Sur: {config['north_south']}s verde")
        print(f"  Este-Oeste: {config['east_west']}s verde")
        print(f"  Ciclo total: {cycle_time}s")

    # Estadísticas
    print("\n" + "="*70)
    print("ESTADÍSTICAS")
    print("="*70)
    stats = optimizer.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
