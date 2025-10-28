"""
Algoritmo Branch & Bound para coordinación de semáforos.

Este algoritmo coordina múltiples semáforos ajustando sus offsets para
crear "ondas verdes" que permitan a los vehículos pasar varios semáforos
sin detenerse.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from queue import PriorityQueue
import time as timer


class BranchAndBoundNode:
    """
    Nodo del árbol de búsqueda para Branch & Bound.

    Cada nodo representa una asignación parcial de offsets.
    """

    def __init__(self, level: int, offsets: Dict[int, int],
                 cost: float, lower_bound: float):
        """
        Inicializa un nodo del árbol.

        Args:
            level: Profundidad en el árbol (número de offsets asignados)
            offsets: Diccionario {intersection_id: offset}
            cost: Costo real de esta configuración parcial
            lower_bound: Cota inferior del costo final
        """
        self.level = level
        self.offsets = offsets.copy()
        self.cost = cost
        self.lower_bound = lower_bound

    def __lt__(self, other):
        """Comparación para PriorityQueue (menor lower_bound primero)."""
        return self.lower_bound < other.lower_bound


class BranchAndBoundCoordinator:
    """
    Coordinador de semáforos usando Branch & Bound.

    Encuentra offsets óptimos entre semáforos para crear ondas verdes,
    minimizando el delay total de la red.
    """

    def __init__(self, network, simulator=None, offset_step: int = 5,
                 max_offset: int = 120, time_limit: float = 300.0,
                 max_nodes: int = 10000):
        """
        Inicializa el coordinador Branch & Bound.

        Args:
            network: Instancia de TrafficNetwork
            simulator: Simulador para evaluar configuraciones (opcional)
            offset_step: Paso entre offsets probados (segundos)
            max_offset: Offset máximo permitido (segundos)
            time_limit: Tiempo límite de búsqueda (segundos)
            max_nodes: Número máximo de nodos a explorar
        """
        self.network = network
        self.simulator = simulator
        self.offset_step = offset_step
        self.max_offset = max_offset
        self.time_limit = time_limit
        self.max_nodes = max_nodes

        # Mejor solución encontrada
        self.best_solution = None
        self.best_cost = float('inf')

        # Estadísticas
        self.nodes_explored = 0
        self.nodes_pruned = 0
        self.start_time = None

    def calculate_lower_bound(self, node: BranchAndBoundNode,
                              light_configs: Dict, sorted_ids: List[int]) -> float:
        """
        Calcula una cota inferior optimista del costo final.

        Asume que los semáforos restantes se pueden configurar perfectamente
        para minimizar el delay.

        Args:
            node: Nodo actual
            light_configs: Configuraciones base de semáforos
            sorted_ids: Lista ordenada de IDs de intersecciones

        Returns:
            float: Cota inferior del costo
        """
        # Costo actual de offsets asignados
        current_cost = node.cost

        # Estimar costo mínimo de offsets no asignados
        # Simplificación: asumimos coordinación perfecta (costo 0)
        # En realidad, esto sería más sofisticado

        remaining_intersections = len(sorted_ids) - node.level

        # Costo optimista: cada semáforo restante agrega delay mínimo
        # Asumimos 5 segundos de delay mínimo por intersección
        min_delay_per_intersection = 5.0
        optimistic_remaining_cost = remaining_intersections * min_delay_per_intersection

        return current_cost + optimistic_remaining_cost

    def evaluate_configuration(self, offsets: Dict[int, int],
                               light_configs: Dict) -> float:
        """
        Evalúa el costo de una configuración de offsets.

        Args:
            offsets: Diccionario {intersection_id: offset}
            light_configs: Configuraciones base de semáforos

        Returns:
            float: Costo estimado (delay total)
        """
        # Si tenemos simulador, usar simulación completa
        if self.simulator:
            # Crear configuración completa
            full_config = {}
            for int_id, base_config in light_configs.items():
                full_config[int_id] = base_config.copy()
                full_config[int_id]['offset'] = offsets.get(int_id, 0)

            # Configurar y ejecutar simulación corta
            self.simulator.configure_traffic_lights(full_config)
            self.simulator.reset()

            # Simulación corta para evaluar (60 segundos)
            metrics = self.simulator.run(duration=60, verbose=False)

            return metrics['avg_delay']

        # Sin simulador, usar modelo simplificado
        return self._estimate_delay_model(offsets, light_configs)

    def _estimate_delay_model(self, offsets: Dict[int, int],
                              light_configs: Dict) -> float:
        """
        Modelo simplificado para estimar delay sin simulación.

        Evalúa qué tan bien coordinados están los semáforos secuenciales.

        Args:
            offsets: Offsets configurados
            light_configs: Configuraciones de semáforos

        Returns:
            float: Delay estimado
        """
        # Obtener intersecciones ordenadas (asumimos secuencia lineal)
        sorted_ids = sorted(self.network.get_all_intersection_ids())

        total_delay = 0.0

        # Evaluar coordinación entre pares consecutivos
        for i in range(len(sorted_ids) - 1):
            from_id = sorted_ids[i]
            to_id = sorted_ids[i + 1]

            # Obtener offsets
            offset_from = offsets.get(from_id, 0)
            offset_to = offsets.get(to_id, 0)

            # Obtener distancia y tiempo de viaje entre intersecciones
            segment = self.network.get_segment(from_id, to_id)
            if segment:
                travel_time = segment.travel_time_s
            else:
                travel_time = 20  # Asumido si no hay segmento

            # Tiempo ideal de offset para onda verde
            ideal_offset_diff = travel_time

            # Diferencia real de offsets
            actual_offset_diff = offset_to - offset_from

            # Normalizar a ciclo (asumimos 90s)
            cycle_length = 90
            offset_error = abs(actual_offset_diff - ideal_offset_diff) % cycle_length

            # Minimizar error (mejor si está cerca de 0 o cerca de ciclo)
            offset_error = min(offset_error, cycle_length - offset_error)

            # Penalización por descoordinación
            # Si offset_error es 0: coordinación perfecta (delay bajo)
            # Si offset_error es cycle/2: peor caso (delay alto)
            delay_penalty = (offset_error / (cycle_length / 2)) * 50.0

            total_delay += delay_penalty

        # Agregar delay base
        base_delay = len(sorted_ids) * 10.0  # 10s por intersección base

        return base_delay + total_delay

    def coordinate_lights(self, light_configs: Dict[int, Dict]) -> Dict[int, int]:
        """
        Encuentra offsets óptimos usando Branch & Bound.

        Args:
            light_configs: Configuraciones base {intersection_id: {phases}}

        Returns:
            dict: Offsets óptimos {intersection_id: offset}
        """
        print("\nIniciando Branch & Bound para coordinación...")
        self.start_time = timer.time()

        # Reiniciar estadísticas
        self.nodes_explored = 0
        self.nodes_pruned = 0
        self.best_solution = None
        self.best_cost = float('inf')

        # Obtener IDs ordenados (asumimos red lineal)
        sorted_ids = sorted(self.network.get_all_intersection_ids())

        # Cola de prioridad para explorar nodos
        queue = PriorityQueue()

        # Nodo inicial (sin offsets asignados)
        initial_node = BranchAndBoundNode(
            level=0,
            offsets={},
            cost=0.0,
            lower_bound=0.0
        )
        queue.put((initial_node.lower_bound, initial_node))

        # Branch & Bound
        while not queue.empty() and self.nodes_explored < self.max_nodes:
            # Verificar tiempo límite
            elapsed = timer.time() - self.start_time
            if elapsed > self.time_limit:
                print(f"  Tiempo límite alcanzado ({self.time_limit}s)")
                break

            # Obtener nodo con menor lower bound
            _, current_node = queue.get()

            self.nodes_explored += 1

            # Poda: si lower_bound ≥ mejor costo, no explorar
            if current_node.lower_bound >= self.best_cost:
                self.nodes_pruned += 1
                continue

            # Verificar si es solución completa
            if current_node.level >= len(sorted_ids):
                # Evaluar solución completa
                cost = self.evaluate_configuration(current_node.offsets, light_configs)

                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = current_node.offsets.copy()
                    print(f"  Nueva mejor solución: costo={cost:.2f}, "
                          f"nodos explorados={self.nodes_explored}")

                continue

            # Ramificar: probar diferentes offsets para siguiente intersección
            next_int_id = sorted_ids[current_node.level]

            for offset in range(0, self.max_offset + 1, self.offset_step):
                # Crear nuevo nodo hijo
                new_offsets = current_node.offsets.copy()
                new_offsets[next_int_id] = offset

                # Evaluar costo parcial
                partial_cost = self._estimate_delay_model(new_offsets, light_configs)

                # Calcular lower bound
                child_node = BranchAndBoundNode(
                    level=current_node.level + 1,
                    offsets=new_offsets,
                    cost=partial_cost,
                    lower_bound=partial_cost  # Simplificado
                )

                # Poda temprana
                if child_node.lower_bound < self.best_cost:
                    queue.put((child_node.lower_bound, child_node))
                else:
                    self.nodes_pruned += 1

        elapsed = timer.time() - self.start_time

        print(f"  Búsqueda completada en {elapsed:.2f}s")
        print(f"  Nodos explorados: {self.nodes_explored}")
        print(f"  Nodos podados: {self.nodes_pruned}")
        print(f"  Mejor costo: {self.best_cost:.2f}")

        # Si no encontramos solución, usar offsets 0
        if self.best_solution is None:
            self.best_solution = {int_id: 0 for int_id in sorted_ids}

        return self.best_solution

    def generate_configuration(self, light_configs: Dict[int, Dict]) -> Dict[int, Dict]:
        """
        Genera configuración completa con offsets optimizados.

        Args:
            light_configs: Configuraciones base de semáforos

        Returns:
            dict: Configuración completa con offsets
        """
        # Encontrar offsets óptimos
        optimal_offsets = self.coordinate_lights(light_configs)

        # Crear configuración completa
        full_config = {}
        for int_id, base_config in light_configs.items():
            full_config[int_id] = base_config.copy()
            full_config[int_id]['offset'] = optimal_offsets.get(int_id, 0)

        return full_config

    def get_statistics(self) -> Dict:
        """
        Retorna estadísticas del algoritmo.

        Returns:
            dict: Estadísticas de operación
        """
        elapsed = 0.0
        if self.start_time:
            elapsed = timer.time() - self.start_time

        return {
            'algorithm': 'Branch & Bound',
            'nodes_explored': self.nodes_explored,
            'nodes_pruned': self.nodes_pruned,
            'pruning_rate': self.nodes_pruned / max(1, self.nodes_explored + self.nodes_pruned),
            'best_cost': self.best_cost,
            'computation_time': elapsed,
            'offset_step': self.offset_step
        }


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.simulator import TrafficNetwork
    from src.utils.config import INTERSECTIONS_FILE

    print("="*70)
    print("EJEMPLO: Branch & Bound para Coordinación")
    print("="*70)

    # Cargar red
    network = TrafficNetwork(str(INTERSECTIONS_FILE))

    # Crear coordinador
    coordinator = BranchAndBoundCoordinator(
        network,
        simulator=None,  # Sin simulador para demo rápida
        offset_step=10,
        max_offset=120,
        time_limit=60.0
    )

    # Configuraciones base de semáforos (de DP por ejemplo)
    light_configs = {}
    for int_id in network.get_all_intersection_ids():
        light_configs[int_id] = {
            'north_south': 30,
            'east_west': 35
        }

    print("\nConfiguraciones base:")
    for int_id, config in light_configs.items():
        print(f"  Intersección {int_id}: NS={config['north_south']}s, "
              f"EW={config['east_west']}s")

    print("\n" + "="*70)
    print("OPTIMIZACIÓN DE OFFSETS")
    print("="*70)

    # Coordinar semáforos
    full_config = coordinator.generate_configuration(light_configs)

    print("\n" + "="*70)
    print("CONFIGURACIÓN ÓPTIMA CON OFFSETS")
    print("="*70)
    for int_id in sorted(full_config.keys()):
        config = full_config[int_id]
        print(f"\nIntersección {int_id}:")
        print(f"  Norte-Sur: {config['north_south']}s")
        print(f"  Este-Oeste: {config['east_west']}s")
        print(f"  Offset: {config['offset']}s")

    # Estadísticas
    print("\n" + "="*70)
    print("ESTADÍSTICAS")
    print("="*70)
    stats = coordinator.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
