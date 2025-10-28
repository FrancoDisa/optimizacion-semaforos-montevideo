"""
Motor principal de simulación de tráfico vehicular.

Este módulo implementa el simulador que coordina todos los componentes:
red vial, semáforos, vehículos, y generación de tráfico.
"""

from typing import Dict, List, Optional
from pathlib import Path
import time as timer

from .traffic_network import TrafficNetwork
from .traffic_light import TrafficLight
from .vehicle import Vehicle, VehicleState
from .traffic_generator import TrafficGenerator, TrafficScenario


class TrafficSimulator:
    """
    Motor principal de simulación de tráfico.

    Coordina la interacción entre vehículos, semáforos y red vial,
    ejecutando la simulación paso a paso y recolectando métricas.
    """

    def __init__(self, network: TrafficNetwork, scenario: TrafficScenario):
        """
        Inicializa el simulador.

        Args:
            network: Red vial
            scenario: Escenario de tráfico
        """
        self.network = network
        self.scenario = scenario

        # Generador de tráfico
        self.traffic_generator = TrafficGenerator(network, scenario)

        # Semáforos
        self.traffic_lights: Dict[int, TrafficLight] = {}
        self._initialize_traffic_lights()

        # Vehículos
        self.active_vehicles: List[Vehicle] = []
        self.completed_vehicles: List[Vehicle] = []

        # Estado de simulación
        self.current_time = 0.0
        self.dt = 1.0  # Paso de tiempo en segundos

        # Métricas en tiempo real
        self.metrics_history = []
        self.queue_length_history = []

        # Control de simulación
        self.is_running = False
        self.real_time_start = None

        print(f"✓ Simulador inicializado")
        print(f"  Red: {network.network_name}")
        print(f"  Escenario: {scenario.name}")
        print(f"  Intersecciones: {len(network.intersections)}")

    def _initialize_traffic_lights(self):
        """
        Crea y configura semáforos para cada intersección.

        Por defecto, cada semáforo tiene configuración estándar.
        """
        for intersection_id, intersection in self.network.intersections.items():
            traffic_light = TrafficLight(
                intersection_id=intersection_id,
                intersection_name=intersection.name
            )

            # Configuración por defecto
            traffic_light.set_configuration({
                "north_south": 30,
                "east_west": 30
            }, offset=0)

            self.traffic_lights[intersection_id] = traffic_light
            intersection.traffic_light = traffic_light

    def configure_traffic_lights(self, configuration: Dict[int, Dict]):
        """
        Configura los semáforos según parámetros dados.

        Args:
            configuration: Dict {intersection_id: {'north_south': 35, 'east_west': 40, 'offset': 10}}
        """
        for intersection_id, config in configuration.items():
            if intersection_id in self.traffic_lights:
                light = self.traffic_lights[intersection_id]

                # Extraer offset si existe
                offset = config.pop('offset', 0)

                # Configurar fases
                light.set_configuration(config, offset=offset)

                print(f"  Semáforo {intersection_id} configurado: {config}, offset={offset}s")

    def run(self, duration: int, verbose: bool = False) -> Dict:
        """
        Ejecuta la simulación por un tiempo determinado.

        Args:
            duration: Duración de la simulación en segundos
            verbose: Si True, imprime progreso

        Returns:
            dict: Métricas finales de la simulación
        """
        print(f"\n{'='*70}")
        print(f"INICIANDO SIMULACIÓN")
        print(f"{'='*70}")
        print(f"Duración: {duration}s ({duration/60:.1f} minutos)")

        self.reset()
        self.is_running = True
        self.real_time_start = timer.time()

        num_steps = int(duration / self.dt)

        for step in range(num_steps):
            self.step()

            # Progreso cada 10% (o cada 60s si duration < 600)
            if verbose:
                report_interval = max(60, duration // 10)
                if step % int(report_interval / self.dt) == 0:
                    self._print_progress()

        self.is_running = False

        # Calcular métricas finales
        metrics = self.calculate_final_metrics()

        print(f"\n{'='*70}")
        print(f"SIMULACIÓN COMPLETADA")
        print(f"{'='*70}")
        self._print_summary(metrics)

        return metrics

    def step(self):
        """
        Ejecuta un paso de simulación (típicamente 1 segundo).

        Este es el método central que coordina todas las actualizaciones.
        """
        # 1. Generar nuevos vehículos
        self._spawn_vehicles()

        # 2. Actualizar semáforos
        self._update_traffic_lights()

        # 3. Actualizar vehículos
        self._update_vehicles()

        # 4. Registrar métricas instantáneas
        self._record_metrics()

        # 5. Avanzar tiempo
        self.current_time += self.dt

    def _spawn_vehicles(self):
        """Genera nuevos vehículos según el escenario."""
        if self.traffic_generator.should_spawn_vehicle(self.current_time, self.dt):
            vehicle = self.traffic_generator.generate_vehicle(self.current_time)

            if vehicle:
                # Asignar al primer segmento de la ruta
                from_id = vehicle.get_current_from_intersection()
                to_id = vehicle.get_current_to_intersection()

                segment = self.network.get_segment(from_id, to_id)

                if segment and not segment.is_full():
                    vehicle.current_segment = segment
                    segment.add_vehicle(vehicle)
                    self.active_vehicles.append(vehicle)

    def _update_traffic_lights(self):
        """Actualiza el estado de todos los semáforos."""
        for traffic_light in self.traffic_lights.values():
            traffic_light.update(self.current_time, self.dt)

    def _update_vehicles(self):
        """Actualiza el estado de todos los vehículos activos."""
        vehicles_to_remove = []

        for vehicle in self.active_vehicles:
            # Verificar si el vehículo ya llegó
            if vehicle.has_arrived():
                if vehicle.arrival_time is None:
                    vehicle.arrival_time = self.current_time
                vehicles_to_remove.append(vehicle)
                continue

            # Asignar referencias necesarias
            self._update_vehicle_references(vehicle)

            # Actualizar vehículo
            vehicle.update(self.dt, self.current_time)

            # Verificar si cambió de segmento
            self._handle_segment_transition(vehicle)

        # Mover vehículos completados
        for vehicle in vehicles_to_remove:
            self.active_vehicles.remove(vehicle)
            self.completed_vehicles.append(vehicle)

            # Remover del segmento actual
            if vehicle.current_segment:
                vehicle.current_segment.remove_vehicle(vehicle)

    def _update_vehicle_references(self, vehicle: Vehicle):
        """
        Actualiza las referencias de un vehículo a otros objetos.

        Args:
            vehicle: Vehículo a actualizar
        """
        if vehicle.has_arrived():
            return

        # Actualizar referencia al segmento actual
        from_id = vehicle.get_current_from_intersection()
        to_id = vehicle.get_current_to_intersection()
        vehicle.current_segment = self.network.get_segment(from_id, to_id)

        # Actualizar referencia al próximo semáforo
        vehicle.next_traffic_light = self.traffic_lights.get(to_id)

        # Encontrar vehículo adelante en el mismo segmento
        vehicle.vehicle_ahead = None
        if vehicle.current_segment:
            vehicles_on_segment = vehicle.current_segment.vehicles
            vehicles_ahead = [
                v for v in vehicles_on_segment
                if v != vehicle and v.position_on_edge > vehicle.position_on_edge
            ]
            if vehicles_ahead:
                # El más cercano adelante
                vehicle.vehicle_ahead = min(vehicles_ahead, key=lambda v: v.position_on_edge)

    def _handle_segment_transition(self, vehicle: Vehicle):
        """
        Maneja la transición de un vehículo al finalizar un segmento.

        Args:
            vehicle: Vehículo a verificar
        """
        if not vehicle.current_segment:
            return

        # Verificar si llegó al final del segmento
        if vehicle.position_on_edge >= vehicle.current_segment.length_m:
            # Verificar estado del semáforo
            direction = vehicle.get_direction()
            intersection = self.network.get_intersection(vehicle.get_current_to_intersection())

            if not intersection or not intersection.traffic_light:
                # No hay semáforo, pasar libremente
                self._move_vehicle_to_next_segment(vehicle)
                return

            # Hay semáforo, verificar si puede pasar
            if intersection.traffic_light.can_vehicle_pass(direction):
                # Verde, puede pasar
                self._move_vehicle_to_next_segment(vehicle)
            else:
                # Rojo/Amarillo, debe detenerse
                vehicle.current_speed = 0.0
                vehicle.state = VehicleState.STOPPED_AT_LIGHT

                # Agregar a cola de la intersección
                if vehicle not in intersection.queue_north_south and \
                   vehicle not in intersection.queue_east_west and \
                   vehicle not in intersection.queue_south_north and \
                   vehicle not in intersection.queue_west_east:
                    # Determinar dirección para cola
                    direction_map = {
                        'north': 'north_south',
                        'south': 'south_north',
                        'east': 'east_west',
                        'west': 'west_east'
                    }
                    queue_direction = direction_map.get(direction, 'north_south')
                    intersection.add_vehicle_to_queue(vehicle, queue_direction)

    def _move_vehicle_to_next_segment(self, vehicle: Vehicle):
        """
        Mueve un vehículo al siguiente segmento de su ruta.

        Args:
            vehicle: Vehículo a mover
        """
        # Remover del segmento actual
        if vehicle.current_segment:
            vehicle.current_segment.remove_vehicle(vehicle)

        # Remover de cualquier cola
        current_to_id = vehicle.get_current_to_intersection()
        intersection = self.network.get_intersection(current_to_id)
        if intersection:
            direction = vehicle.get_direction()
            direction_map = {
                'north': 'north_south',
                'south': 'south_north',
                'east': 'east_west',
                'west': 'west_east'
            }
            queue_direction = direction_map.get(direction, 'north_south')
            intersection.remove_vehicle_from_queue(vehicle, queue_direction)

        # Avanzar índice de ruta
        vehicle.current_edge_index += 1
        vehicle.position_on_edge = 0.0

        # Verificar si llegó a destino
        if vehicle.current_edge_index >= len(vehicle.route) - 1:
            vehicle.state = VehicleState.ARRIVED
            return

        # Asignar nuevo segmento
        from_id = vehicle.get_current_from_intersection()
        to_id = vehicle.get_current_to_intersection()
        new_segment = self.network.get_segment(from_id, to_id)

        if new_segment and not new_segment.is_full():
            vehicle.current_segment = new_segment
            new_segment.add_vehicle(vehicle)
        else:
            # Segmento lleno, vehículo debe esperar
            vehicle.state = VehicleState.WAITING_IN_QUEUE

    def _record_metrics(self):
        """Registra métricas instantáneas de la simulación."""
        # Longitudes de cola en cada intersección
        queue_snapshot = {}
        for int_id, intersection in self.network.intersections.items():
            queue_snapshot[int_id] = intersection.get_total_queue_length()

        self.queue_length_history.append({
            'time': self.current_time,
            'queues': queue_snapshot.copy()
        })

    def calculate_final_metrics(self) -> Dict:
        """
        Calcula métricas finales de la simulación.

        Returns:
            dict: Diccionario con todas las métricas
        """
        if not self.completed_vehicles:
            return {
                'avg_delay': 0.0,
                'avg_queue_length': 0.0,
                'throughput': 0,
                'avg_stops': 0.0,
                'max_queue_length': 0,
                'total_travel_time': 0.0,
                'avg_speed_kmh': 0.0,
                'computation_time': 0.0,
                'vehicles_completed': 0,
                'vehicles_active': len(self.active_vehicles)
            }

        # Calcular tiempo de cómputo real
        computation_time = 0.0
        if self.real_time_start:
            computation_time = timer.time() - self.real_time_start

        # Retrasos
        delays = []
        for vehicle in self.completed_vehicles:
            ideal_time = self.network.get_path_travel_time(vehicle.route)
            delay = vehicle.get_delay(ideal_time)
            delays.append(delay)

        avg_delay = sum(delays) / len(delays) if delays else 0.0

        # Paradas
        stops = [v.num_stops for v in self.completed_vehicles]
        avg_stops = sum(stops) / len(stops) if stops else 0.0

        # Tiempos de espera
        waiting_times = [v.total_waiting_time for v in self.completed_vehicles]
        avg_waiting_time = sum(waiting_times) / len(waiting_times) if waiting_times else 0.0

        # Velocidades promedio
        speeds = [v.get_average_speed_kmh() for v in self.completed_vehicles]
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

        # Colas
        all_queue_lengths = []
        for snapshot in self.queue_length_history:
            all_queue_lengths.extend(snapshot['queues'].values())

        avg_queue_length = sum(all_queue_lengths) / len(all_queue_lengths) if all_queue_lengths else 0.0
        max_queue_length = max(all_queue_lengths) if all_queue_lengths else 0

        # Throughput
        throughput = len(self.completed_vehicles)
        throughput_per_hour = (throughput / self.current_time) * 3600 if self.current_time > 0 else 0

        # Tiempo total de viaje
        total_travel_time = sum(v.get_travel_time(v.arrival_time or self.current_time)
                               for v in self.completed_vehicles)

        return {
            'avg_delay': avg_delay,
            'avg_waiting_time': avg_waiting_time,
            'avg_queue_length': avg_queue_length,
            'max_queue_length': max_queue_length,
            'throughput': throughput,
            'throughput_per_hour': throughput_per_hour,
            'avg_stops': avg_stops,
            'total_travel_time': total_travel_time,
            'avg_speed_kmh': avg_speed,
            'computation_time': computation_time,
            'vehicles_completed': len(self.completed_vehicles),
            'vehicles_active': len(self.active_vehicles),
            'vehicles_generated': self.traffic_generator.total_vehicles_generated,
            'simulation_time': self.current_time
        }

    def _print_progress(self):
        """Imprime progreso de la simulación."""
        print(f"\n[T={self.current_time:6.0f}s] "
              f"Activos: {len(self.active_vehicles):3d} | "
              f"Completados: {len(self.completed_vehicles):3d} | "
              f"Generados: {self.traffic_generator.total_vehicles_generated:3d}")

    def _print_summary(self, metrics: Dict):
        """
        Imprime resumen de métricas finales.

        Args:
            metrics: Diccionario de métricas
        """
        print(f"\nVehículos:")
        print(f"  Generados:   {metrics['vehicles_generated']}")
        print(f"  Completados: {metrics['vehicles_completed']}")
        print(f"  Activos:     {metrics['vehicles_active']}")
        print(f"  Throughput:  {metrics['throughput_per_hour']:.1f} veh/hora")

        print(f"\nTiempos:")
        print(f"  Retraso promedio:     {metrics['avg_delay']:.2f} s")
        print(f"  Espera promedio:      {metrics['avg_waiting_time']:.2f} s")
        print(f"  Velocidad promedio:   {metrics['avg_speed_kmh']:.2f} km/h")

        print(f"\nColas:")
        print(f"  Longitud promedio:    {metrics['avg_queue_length']:.2f} vehículos")
        print(f"  Longitud máxima:      {metrics['max_queue_length']} vehículos")
        print(f"  Paradas promedio:     {metrics['avg_stops']:.2f} paradas/vehículo")

        print(f"\nRendimiento:")
        print(f"  Tiempo de simulación: {metrics['simulation_time']:.0f} s")
        print(f"  Tiempo de cómputo:    {metrics['computation_time']:.2f} s")
        if metrics['computation_time'] > 0:
            speedup = metrics['simulation_time'] / metrics['computation_time']
            print(f"  Speedup:              {speedup:.1f}x")

    def reset(self):
        """Reinicia el simulador al estado inicial."""
        self.current_time = 0.0
        self.active_vehicles.clear()
        self.completed_vehicles.clear()
        self.metrics_history.clear()
        self.queue_length_history.clear()

        # Reiniciar generador
        self.traffic_generator.reset()

        # Reiniciar semáforos
        for traffic_light in self.traffic_lights.values():
            traffic_light.reset()

        # Limpiar colas en intersecciones
        for intersection in self.network.intersections.values():
            intersection.clear_queues()

        # Limpiar vehículos de segmentos
        for segment in self.network.segments.values():
            segment.vehicles.clear()

    def get_current_state(self) -> Dict:
        """
        Retorna el estado actual completo de la simulación.

        Returns:
            dict: Estado actual
        """
        return {
            'time': self.current_time,
            'active_vehicles': len(self.active_vehicles),
            'completed_vehicles': len(self.completed_vehicles),
            'traffic_lights': {
                int_id: {
                    'phase': light.get_current_phase().name,
                    'time_in_phase': light.time_in_current_phase,
                    'cycles_completed': light.total_cycles_completed
                }
                for int_id, light in self.traffic_lights.items()
            },
            'queues': {
                int_id: intersection.get_total_queue_length()
                for int_id, intersection in self.network.intersections.items()
            }
        }


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.utils.config import INTERSECTIONS_FILE, MEDIUM_FLOW_FILE

    print("="*70)
    print("EJEMPLO: Simulador de Tráfico")
    print("="*70)

    # Cargar red y escenario
    network = TrafficNetwork(str(INTERSECTIONS_FILE))
    scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))

    # Crear simulador
    simulator = TrafficSimulator(network, scenario)

    # Ejecutar simulación de 5 minutos
    metrics = simulator.run(duration=300, verbose=True)

    print("\n" + "="*70)
    print("MÉTRICAS DETALLADAS")
    print("="*70)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:25s}: {value:.2f}")
        else:
            print(f"  {key:25s}: {value}")
