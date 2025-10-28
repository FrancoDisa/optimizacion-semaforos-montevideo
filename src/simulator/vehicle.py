"""
Modelo de vehículo con física y comportamiento realista.

Este módulo implementa el comportamiento de un vehículo individual,
incluyendo física de movimiento, respuesta a semáforos, y seguimiento
de ruta en la red vial.
"""

from typing import List, Optional, Tuple
from enum import Enum
import math


class VehicleState(Enum):
    """Estados posibles de un vehículo."""
    MOVING = "moving"              # Moviéndose normalmente
    STOPPED_AT_LIGHT = "stopped"   # Detenido en semáforo rojo
    WAITING_IN_QUEUE = "waiting"   # Esperando en cola
    ACCELERATING = "accelerating"  # Acelerando
    BRAKING = "braking"            # Frenando
    ARRIVED = "arrived"            # Llegó a destino


class Vehicle:
    """
    Representa un vehículo individual en la simulación.

    El vehículo tiene física realista, responde a semáforos,
    sigue una ruta predeterminada, y acumula estadísticas de
    su viaje.
    """

    # Contador global para IDs únicos
    _next_id = 1

    def __init__(self, origin: int, destination: int, spawn_time: float,
                 route: List[int], max_speed_ms: float = 12.5):
        """
        Inicializa un vehículo.

        Args:
            origin: ID de intersección de origen
            destination: ID de intersección de destino
            spawn_time: Tiempo de generación del vehículo (segundos)
            route: Lista de IDs de intersecciones formando la ruta
            max_speed_ms: Velocidad máxima en m/s (default: 12.5 m/s = 45 km/h)
        """
        # Identificación
        self.id = Vehicle._next_id
        Vehicle._next_id += 1

        # Ruta y ubicación
        self.origin = origin
        self.destination = destination
        self.route = route
        self.current_edge_index = 0  # Índice en la ruta (qué segmento está recorriendo)
        self.position_on_edge = 0.0  # Metros desde el inicio del segmento actual

        # Física del vehículo
        self.length = 4.5  # metros (vehículo estándar)
        self.width = 2.0   # metros
        self.max_speed = max_speed_ms  # m/s
        self.current_speed = 0.0  # m/s
        self.acceleration = 2.0  # m/s² (aceleración normal)
        self.max_deceleration = 3.5  # m/s² (frenado de emergencia)
        self.comfortable_deceleration = 2.0  # m/s² (frenado normal)

        # Comportamiento
        self.reaction_time = 1.0  # segundos
        self.safe_distance = 2.0  # metros mínimos con vehículo adelante
        self.stop_distance_from_light = 1.0  # metros antes del semáforo

        # Estado
        self.state = VehicleState.MOVING
        self.spawn_time = spawn_time
        self.arrival_time: Optional[float] = None

        # Estadísticas
        self.total_waiting_time = 0.0  # segundos detenido
        self.num_stops = 0  # número de veces que se detuvo completamente
        self.distance_traveled = 0.0  # metros recorridos
        self.time_stopped_at_lights = 0.0  # tiempo específico en semáforos

        # Referencias externas (se asignan durante la simulación)
        self.current_segment = None  # RoadSegment actual
        self.vehicle_ahead = None  # Vehículo adelante (si existe)
        self.next_traffic_light = None  # Próximo semáforo

        # Historial de estados (para análisis)
        self.state_history = []
        self._was_stopped = False  # Para detectar nuevas paradas

    def get_current_from_intersection(self) -> int:
        """Retorna el ID de la intersección de origen del segmento actual."""
        if self.has_arrived():
            return self.destination
        return self.route[self.current_edge_index]

    def get_current_to_intersection(self) -> int:
        """Retorna el ID de la intersección de destino del segmento actual."""
        if self.has_arrived():
            return self.destination
        return self.route[self.current_edge_index + 1]

    def get_direction(self) -> str:
        """
        Determina la dirección de movimiento del vehículo.

        Returns:
            str: Dirección cardinal ("north", "south", "east", "west")
        """
        # En una implementación real, esto se calcularía basándose en
        # las coordenadas GPS de las intersecciones.
        # Por simplicidad, usamos una heurística basada en los IDs.

        if self.has_arrived():
            return "none"

        from_id = self.get_current_from_intersection()
        to_id = self.get_current_to_intersection()

        # Heurística simple: si to_id > from_id, vamos "hacia adelante"
        # En Av. Brasil, esto típicamente significa oeste
        if to_id > from_id:
            return "west"  # Alejándose de la rambla
        else:
            return "east"  # Hacia la rambla

    def update(self, dt: float, current_time: float):
        """
        Actualiza el estado del vehículo.

        Este es el método principal que se llama en cada paso de simulación.

        Args:
            dt: Paso de tiempo (segundos)
            current_time: Tiempo actual de simulación (segundos)
        """
        if self.has_arrived():
            return

        # 1. Determinar velocidad objetivo basándose en condiciones
        target_speed = self._calculate_target_speed()

        # 2. Ajustar velocidad actual hacia la objetivo
        self._adjust_speed(target_speed, dt)

        # 3. Actualizar posición
        distance_moved = self.current_speed * dt
        self.position_on_edge += distance_moved
        self.distance_traveled += distance_moved

        # 4. Verificar si llegamos al final del segmento
        if self.current_segment and self.position_on_edge >= self.current_segment.length_m:
            self._move_to_next_segment()

        # 5. Actualizar estadísticas
        self._update_statistics(dt, current_time)

        # 6. Registrar estado en historial
        self.state_history.append({
            'time': current_time,
            'position': self.position_on_edge,
            'speed': self.current_speed,
            'state': self.state.value
        })

    def _calculate_target_speed(self) -> float:
        """
        Calcula la velocidad objetivo basándose en las condiciones actuales.

        Considera:
        - Semáforos adelante
        - Vehículos adelante
        - Límite de velocidad
        - Distancia al final del segmento

        Returns:
            float: Velocidad objetivo en m/s
        """
        # Por defecto, queremos ir a velocidad máxima
        target_speed = self.max_speed

        # 1. Verificar semáforo adelante
        if self.next_traffic_light:
            distance_to_light = self._get_distance_to_next_light()

            # Si el semáforo está rojo o amarillo
            if not self.next_traffic_light.can_vehicle_pass(self.get_direction()):
                # Calcular si necesitamos frenar
                stopping_distance = self._calculate_stopping_distance()

                if distance_to_light <= stopping_distance:
                    # Necesitamos frenar
                    target_speed = 0.0
                    self.state = VehicleState.BRAKING
                elif distance_to_light <= stopping_distance * 2:
                    # Reducir velocidad preventivamente
                    target_speed = min(target_speed, self.max_speed * 0.5)

        # 2. Verificar vehículo adelante
        if self.vehicle_ahead:
            distance_to_vehicle = self._get_distance_to_vehicle_ahead()

            if distance_to_vehicle < self.safe_distance + 5:
                # Muy cerca, igualar velocidad del vehículo adelante
                target_speed = min(target_speed, self.vehicle_ahead.current_speed)
            elif distance_to_vehicle < self.safe_distance + 15:
                # Cerca, reducir velocidad
                target_speed = min(target_speed, self.max_speed * 0.7)

        # 3. Verificar distancia al final del segmento
        if self.current_segment:
            distance_to_end = self.current_segment.length_m - self.position_on_edge
            if distance_to_end < 20:  # Últimos 20 metros
                # Reducir velocidad al acercarse a intersección
                target_speed = min(target_speed, self.max_speed * 0.6)

        return max(0.0, target_speed)

    def _adjust_speed(self, target_speed: float, dt: float):
        """
        Ajusta la velocidad actual hacia la velocidad objetivo.

        Args:
            target_speed: Velocidad objetivo en m/s
            dt: Paso de tiempo en segundos
        """
        speed_diff = target_speed - self.current_speed

        if abs(speed_diff) < 0.1:
            # Ya estamos en la velocidad objetivo
            self.current_speed = target_speed
            if target_speed == 0:
                self.state = VehicleState.STOPPED_AT_LIGHT
            elif target_speed > 0:
                self.state = VehicleState.MOVING
            return

        if speed_diff > 0:
            # Necesitamos acelerar
            self.state = VehicleState.ACCELERATING
            acceleration = min(self.acceleration, speed_diff / dt)
            self.current_speed += acceleration * dt
        else:
            # Necesitamos frenar
            self.state = VehicleState.BRAKING
            deceleration = min(self.comfortable_deceleration, -speed_diff / dt)
            self.current_speed -= deceleration * dt

        # Asegurar límites
        self.current_speed = max(0.0, min(self.current_speed, self.max_speed))

    def _calculate_stopping_distance(self) -> float:
        """
        Calcula la distancia necesaria para detenerse completamente.

        Usa cinemática: d = v² / (2a) + v * t_reacción

        Returns:
            float: Distancia de frenado en metros
        """
        if self.current_speed == 0:
            return 0.0

        # Distancia durante tiempo de reacción
        reaction_distance = self.current_speed * self.reaction_time

        # Distancia de frenado
        braking_distance = (self.current_speed ** 2) / (2 * self.comfortable_deceleration)

        return reaction_distance + braking_distance + self.stop_distance_from_light

    def _get_distance_to_next_light(self) -> float:
        """
        Calcula la distancia al próximo semáforo.

        Returns:
            float: Distancia en metros (infinito si no hay semáforo)
        """
        if not self.current_segment:
            return float('inf')

        # Distancia = lo que falta del segmento actual
        distance = self.current_segment.length_m - self.position_on_edge

        return distance

    def _get_distance_to_vehicle_ahead(self) -> float:
        """
        Calcula la distancia al vehículo adelante.

        Returns:
            float: Distancia en metros (infinito si no hay vehículo)
        """
        if not self.vehicle_ahead:
            return float('inf')

        # Si está en el mismo segmento
        if (self.vehicle_ahead.current_segment == self.current_segment and
            self.vehicle_ahead.position_on_edge > self.position_on_edge):
            return self.vehicle_ahead.position_on_edge - self.position_on_edge - self.length

        # En otro caso, asumir distancia grande
        return float('inf')

    def _move_to_next_segment(self):
        """Avanza al siguiente segmento de la ruta."""
        self.current_edge_index += 1
        self.position_on_edge = 0.0

        # Verificar si hemos llegado al destino
        if self.current_edge_index >= len(self.route) - 1:
            self.state = VehicleState.ARRIVED
            # arrival_time se establece externamente por el simulador

    def _update_statistics(self, dt: float, current_time: float):
        """
        Actualiza estadísticas del vehículo.

        Args:
            dt: Paso de tiempo
            current_time: Tiempo actual
        """
        # Detectar paradas (velocidad ~0)
        is_stopped = self.current_speed < 0.1

        if is_stopped:
            self.total_waiting_time += dt

            # Contar nueva parada
            if not self._was_stopped:
                self.num_stops += 1
                self._was_stopped = True

            # Si está detenido por semáforo
            if self.state == VehicleState.STOPPED_AT_LIGHT:
                self.time_stopped_at_lights += dt
        else:
            self._was_stopped = False

    def has_arrived(self) -> bool:
        """Verifica si el vehículo llegó a su destino."""
        return self.state == VehicleState.ARRIVED

    def get_travel_time(self, current_time: float) -> float:
        """
        Calcula el tiempo total de viaje.

        Args:
            current_time: Tiempo actual de simulación

        Returns:
            float: Tiempo de viaje en segundos
        """
        if self.has_arrived() and self.arrival_time:
            return self.arrival_time - self.spawn_time
        else:
            return current_time - self.spawn_time

    def get_average_speed_kmh(self) -> float:
        """
        Calcula la velocidad promedio del viaje.

        Returns:
            float: Velocidad promedio en km/h
        """
        if self.distance_traveled == 0:
            return 0.0

        # Velocidad = distancia / tiempo
        travel_time = self.get_travel_time(
            self.arrival_time if self.arrival_time else self.spawn_time
        )

        if travel_time == 0:
            return 0.0

        avg_speed_ms = self.distance_traveled / travel_time
        return avg_speed_ms * 3.6  # Convertir a km/h

    def get_delay(self, ideal_travel_time: float) -> float:
        """
        Calcula el retraso respecto al tiempo ideal.

        Args:
            ideal_travel_time: Tiempo de viaje sin semáforos ni tráfico (segundos)

        Returns:
            float: Retraso en segundos
        """
        actual_time = self.get_travel_time(
            self.arrival_time if self.arrival_time else self.spawn_time
        )
        return max(0.0, actual_time - ideal_travel_time)

    def get_statistics(self) -> dict:
        """
        Retorna un diccionario con todas las estadísticas del vehículo.

        Returns:
            dict: Estadísticas completas
        """
        return {
            'vehicle_id': self.id,
            'origin': self.origin,
            'destination': self.destination,
            'spawn_time': self.spawn_time,
            'arrival_time': self.arrival_time,
            'travel_time': self.get_travel_time(self.arrival_time or self.spawn_time),
            'distance_traveled': self.distance_traveled,
            'avg_speed_kmh': self.get_average_speed_kmh(),
            'total_waiting_time': self.total_waiting_time,
            'time_at_lights': self.time_stopped_at_lights,
            'num_stops': self.num_stops,
            'arrived': self.has_arrived()
        }

    def get_status_string(self) -> str:
        """
        Retorna una representación visual del estado actual.

        Returns:
            str: String con estado formateado
        """
        symbols = {
            VehicleState.MOVING: "🚗",
            VehicleState.STOPPED_AT_LIGHT: "🛑",
            VehicleState.WAITING_IN_QUEUE: "⏸️",
            VehicleState.ACCELERATING: "🏃",
            VehicleState.BRAKING: "🔽",
            VehicleState.ARRIVED: "✓"
        }

        symbol = symbols.get(self.state, "🚗")
        speed_kmh = self.current_speed * 3.6

        if self.has_arrived():
            return f"{symbol} Vehículo #{self.id} - ARRIBÓ AL DESTINO"

        status = f"{symbol} Vehículo #{self.id} | "
        status += f"Estado: {self.state.value.upper()} | "
        status += f"Velocidad: {speed_kmh:.1f} km/h | "
        status += f"Segmento: {self.get_current_from_intersection()}→{self.get_current_to_intersection()} | "
        status += f"Posición: {self.position_on_edge:.1f}m | "
        status += f"Paradas: {self.num_stops}"

        return status

    def __str__(self) -> str:
        return f"Vehicle(#{self.id}, {self.origin}→{self.destination})"

    def __repr__(self) -> str:
        return (f"Vehicle(id={self.id}, route={self.origin}→{self.destination}, "
                f"state={self.state.value}, speed={self.current_speed:.2f}m/s)")


if __name__ == "__main__":
    # Ejemplo de uso
    print("="*70)
    print("EJEMPLO: Vehículo en Simulación")
    print("="*70)

    # Crear vehículo con ruta
    route = [1, 2, 3, 4, 5]  # Intersecciones de Av. Brasil
    vehicle = Vehicle(
        origin=1,
        destination=5,
        spawn_time=0.0,
        route=route,
        max_speed_ms=12.5  # 45 km/h
    )

    print(f"\n{vehicle}")
    print(f"Ruta: {' → '.join(map(str, vehicle.route))}")
    print(f"Velocidad máxima: {vehicle.max_speed * 3.6:.1f} km/h")

    # Simular algunos pasos
    print("\n" + "="*70)
    print("SIMULACIÓN DE MOVIMIENTO")
    print("="*70)

    # Configurar un segmento ficticio
    from src.simulator.traffic_network import RoadSegment
    vehicle.current_segment = RoadSegment(1, 2, 280, 22, 2, "oeste")

    for t in range(0, 30, 5):
        vehicle.update(dt=1.0, current_time=t)
        print(f"\nT={t:3d}s: {vehicle.get_status_string()}")

    # Estadísticas finales
    print("\n" + "="*70)
    print("ESTADÍSTICAS")
    print("="*70)
    stats = vehicle.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
