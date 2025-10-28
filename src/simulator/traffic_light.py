"""
Modelo de semáforo con fases y comportamiento realista.

Este módulo implementa el comportamiento de un semáforo multi-fase,
incluyendo tiempos de verde, amarillo, todo-rojo, y coordinación
mediante offsets para ondas verdes.
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class LightState(Enum):
    """Estados posibles de un semáforo."""
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    OFF = "off"


class TrafficLightPhase:
    """
    Representa una fase del semáforo.

    Una fase es un período durante el cual ciertas direcciones
    tienen luz verde mientras otras tienen rojo.
    """

    def __init__(self, name: str, green_duration: int,
                 green_directions: List[str]):
        """
        Inicializa una fase del semáforo.

        Args:
            name: Nombre de la fase (ej: "north_south", "east_west")
            green_duration: Duración del verde en segundos
            green_directions: Lista de direcciones con verde en esta fase
                            (ej: ["north", "south"])
        """
        self.name = name
        self.green_duration = green_duration
        self.green_directions = green_directions
        self.yellow_duration = 3  # segundos (estándar)
        self.all_red_duration = 1  # segundo de despeje

        # Validación
        if green_duration < 10:
            raise ValueError(f"Duración de verde muy corta: {green_duration}s (mín: 10s)")
        if green_duration > 90:
            raise ValueError(f"Duración de verde muy larga: {green_duration}s (máx: 90s)")

    @property
    def total_duration(self) -> int:
        """Duración total de la fase (verde + amarillo + todo-rojo)."""
        return self.green_duration + self.yellow_duration + self.all_red_duration

    def get_state_at_time(self, time_in_phase: float) -> LightState:
        """
        Retorna el estado del semáforo en un momento dado de la fase.

        Args:
            time_in_phase: Tiempo transcurrido desde el inicio de la fase (segundos)

        Returns:
            LightState: Estado del semáforo (GREEN, YELLOW, RED)
        """
        if time_in_phase < 0:
            return LightState.RED
        elif time_in_phase < self.green_duration:
            return LightState.GREEN
        elif time_in_phase < self.green_duration + self.yellow_duration:
            return LightState.YELLOW
        else:
            return LightState.RED

    def __str__(self) -> str:
        return f"Phase({self.name}: {self.green_duration}s green, {self.green_directions})"

    def __repr__(self) -> str:
        return (f"TrafficLightPhase(name='{self.name}', "
                f"green={self.green_duration}s, "
                f"total={self.total_duration}s)")


class TrafficLight:
    """
    Representa un semáforo completo con múltiples fases.

    El semáforo controla el flujo vehicular en una intersección,
    alternando entre diferentes fases para permitir el paso de
    vehículos en distintas direcciones.
    """

    def __init__(self, intersection_id: int, intersection_name: str = ""):
        """
        Inicializa un semáforo.

        Args:
            intersection_id: ID de la intersección que controla
            intersection_name: Nombre de la intersección (opcional)
        """
        self.intersection_id = intersection_id
        self.intersection_name = intersection_name

        # Fases del semáforo
        self.phases: List[TrafficLightPhase] = []
        self.current_phase_index = 0
        self.time_in_current_phase = 0.0

        # Offset para coordinación (ondas verdes)
        self.offset = 0  # segundos de desfase respecto a t=0

        # Estado
        self.is_active = True
        self.start_time = 0.0

        # Configuración por defecto: 2 fases simples
        self._set_default_configuration()

        # Estadísticas
        self.total_cycles_completed = 0
        self.phase_change_history = []

    def _set_default_configuration(self):
        """
        Configura el semáforo con una configuración estándar de 2 fases.

        Fase 1: Norte-Sur (30s verde)
        Fase 2: Este-Oeste (30s verde)
        Ciclo total: ~68s
        """
        self.phases = [
            TrafficLightPhase(
                name="north_south",
                green_duration=30,
                green_directions=["north", "south"]
            ),
            TrafficLightPhase(
                name="east_west",
                green_duration=30,
                green_directions=["east", "west"]
            )
        ]

    def set_configuration(self, phase_configs: Dict[str, int], offset: int = 0):
        """
        Configura las duraciones de las fases del semáforo.

        Args:
            phase_configs: Diccionario {nombre_fase: duración_verde_segundos}
                          Ej: {"north_south": 35, "east_west": 40, "left_turns": 15}
            offset: Desfase en segundos para coordinación

        Raises:
            ValueError: Si la configuración no es válida
        """
        self.phases = []

        # Configurar fases según el diccionario
        if "north_south" in phase_configs:
            self.phases.append(TrafficLightPhase(
                name="north_south",
                green_duration=phase_configs["north_south"],
                green_directions=["north", "south"]
            ))

        if "east_west" in phase_configs:
            self.phases.append(TrafficLightPhase(
                name="east_west",
                green_duration=phase_configs["east_west"],
                green_directions=["east", "west"]
            ))

        if "left_turns" in phase_configs:
            self.phases.append(TrafficLightPhase(
                name="left_turns",
                green_duration=phase_configs["left_turns"],
                green_directions=["left_north", "left_south", "left_east", "left_west"]
            ))

        if not self.phases:
            raise ValueError("Debe configurarse al menos una fase")

        # Configurar offset
        self.offset = max(0, min(offset, 120))  # Limitar offset entre 0 y 120s

        # Resetear estado
        self.current_phase_index = 0
        self.time_in_current_phase = 0.0

    def update(self, current_time: float, dt: float = 1.0):
        """
        Actualiza el estado del semáforo.

        Args:
            current_time: Tiempo actual de simulación (segundos)
            dt: Paso de tiempo (segundos)
        """
        if not self.is_active or not self.phases:
            return

        # Ajustar por offset (el semáforo empieza en offset segundos)
        adjusted_time = current_time - self.offset
        if adjusted_time < 0:
            # El semáforo aún no ha empezado
            return

        # Incrementar tiempo en fase actual
        self.time_in_current_phase += dt

        # Verificar si es momento de cambiar de fase
        current_phase = self.phases[self.current_phase_index]
        if self.time_in_current_phase >= current_phase.total_duration:
            # Cambiar a siguiente fase
            self.current_phase_index = (self.current_phase_index + 1) % len(self.phases)
            self.time_in_current_phase = 0.0

            # Registrar cambio
            self.phase_change_history.append({
                'time': current_time,
                'phase': self.phases[self.current_phase_index].name
            })

            # Contar ciclos completos
            if self.current_phase_index == 0:
                self.total_cycles_completed += 1

    def get_current_phase(self) -> TrafficLightPhase:
        """Retorna la fase actual del semáforo."""
        if not self.phases:
            raise RuntimeError("Semáforo sin fases configuradas")
        return self.phases[self.current_phase_index]

    def get_state(self, direction: str) -> LightState:
        """
        Retorna el estado del semáforo para una dirección específica.

        Args:
            direction: Dirección del vehículo ("north", "south", "east", "west")

        Returns:
            LightState: Estado del semáforo (GREEN, YELLOW, RED)
        """
        if not self.is_active or not self.phases:
            return LightState.OFF

        current_phase = self.get_current_phase()
        state_in_phase = current_phase.get_state_at_time(self.time_in_current_phase)

        # Si la fase actual permite esta dirección y estamos en verde/amarillo
        if direction in current_phase.green_directions:
            return state_in_phase
        else:
            # Cualquier otra dirección está en rojo
            return LightState.RED

    def can_vehicle_pass(self, direction: str) -> bool:
        """
        Determina si un vehículo puede pasar el semáforo.

        Args:
            direction: Dirección del vehículo

        Returns:
            bool: True si puede pasar (verde), False en caso contrario
        """
        return self.get_state(direction) == LightState.GREEN

    def get_time_until_green(self, direction: str) -> float:
        """
        Calcula cuánto tiempo falta para que una dirección tenga verde.

        Args:
            direction: Dirección a consultar

        Returns:
            float: Segundos hasta el próximo verde (0 si ya está en verde)
        """
        if not self.phases:
            return float('inf')

        current_phase = self.get_current_phase()

        # Si ya está en verde, retornar 0
        if direction in current_phase.green_directions:
            if current_phase.get_state_at_time(self.time_in_current_phase) == LightState.GREEN:
                return 0.0

        # Calcular tiempo hasta próximo verde
        time_remaining_in_phase = current_phase.total_duration - self.time_in_current_phase
        time_to_green = time_remaining_in_phase

        # Buscar en fases siguientes
        next_phase_index = (self.current_phase_index + 1) % len(self.phases)
        searched_phases = 0

        while searched_phases < len(self.phases):
            phase = self.phases[next_phase_index]

            if direction in phase.green_directions:
                # Encontramos la fase, retornar tiempo acumulado
                return time_to_green

            # No está en esta fase, sumar su duración y continuar
            time_to_green += phase.total_duration
            next_phase_index = (next_phase_index + 1) % len(self.phases)
            searched_phases += 1

        # Si no encontramos la dirección en ninguna fase
        return float('inf')

    def get_cycle_length(self) -> int:
        """
        Retorna la duración total del ciclo completo del semáforo.

        Returns:
            int: Segundos para completar todas las fases
        """
        return sum(phase.total_duration for phase in self.phases)

    def get_efficiency_ratio(self, direction: str) -> float:
        """
        Calcula el ratio de eficiencia para una dirección.

        El ratio es la fracción del ciclo durante la cual esa dirección tiene verde.

        Args:
            direction: Dirección a analizar

        Returns:
            float: Ratio entre 0.0 y 1.0
        """
        cycle_length = self.get_cycle_length()
        if cycle_length == 0:
            return 0.0

        green_time = 0
        for phase in self.phases:
            if direction in phase.green_directions:
                green_time += phase.green_duration

        return green_time / cycle_length

    def reset(self, current_time: float = 0.0):
        """
        Reinicia el semáforo al inicio del ciclo.

        Args:
            current_time: Tiempo actual de simulación
        """
        self.current_phase_index = 0
        self.time_in_current_phase = 0.0
        self.start_time = current_time
        self.total_cycles_completed = 0
        self.phase_change_history.clear()

    def get_status_string(self) -> str:
        """
        Retorna una representación visual del estado actual.

        Returns:
            str: String con estado formateado
        """
        if not self.phases:
            return "⚫ SEMÁFORO APAGADO"

        current_phase = self.get_current_phase()
        state = current_phase.get_state_at_time(self.time_in_current_phase)

        # Símbolos para cada estado
        symbols = {
            LightState.GREEN: "🟢",
            LightState.YELLOW: "🟡",
            LightState.RED: "🔴",
            LightState.OFF: "⚫"
        }

        symbol = symbols.get(state, "⚫")

        status = f"{symbol} Fase: {current_phase.name} | "
        status += f"Estado: {state.value.upper()} | "
        status += f"Tiempo en fase: {self.time_in_current_phase:.1f}s / {current_phase.total_duration}s | "
        status += f"Ciclo: {self.total_cycles_completed}"

        return status

    def visualize_current_state(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Crea una visualización del estado actual del semáforo.

        Args:
            figsize: Tamaño de la figura

        Returns:
            plt.Figure: Figura de matplotlib
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Panel 1: Estado actual del semáforo
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.axis('off')
        ax1.set_title(f"Semáforo {self.intersection_id}\n{self.intersection_name}",
                     fontsize=12, fontweight='bold')

        # Dibujar semáforo
        colors = {'green': '#00FF00', 'yellow': '#FFFF00', 'red': '#FF0000', 'off': '#333333'}

        current_state = self.get_current_phase().get_state_at_time(self.time_in_current_phase)

        for i, state in enumerate([LightState.RED, LightState.YELLOW, LightState.GREEN]):
            y_pos = 7 - i * 2
            color = colors[state.value] if state == current_state else colors['off']
            circle = patches.Circle((5, y_pos), 0.7, color=color, ec='black', linewidth=2)
            ax1.add_patch(circle)

        # Texto de estado
        status_text = self.get_status_string()
        ax1.text(5, 1, status_text, ha='center', va='top', fontsize=8, wrap=True)

        # Panel 2: Diagrama de fases
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, len(self.phases))
        ax2.set_title("Configuración de Fases", fontsize=12, fontweight='bold')
        ax2.set_yticks(range(len(self.phases)))
        ax2.set_yticklabels([f"Fase {i+1}: {p.name}" for i, p in enumerate(self.phases)])
        ax2.set_xlabel("Duración (s)")

        # Dibujar barras de fases
        for i, phase in enumerate(self.phases):
            # Verde
            green_width = phase.green_duration / self.get_cycle_length()
            ax2.barh(i, green_width, left=0, height=0.6, color='green', alpha=0.7)

            # Amarillo
            yellow_width = phase.yellow_duration / self.get_cycle_length()
            ax2.barh(i, yellow_width, left=green_width, height=0.6, color='yellow', alpha=0.7)

            # Todo-rojo
            red_width = phase.all_red_duration / self.get_cycle_length()
            ax2.barh(i, red_width, left=green_width + yellow_width, height=0.6,
                    color='red', alpha=0.7)

            # Resaltar fase actual
            if i == self.current_phase_index:
                ax2.barh(i, 1, height=0.8, fill=False, edgecolor='blue', linewidth=3)

            # Etiqueta de duración
            ax2.text(0.5, i, f"{phase.green_duration}s", ha='center', va='center',
                    fontweight='bold')

        ax2.set_xlim(0, 1.1)
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        return fig

    def __str__(self) -> str:
        phase_names = [p.name for p in self.phases]
        return f"TrafficLight({self.intersection_id}, phases={phase_names})"

    def __repr__(self) -> str:
        return (f"TrafficLight(id={self.intersection_id}, "
                f"phases={len(self.phases)}, "
                f"cycle={self.get_cycle_length()}s, "
                f"offset={self.offset}s)")


if __name__ == "__main__":
    # Ejemplo de uso
    print("="*70)
    print("EJEMPLO: Semáforo Básico")
    print("="*70)

    # Crear semáforo
    light = TrafficLight(intersection_id=1, intersection_name="Av. Brasil y Rambla")

    # Configurar fases personalizadas
    light.set_configuration({
        "north_south": 35,
        "east_west": 40,
        "left_turns": 15
    }, offset=0)

    print(f"\n{light}")
    print(f"Ciclo completo: {light.get_cycle_length()} segundos")

    # Simular avance del tiempo
    print("\n" + "="*70)
    print("SIMULACIÓN DE TIEMPO")
    print("="*70)

    for t in [0, 20, 35, 38, 40, 75, 78, 90]:
        light.update(t, dt=1.0)
        print(f"\nT={t:3d}s: {light.get_status_string()}")
        print(f"  Norte puede pasar: {light.can_vehicle_pass('north')}")
        print(f"  Este puede pasar:  {light.can_vehicle_pass('east')}")

    # Ratios de eficiencia
    print("\n" + "="*70)
    print("RATIOS DE EFICIENCIA")
    print("="*70)
    for direction in ['north', 'east']:
        ratio = light.get_efficiency_ratio(direction)
        print(f"  {direction.capitalize()}: {ratio:.2%} del ciclo")
