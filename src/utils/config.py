"""
Configuración global del sistema de optimización de semáforos.

Este módulo contiene todas las constantes y parámetros de configuración
utilizados en el proyecto.
"""

import os
from pathlib import Path

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MONTEVIDEO_DIR = DATA_DIR / "montevideo"
SCENARIOS_DIR = DATA_DIR / "scenarios"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

# Archivos de datos
INTERSECTIONS_FILE = MONTEVIDEO_DIR / "pocitos_intersections.json"
LOW_FLOW_FILE = SCENARIOS_DIR / "low_flow.json"
MEDIUM_FLOW_FILE = SCENARIOS_DIR / "medium_flow.json"
HIGH_FLOW_FILE = SCENARIOS_DIR / "high_flow.json"


# Parámetros del simulador
class SimulatorConfig:
    """Configuración del simulador de tráfico."""

    # Tiempo
    TIME_STEP = 1.0  # Paso de simulación en segundos
    DEFAULT_SIMULATION_DURATION = 3600  # 1 hora en segundos

    # Vehículos
    VEHICLE_LENGTH = 4.5  # metros
    VEHICLE_WIDTH = 2.0   # metros
    MIN_SAFE_DISTANCE = 2.0  # metros

    # Velocidades
    MAX_SPEED_KMH = 45  # km/h
    MAX_SPEED_MS = MAX_SPEED_KMH / 3.6  # m/s
    ACCELERATION = 2.0  # m/s²
    DECELERATION = 3.5  # m/s²

    # Comportamiento
    REACTION_TIME = 1.0  # segundos
    YELLOW_LIGHT_DURATION = 3.0  # segundos
    ALL_RED_DURATION = 1.0  # segundos (tiempo de despeje)


# Parámetros de semáforos
class TrafficLightConfig:
    """Configuración de semáforos."""

    # Ciclo de semáforo
    MIN_CYCLE_LENGTH = 60  # segundos
    MAX_CYCLE_LENGTH = 120  # segundos
    DEFAULT_CYCLE_LENGTH = 90  # segundos

    # Duraciones de fases
    MIN_GREEN_TIME = 10  # segundos
    MAX_GREEN_TIME = 60  # segundos
    DEFAULT_GREEN_TIME = 30  # segundos

    YELLOW_TIME = 3  # segundos
    ALL_RED_TIME = 1  # segundos (despeje)

    # Fases
    PHASES = ["north_south", "east_west", "left_turns"]

    # Offsets para coordinación (Branch & Bound)
    MIN_OFFSET = 0
    MAX_OFFSET = 120


# Parámetros de Programación Dinámica
class DynamicProgrammingConfig:
    """Configuración del algoritmo de Programación Dinámica."""

    TIME_GRANULARITY = 5  # Granularidad de tiempo en segundos
    MAX_STATES = 10000  # Máximo número de estados a explorar


# Parámetros de Branch & Bound
class BranchAndBoundConfig:
    """Configuración del algoritmo de Branch & Bound."""

    MAX_DEPTH = 10  # Profundidad máxima del árbol
    TIME_LIMIT = 300  # Tiempo límite en segundos
    OFFSET_STEP = 5  # Paso de offset en segundos


# Parámetros de Algoritmos Genéticos
class GeneticAlgorithmConfig:
    """Configuración del algoritmo genético."""

    POPULATION_SIZE = 100
    GENERATIONS = 200

    # Operadores genéticos
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.8

    # Selección
    TOURNAMENT_SIZE = 5
    ELITISM_COUNT = 5  # Mejores individuos que pasan directamente

    # Codificación
    # Cada individuo: [green_ns_1, green_ew_1, offset_1, green_ns_2, ...]
    GENE_MIN_VALUE = 10
    GENE_MAX_VALUE = 60


# Parámetros de Max-Pressure
class MaxPressureConfig:
    """Configuración de la heurística Max-Pressure."""

    MIN_PHASE_DURATION = 10  # segundos mínimos por fase
    PRESSURE_THRESHOLD = 5  # Umbral para cambiar de fase


# Métricas de evaluación
class MetricsConfig:
    """Configuración de métricas de evaluación."""

    METRICS = [
        "avg_delay",           # Retraso promedio por vehículo (s)
        "avg_queue_length",    # Longitud media de cola
        "throughput",          # Vehículos procesados por hora
        "avg_stops",           # Paradas promedio por vehículo
        "max_queue_length",    # Longitud máxima de cola
        "total_travel_time",   # Tiempo total de viaje
        "computation_time",    # Tiempo de cómputo del algoritmo
    ]


# Visualización
class VisualizationConfig:
    """Configuración de visualización."""

    FIGURE_SIZE = (12, 8)
    DPI = 100
    SAVE_FORMAT = "png"

    # Colores de semáforos
    LIGHT_COLORS = {
        "green": "#00FF00",
        "yellow": "#FFFF00",
        "red": "#FF0000",
        "off": "#888888"
    }

    # Colores de algoritmos
    ALGORITHM_COLORS = {
        "dynamic_programming": "#1f77b4",
        "branch_and_bound": "#ff7f0e",
        "genetic_algorithm": "#2ca02c",
        "max_pressure": "#d62728",
        "fixed_time": "#9467bd"
    }


# Logging
class LoggingConfig:
    """Configuración de logging."""

    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = PROJECT_ROOT / "simulation.log"


# Crear directorios si no existen
def ensure_directories():
    """Crea los directorios necesarios si no existen."""
    for directory in [DATA_DIR, MONTEVIDEO_DIR, SCENARIOS_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"Directorio del proyecto: {PROJECT_ROOT}")
    print(f"Directorio de datos: {DATA_DIR}")
    print(f"Archivo de intersecciones: {INTERSECTIONS_FILE}")
    ensure_directories()
    print("Directorios verificados/creados correctamente")
