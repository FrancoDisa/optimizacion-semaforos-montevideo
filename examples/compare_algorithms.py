"""
Script de comparación: Todos los algoritmos de optimización

Este script ejecuta los 4 algoritmos de optimización y compara sus resultados:
1. Max-Pressure (Baseline)
2. Programación Dinámica
3. Branch & Bound
4. Algoritmos Genéticos
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulator import TrafficNetwork, TrafficSimulator, TrafficScenario
from src.algorithms import (
    MaxPressureController,
    DynamicProgrammingOptimizer,
    BranchAndBoundCoordinator,
    GeneticOptimizer
)
from src.utils.config import INTERSECTIONS_FILE, MEDIUM_FLOW_FILE
from src.utils.metrics import MetricsCalculator
import time as timer


def run_baseline():
    """Ejecuta simulación con configuración baseline (tiempos fijos)."""
    print(f"\n{'='*70}")
    print("1. BASELINE - Tiempos Fijos Estándar")
    print(f"{'='*70}")

    network = TrafficNetwork(str(INTERSECTIONS_FILE))
    scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))
    simulator = TrafficSimulator(network, scenario)

    # Configuración baseline: todos iguales, sin coordinación
    baseline_config = {}
    for int_id in network.get_all_intersection_ids():
        baseline_config[int_id] = {
            'north_south': 30,
            'east_west': 30,
            'offset': 0
        }

    simulator.configure_traffic_lights(baseline_config)

    start = timer.time()
    metrics = simulator.run(duration=600, verbose=False)
    elapsed = timer.time() - start

    metrics['optimization_time'] = 0.0  # No hay optimización
    metrics['total_time'] = elapsed

    return metrics


def run_max_pressure():
    """Ejecuta simulación con controlador Max-Pressure."""
    print(f"\n{'='*70}")
    print("2. MAX-PRESSURE - Heurística en Tiempo Real")
    print(f"{'='*70}")

    network = TrafficNetwork(str(INTERSECTIONS_FILE))
    scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))
    simulator = TrafficSimulator(network, scenario)

    # Max-Pressure genera configuración estándar
    # La inteligencia está en cuándo cambiar de fase
    controller = MaxPressureController(network)

    start = timer.time()

    # Configuración básica (Max-Pressure la ajusta dinámicamente)
    mp_config = controller.generate_configuration({}, current_time=0.0)
    simulator.configure_traffic_lights(mp_config)

    metrics = simulator.run(duration=600, verbose=False)
    elapsed = timer.time() - start

    mp_stats = controller.get_statistics()
    print(f"\n  Estadísticas Max-Pressure:")
    print(f"    Cambios de fase: {mp_stats['phase_changes']}")
    print(f"    Decisiones tomadas: {mp_stats['decisions_made']}")

    metrics['optimization_time'] = 0.0  # Tiempo real, no offline
    metrics['total_time'] = elapsed

    return metrics


def run_dynamic_programming():
    """Ejecuta simulación con configuración de Programación Dinámica."""
    print(f"\n{'='*70}")
    print("3. PROGRAMACIÓN DINÁMICA - Optimización Individual")
    print(f"{'='*70}")

    network = TrafficNetwork(str(INTERSECTIONS_FILE))
    scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))
    simulator = TrafficSimulator(network, scenario)

    # Crear optimizador
    optimizer = DynamicProgrammingOptimizer(
        network,
        cycle_length=90,
        time_granularity=5
    )

    # Preparar datos de tráfico (estimados del escenario)
    traffic_data = {}
    for int_id in network.get_all_intersection_ids():
        traffic_data[int_id] = {
            'north_south_arrival_rate': 0.28,  # ~1000 veh/h
            'east_west_arrival_rate': 0.33     # ~1200 veh/h
        }

    # Optimizar
    opt_start = timer.time()
    dp_config = optimizer.optimize_network(traffic_data)
    opt_elapsed = timer.time() - opt_start

    print(f"\n  Tiempo de optimización: {opt_elapsed:.2f}s")

    # Simular con configuración optimizada
    simulator.configure_traffic_lights(dp_config)

    sim_start = timer.time()
    metrics = simulator.run(duration=600, verbose=False)
    sim_elapsed = timer.time() - sim_start

    metrics['optimization_time'] = opt_elapsed
    metrics['total_time'] = opt_elapsed + sim_elapsed

    dp_stats = optimizer.get_statistics()
    print(f"\n  Estadísticas DP:")
    print(f"    Optimizaciones: {dp_stats['optimizations_performed']}")

    return metrics


def run_branch_and_bound():
    """Ejecuta simulación con configuración de Branch & Bound."""
    print(f"\n{'='*70}")
    print("4. BRANCH & BOUND - Coordinación (Ondas Verdes)")
    print(f"{'='*70}")

    network = TrafficNetwork(str(INTERSECTIONS_FILE))
    scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))
    simulator = TrafficSimulator(network, scenario)

    # Primero usar DP para obtener tiempos base
    dp_optimizer = DynamicProgrammingOptimizer(network, cycle_length=90)
    traffic_data = {}
    for int_id in network.get_all_intersection_ids():
        traffic_data[int_id] = {
            'north_south_arrival_rate': 0.28,
            'east_west_arrival_rate': 0.33
        }

    print("  Optimizando tiempos individuales con DP...")
    base_config = dp_optimizer.optimize_network(traffic_data)

    # Ahora usar B&B para coordinar offsets
    print("  Coordinando offsets con Branch & Bound...")
    coordinator = BranchAndBoundCoordinator(
        network,
        simulator=None,  # Sin simulador para ser más rápido
        offset_step=10,
        time_limit=60.0
    )

    opt_start = timer.time()
    bb_config = coordinator.generate_configuration(base_config)
    opt_elapsed = timer.time() - opt_start

    # Simular con configuración optimizada
    simulator.configure_traffic_lights(bb_config)

    sim_start = timer.time()
    metrics = simulator.run(duration=600, verbose=False)
    sim_elapsed = timer.time() - sim_start

    metrics['optimization_time'] = opt_elapsed
    metrics['total_time'] = opt_elapsed + sim_elapsed

    bb_stats = coordinator.get_statistics()
    print(f"\n  Estadísticas Branch & Bound:")
    print(f"    Nodos explorados: {bb_stats['nodes_explored']}")
    print(f"    Nodos podados: {bb_stats['nodes_pruned']}")
    print(f"    Tasa de poda: {bb_stats['pruning_rate']:.1%}")

    return metrics


def run_genetic_algorithm():
    """Ejecuta simulación con configuración de Algoritmos Genéticos."""
    print(f"\n{'='*70}")
    print("5. ALGORITMOS GENÉTICOS - Optimización Global")
    print(f"{'='*70}")

    network = TrafficNetwork(str(INTERSECTIONS_FILE))
    scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))
    simulator = TrafficSimulator(network, scenario)

    # Crear optimizador genético
    optimizer = GeneticOptimizer(
        network=network,
        simulator=simulator,
        population_size=30,      # Reducido para demo
        generations=50,          # Reducido para demo
        mutation_rate=0.15,
        crossover_rate=0.8,
        elitism_count=3
    )

    # Evolucionar
    opt_start = timer.time()
    ga_config = optimizer.evolve()
    opt_elapsed = timer.time() - opt_start

    # Simular con mejor configuración
    simulator.configure_traffic_lights(ga_config)
    simulator.reset()

    sim_start = timer.time()
    metrics = simulator.run(duration=600, verbose=False)
    sim_elapsed = timer.time() - sim_start

    metrics['optimization_time'] = opt_elapsed
    metrics['total_time'] = opt_elapsed + sim_elapsed

    ga_stats = optimizer.get_statistics()
    print(f"\n  Estadísticas Algoritmo Genético:")
    print(f"    Evaluaciones: {ga_stats['evaluations_performed']}")
    print(f"    Mejor fitness: {ga_stats['best_fitness']:.2f}")

    return metrics


def print_comparison(results: dict):
    """Imprime tabla comparativa de resultados."""
    print(f"\n{'='*80}")
    print("TABLA COMPARATIVA DE RESULTADOS")
    print(f"{'='*80}")

    calc = MetricsCalculator()
    df = calc.create_summary_dataframe(results)

    print("\n" + df.to_string(index=False))

    # Calcular mejoras respecto a baseline
    print(f"\n{'='*80}")
    print("MEJORAS RESPECTO A BASELINE")
    print(f"{'='*80}")

    baseline_metrics = results['Baseline']

    for algo_name, metrics in results.items():
        if algo_name == 'Baseline':
            continue

        print(f"\n{algo_name}:")
        improvements = calc.calculate_improvement(baseline_metrics, metrics)

        for metric, improvement in improvements.items():
            symbol = "✓" if improvement > 0 else "✗"
            print(f"  {symbol} {metric:25s}: {improvement:+.1f}%")


def main():
    """Función principal."""
    print("="*80)
    print("COMPARACIÓN DE ALGORITMOS DE OPTIMIZACIÓN")
    print("Av. Brasil, Pocitos - Montevideo")
    print("Escenario: Flujo Medio (400 veh/h)")
    print("Duración: 10 minutos de simulación")
    print("="*80)

    results = {}

    # 1. Baseline
    results['Baseline'] = run_baseline()

    # 2. Max-Pressure
    results['Max-Pressure'] = run_max_pressure()

    # 3. Programación Dinámica
    results['DP'] = run_dynamic_programming()

    # 4. Branch & Bound
    results['B&B'] = run_branch_and_bound()

    # 5. Algoritmos Genéticos
    results['GA'] = run_genetic_algorithm()

    # Comparar resultados
    print_comparison(results)

    print(f"\n{'='*80}")
    print("COMPARACIÓN COMPLETADA")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
