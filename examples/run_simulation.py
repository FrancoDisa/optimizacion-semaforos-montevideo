"""
Script de ejemplo: Simulación completa de tráfico en Av. Brasil, Pocitos

Este script demuestra cómo usar el sistema completo de simulación
para evaluar diferentes configuraciones de semáforos.
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulator import TrafficNetwork, TrafficSimulator, TrafficScenario
from src.utils.config import INTERSECTIONS_FILE, LOW_FLOW_FILE, MEDIUM_FLOW_FILE, HIGH_FLOW_FILE
from src.utils.metrics import MetricsCalculator


def run_baseline_simulation():
    """
    Ejecuta simulación con configuración baseline (tiempos fijos estándar).

    Returns:
        dict: Métricas de la simulación
    """
    print("\n" + "="*70)
    print("SIMULACIÓN BASELINE - Tiempos Fijos Estándar")
    print("="*70)

    # Cargar red y escenario
    network = TrafficNetwork(str(INTERSECTIONS_FILE))
    scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))

    # Crear simulador
    simulator = TrafficSimulator(network, scenario)

    # Configuración baseline: todos los semáforos iguales
    baseline_config = {}
    for int_id in network.get_all_intersection_ids():
        baseline_config[int_id] = {
            'north_south': 30,
            'east_west': 30,
            'offset': 0  # Sin coordinación
        }

    simulator.configure_traffic_lights(baseline_config)

    # Ejecutar simulación de 10 minutos
    metrics = simulator.run(duration=600, verbose=True)

    return metrics


def run_optimized_simulation():
    """
    Ejecuta simulación con configuración optimizada manualmente.

    Returns:
        dict: Métricas de la simulación
    """
    print("\n" + "="*70)
    print("SIMULACIÓN OPTIMIZADA - Tiempos Ajustados + Ondas Verdes")
    print("="*70)

    # Cargar red y escenario
    network = TrafficNetwork(str(INTERSECTIONS_FILE))
    scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))

    # Crear simulador
    simulator = TrafficSimulator(network, scenario)

    # Configuración optimizada: tiempos diferentes + offsets para onda verde
    # Asumiendo flujo principal este-oeste (hacia la rambla en hora pico)
    optimized_config = {
        1: {  # Rambla (más tráfico entrando/saliendo)
            'north_south': 20,
            'east_west': 45,
            'offset': 0
        },
        2: {  # Benito Blanco
            'north_south': 25,
            'east_west': 40,
            'offset': 22  # Coordinado para onda verde
        },
        3: {  # Lázaro Gadea
            'north_south': 25,
            'east_west': 40,
            'offset': 44
        },
        4: {  # Alejandro Chucarro
            'north_south': 25,
            'east_west': 40,
            'offset': 65
        },
        5: {  # Paulino Pimienta (entrada desde interior)
            'north_south': 20,
            'east_west': 45,
            'offset': 86
        }
    }

    simulator.configure_traffic_lights(optimized_config)

    # Ejecutar simulación de 10 minutos
    metrics = simulator.run(duration=600, verbose=True)

    return metrics


def compare_scenarios():
    """Compara diferentes escenarios de tráfico."""
    print("\n" + "="*80)
    print("COMPARACIÓN DE ESCENARIOS DE TRÁFICO")
    print("="*80)

    scenarios_files = [
        (LOW_FLOW_FILE, "Flujo Bajo"),
        (MEDIUM_FLOW_FILE, "Flujo Medio"),
        (HIGH_FLOW_FILE, "Flujo Alto")
    ]

    results = {}

    for scenario_file, scenario_name in scenarios_files:
        print(f"\n{'='*80}")
        print(f"Escenario: {scenario_name}")
        print(f"{'='*80}")

        network = TrafficNetwork(str(INTERSECTIONS_FILE))
        scenario = TrafficScenario(str(scenario_file))
        simulator = TrafficSimulator(network, scenario)

        # Configuración estándar para comparar
        config = {}
        for int_id in network.get_all_intersection_ids():
            config[int_id] = {
                'north_south': 30,
                'east_west': 30,
                'offset': 0
            }

        simulator.configure_traffic_lights(config)

        # Simulación de 5 minutos
        metrics = simulator.run(duration=300, verbose=False)

        results[scenario_name] = metrics

    # Crear tabla comparativa
    print("\n" + "="*80)
    print("RESUMEN COMPARATIVO")
    print("="*80)

    calc = MetricsCalculator()
    df = calc.create_summary_dataframe(results)

    print(df.to_string(index=False))


def main():
    """Función principal del ejemplo."""
    print("="*80)
    print("EJEMPLO COMPLETO DE SIMULACIÓN DE TRÁFICO")
    print("Av. Brasil, Pocitos - Montevideo")
    print("="*80)

    # 1. Ejecutar simulación baseline
    baseline_metrics = run_baseline_simulation()

    # 2. Ejecutar simulación optimizada
    optimized_metrics = run_optimized_simulation()

    # 3. Comparar resultados
    print("\n" + "="*80)
    print("COMPARACIÓN: Baseline vs. Optimizado")
    print("="*80)

    calc = MetricsCalculator()
    improvements = calc.calculate_improvement(baseline_metrics, optimized_metrics)

    print("\nMétricas Baseline:")
    print(f"  Retraso promedio:     {baseline_metrics['avg_delay']:.2f} s")
    print(f"  Cola promedio:        {baseline_metrics['avg_queue_length']:.2f} veh")
    print(f"  Throughput:           {baseline_metrics['throughput_per_hour']:.0f} veh/h")
    print(f"  Paradas promedio:     {baseline_metrics['avg_stops']:.2f}")

    print("\nMétricas Optimizadas:")
    print(f"  Retraso promedio:     {optimized_metrics['avg_delay']:.2f} s")
    print(f"  Cola promedio:        {optimized_metrics['avg_queue_length']:.2f} veh")
    print(f"  Throughput:           {optimized_metrics['throughput_per_hour']:.0f} veh/h")
    print(f"  Paradas promedio:     {optimized_metrics['avg_stops']:.2f}")

    print("\nMejoras (% respecto a baseline):")
    for metric, improvement in improvements.items():
        symbol = "✓" if improvement > 0 else "✗"
        print(f"  {symbol} {metric:25s}: {improvement:+.1f}%")

    # 4. Comparar diferentes escenarios
    print("\n")
    compare_scenarios()

    print("\n" + "="*80)
    print("EJEMPLO COMPLETADO")
    print("="*80)


if __name__ == "__main__":
    main()
