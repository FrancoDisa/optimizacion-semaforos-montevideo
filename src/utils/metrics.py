"""
Sistema de métricas y análisis de resultados.

Este módulo proporciona funciones para calcular y analizar métricas
de evaluación del sistema de semáforos.
"""

from typing import List, Dict
import numpy as np
import pandas as pd


class MetricsCalculator:
    """
    Calculadora de métricas de evaluación para simulaciones de tráfico.

    Proporciona métodos estáticos para calcular diversas métricas
    de rendimiento del sistema.
    """

    @staticmethod
    def average_delay(vehicles: List) -> float:
        """
        Calcula el retraso promedio por vehículo.

        El retraso es la diferencia entre el tiempo real de viaje
        y el tiempo ideal sin semáforos ni tráfico.

        Args:
            vehicles: Lista de vehículos completados

        Returns:
            float: Retraso promedio en segundos
        """
        if not vehicles:
            return 0.0

        delays = [v.total_waiting_time for v in vehicles]
        return np.mean(delays)

    @staticmethod
    def median_delay(vehicles: List) -> float:
        """
        Calcula el retraso mediano por vehículo.

        Args:
            vehicles: Lista de vehículos completados

        Returns:
            float: Retraso mediano en segundos
        """
        if not vehicles:
            return 0.0

        delays = [v.total_waiting_time for v in vehicles]
        return np.median(delays)

    @staticmethod
    def percentile_delay(vehicles: List, percentile: float = 95) -> float:
        """
        Calcula el percentil del retraso.

        Args:
            vehicles: Lista de vehículos completados
            percentile: Percentil a calcular (0-100)

        Returns:
            float: Retraso en el percentil dado
        """
        if not vehicles:
            return 0.0

        delays = [v.total_waiting_time for v in vehicles]
        return np.percentile(delays, percentile)

    @staticmethod
    def average_queue_length(queue_history: List[Dict]) -> float:
        """
        Calcula la longitud promedio de colas.

        Args:
            queue_history: Historial de longitudes de cola

        Returns:
            float: Longitud promedio de cola
        """
        if not queue_history:
            return 0.0

        all_lengths = []
        for snapshot in queue_history:
            all_lengths.extend(snapshot['queues'].values())

        return np.mean(all_lengths) if all_lengths else 0.0

    @staticmethod
    def max_queue_length(queue_history: List[Dict]) -> int:
        """
        Encuentra la longitud máxima de cola observada.

        Args:
            queue_history: Historial de longitudes de cola

        Returns:
            int: Longitud máxima de cola
        """
        if not queue_history:
            return 0

        all_lengths = []
        for snapshot in queue_history:
            all_lengths.extend(snapshot['queues'].values())

        return max(all_lengths) if all_lengths else 0

    @staticmethod
    def throughput(vehicles: List, simulation_time: float) -> float:
        """
        Calcula el throughput (vehículos procesados por hora).

        Args:
            vehicles: Lista de vehículos completados
            simulation_time: Tiempo total de simulación en segundos

        Returns:
            float: Vehículos procesados por hora
        """
        if simulation_time <= 0:
            return 0.0

        return (len(vehicles) / simulation_time) * 3600

    @staticmethod
    def average_stops(vehicles: List) -> float:
        """
        Calcula el número promedio de paradas por vehículo.

        Args:
            vehicles: Lista de vehículos completados

        Returns:
            float: Número promedio de paradas
        """
        if not vehicles:
            return 0.0

        stops = [v.num_stops for v in vehicles]
        return np.mean(stops)

    @staticmethod
    def average_speed(vehicles: List) -> float:
        """
        Calcula la velocidad promedio de los vehículos.

        Args:
            vehicles: Lista de vehículos completados

        Returns:
            float: Velocidad promedio en km/h
        """
        if not vehicles:
            return 0.0

        speeds = [v.get_average_speed_kmh() for v in vehicles]
        return np.mean(speeds)

    @staticmethod
    def total_travel_time(vehicles: List) -> float:
        """
        Calcula el tiempo total de viaje de todos los vehículos.

        Args:
            vehicles: Lista de vehículos completados

        Returns:
            float: Tiempo total en segundos
        """
        if not vehicles:
            return 0.0

        return sum(v.get_travel_time(v.arrival_time or 0) for v in vehicles)

    @staticmethod
    def fuel_consumption_estimate(vehicles: List) -> float:
        """
        Estima el consumo de combustible total.

        Usa modelo simplificado:
        - Consumo base: 0.08 L/km en movimiento
        - Consumo en ralentí: 0.0006 L/s parado

        Args:
            vehicles: Lista de vehículos completados

        Returns:
            float: Consumo estimado en litros
        """
        if not vehicles:
            return 0.0

        total_fuel = 0.0

        for vehicle in vehicles:
            # Combustible por distancia recorrida
            distance_km = vehicle.distance_traveled / 1000
            fuel_distance = distance_km * 0.08  # L/km

            # Combustible en ralentí (parado)
            fuel_idle = vehicle.total_waiting_time * 0.0006  # L/s

            total_fuel += fuel_distance + fuel_idle

        return total_fuel

    @staticmethod
    def co2_emissions_estimate(vehicles: List) -> float:
        """
        Estima emisiones de CO2.

        Asume 2.3 kg CO2 por litro de combustible.

        Args:
            vehicles: Lista de vehículos completados

        Returns:
            float: Emisiones estimadas en kg de CO2
        """
        fuel_liters = MetricsCalculator.fuel_consumption_estimate(vehicles)
        return fuel_liters * 2.3  # kg CO2 por litro

    @staticmethod
    def create_summary_dataframe(results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Crea un DataFrame con resumen comparativo de algoritmos.

        Args:
            results: Dict {algorithm_name: metrics_dict}

        Returns:
            pd.DataFrame: DataFrame con métricas comparadas
        """
        data = []

        for algo_name, metrics in results.items():
            data.append({
                'Algorithm': algo_name,
                'Avg Delay (s)': metrics.get('avg_delay', 0),
                'Avg Queue': metrics.get('avg_queue_length', 0),
                'Max Queue': metrics.get('max_queue_length', 0),
                'Throughput (veh/h)': metrics.get('throughput_per_hour', 0),
                'Avg Stops': metrics.get('avg_stops', 0),
                'Avg Speed (km/h)': metrics.get('avg_speed_kmh', 0),
                'Computation Time (s)': metrics.get('computation_time', 0),
                'Completed Vehicles': metrics.get('vehicles_completed', 0)
            })

        df = pd.DataFrame(data)

        # Ordenar por retraso promedio (menor es mejor)
        df = df.sort_values('Avg Delay (s)')

        return df

    @staticmethod
    def calculate_improvement(baseline_metrics: Dict, optimized_metrics: Dict) -> Dict:
        """
        Calcula mejoras porcentuales respecto a baseline.

        Args:
            baseline_metrics: Métricas del algoritmo baseline
            optimized_metrics: Métricas del algoritmo optimizado

        Returns:
            dict: Diccionario con mejoras porcentuales
        """
        improvements = {}

        # Métricas donde menor es mejor
        for metric in ['avg_delay', 'avg_queue_length', 'max_queue_length', 'avg_stops']:
            baseline_val = baseline_metrics.get(metric, 0)
            optimized_val = optimized_metrics.get(metric, 0)

            if baseline_val > 0:
                improvement = ((baseline_val - optimized_val) / baseline_val) * 100
                improvements[metric] = improvement
            else:
                improvements[metric] = 0.0

        # Métricas donde mayor es mejor
        for metric in ['throughput_per_hour', 'avg_speed_kmh']:
            baseline_val = baseline_metrics.get(metric, 0)
            optimized_val = optimized_metrics.get(metric, 0)

            if baseline_val > 0:
                improvement = ((optimized_val - baseline_val) / baseline_val) * 100
                improvements[metric] = improvement
            else:
                improvements[metric] = 0.0

        return improvements

    @staticmethod
    def statistical_significance_test(results1: List[float], results2: List[float]) -> Dict:
        """
        Realiza test de significancia estadística entre dos conjuntos de resultados.

        Usa test t de Student para muestras independientes.

        Args:
            results1: Lista de valores del primer grupo
            results2: Lista de valores del segundo grupo

        Returns:
            dict: Resultado del test con p-value y conclusión
        """
        from scipy import stats

        if len(results1) < 2 or len(results2) < 2:
            return {
                'test': 't-test',
                'statistic': None,
                'p_value': None,
                'significant': False,
                'message': 'Muestras insuficientes'
            }

        statistic, p_value = stats.ttest_ind(results1, results2)

        # Nivel de significancia típico: α = 0.05
        significant = p_value < 0.05

        return {
            'test': 't-test',
            'statistic': statistic,
            'p_value': p_value,
            'significant': significant,
            'message': f"{'Diferencia significativa' if significant else 'No hay diferencia significativa'} (p={p_value:.4f})"
        }


if __name__ == "__main__":
    # Ejemplo de uso
    print("="*70)
    print("EJEMPLO: Calculadora de Métricas")
    print("="*70)

    # Crear vehículos ficticios para demostración
    class MockVehicle:
        def __init__(self, delay, stops, distance, speed):
            self.total_waiting_time = delay
            self.num_stops = stops
            self.distance_traveled = distance
            self.arrival_time = 100

        def get_travel_time(self, t):
            return 100

        def get_average_speed_kmh(self):
            return 35.0

    vehicles = [
        MockVehicle(45, 3, 1000, 35),
        MockVehicle(60, 4, 1200, 32),
        MockVehicle(30, 2, 800, 40),
        MockVehicle(55, 3, 1100, 34),
    ]

    calc = MetricsCalculator()

    print("\nMétricas calculadas:")
    print(f"  Retraso promedio:     {calc.average_delay(vehicles):.2f} s")
    print(f"  Retraso mediano:      {calc.median_delay(vehicles):.2f} s")
    print(f"  Retraso P95:          {calc.percentile_delay(vehicles, 95):.2f} s")
    print(f"  Paradas promedio:     {calc.average_stops(vehicles):.2f}")
    print(f"  Velocidad promedio:   {calc.average_speed(vehicles):.2f} km/h")
    print(f"  Consumo combustible:  {calc.fuel_consumption_estimate(vehicles):.3f} L")
    print(f"  Emisiones CO2:        {calc.co2_emissions_estimate(vehicles):.3f} kg")

    # Ejemplo de comparación
    print("\n" + "="*70)
    print("EJEMPLO: Comparación de Algoritmos")
    print("="*70)

    baseline = {
        'avg_delay': 60.0,
        'avg_queue_length': 8.5,
        'throughput_per_hour': 400
    }

    optimized = {
        'avg_delay': 45.0,
        'avg_queue_length': 6.2,
        'throughput_per_hour': 450
    }

    improvements = calc.calculate_improvement(baseline, optimized)

    print("\nMejoras respecto a baseline:")
    for metric, improvement in improvements.items():
        print(f"  {metric:25s}: {improvement:+.1f}%")
