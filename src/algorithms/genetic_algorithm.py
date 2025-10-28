"""
Algoritmo Genético para optimización global de semáforos.

Este algoritmo optimiza toda la red de semáforos simultáneamente,
evolucionando una población de configuraciones completas mediante
selección, cruzamiento y mutación.
"""

from typing import Dict, List, Tuple
import numpy as np
import random
from copy import deepcopy
import time as timer


class Individual:
    """
    Representa un individuo (configuración completa de la red).

    Un individuo es un cromosoma que codifica:
    [green_ns_1, green_ew_1, offset_1, green_ns_2, green_ew_2, offset_2, ...]
    """

    def __init__(self, genes: List[float], intersection_ids: List[int]):
        """
        Inicializa un individuo.

        Args:
            genes: Lista de valores (cromosoma)
            intersection_ids: Lista ordenada de IDs de intersecciones
        """
        self.genes = genes
        self.intersection_ids = intersection_ids
        self.fitness_value = None

    def decode(self) -> Dict[int, Dict]:
        """
        Decodifica el cromosoma a configuración de semáforos.

        Returns:
            dict: {intersection_id: {'north_south': X, 'east_west': Y, 'offset': Z}}
        """
        configuration = {}
        genes_per_intersection = 3  # north_south, east_west, offset

        for i, int_id in enumerate(self.intersection_ids):
            start_idx = i * genes_per_intersection
            configuration[int_id] = {
                'north_south': int(self.genes[start_idx]),
                'east_west': int(self.genes[start_idx + 1]),
                'offset': int(self.genes[start_idx + 2])
            }

        return configuration

    def clone(self):
        """Crea una copia del individuo."""
        return Individual(self.genes.copy(), self.intersection_ids.copy())


class GeneticOptimizer:
    """
    Optimizador basado en Algoritmos Genéticos.

    Evoluciona una población de configuraciones completas de la red,
    usando selección, cruzamiento y mutación para encontrar la
    configuración óptima global.
    """

    def __init__(self, network, simulator, population_size: int = 100,
                 generations: int = 200, mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8, tournament_size: int = 5,
                 elitism_count: int = 5, min_green: int = 10,
                 max_green: int = 60, max_offset: int = 120):
        """
        Inicializa el optimizador genético.

        Args:
            network: Instancia de TrafficNetwork
            simulator: Simulador para evaluar fitness
            population_size: Tamaño de la población
            generations: Número de generaciones a evolucionar
            mutation_rate: Probabilidad de mutación (0-1)
            crossover_rate: Probabilidad de cruzamiento (0-1)
            tournament_size: Tamaño del torneo para selección
            elitism_count: Número de mejores individuos a preservar
            min_green: Tiempo mínimo de verde (segundos)
            max_green: Tiempo máximo de verde (segundos)
            max_offset: Offset máximo (segundos)
        """
        self.network = network
        self.simulator = simulator
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.min_green = min_green
        self.max_green = max_green
        self.max_offset = max_offset

        # IDs de intersecciones ordenadas
        self.intersection_ids = sorted(network.get_all_intersection_ids())
        self.genes_per_individual = len(self.intersection_ids) * 3

        # Población actual
        self.population: List[Individual] = []

        # Mejor individuo encontrado
        self.best_individual = None
        self.best_fitness = float('-inf')

        # Historia de evolución
        self.fitness_history = []
        self.avg_fitness_history = []

        # Estadísticas
        self.evaluations_performed = 0
        self.start_time = None

    def initialize_population(self):
        """Crea población inicial aleatoria."""
        print("Inicializando población...")
        self.population = []

        for i in range(self.population_size):
            genes = []

            for int_id in self.intersection_ids:
                # Tiempo verde norte-sur (aleatorio en rango válido)
                green_ns = random.randint(self.min_green, self.max_green)
                genes.append(float(green_ns))

                # Tiempo verde este-oeste
                green_ew = random.randint(self.min_green, self.max_green)
                genes.append(float(green_ew))

                # Offset
                offset = random.randint(0, self.max_offset)
                genes.append(float(offset))

            individual = Individual(genes, self.intersection_ids)
            self.population.append(individual)

        print(f"  Población de {self.population_size} individuos creada")

    def fitness(self, individual: Individual) -> float:
        """
        Evalúa el fitness de un individuo.

        Ejecuta una simulación corta y retorna -avg_delay
        (negativo porque queremos minimizar delay pero maximizar fitness)

        Args:
            individual: Individuo a evaluar

        Returns:
            float: Fitness (mayor es mejor)
        """
        self.evaluations_performed += 1

        # Decodificar configuración
        configuration = individual.decode()

        # Configurar simulador
        self.simulator.configure_traffic_lights(configuration)
        self.simulator.reset()

        # Ejecutar simulación corta (120 segundos para evaluar)
        try:
            metrics = self.simulator.run(duration=120, verbose=False)

            # Fitness = -delay (queremos minimizar delay)
            # Agregar penalización por colas muy largas
            fitness = -metrics['avg_delay'] - (metrics['max_queue_length'] * 0.5)

            return fitness

        except Exception as e:
            # Si hay error en simulación, fitness muy bajo
            print(f"    Error en evaluación: {e}")
            return -10000.0

    def tournament_selection(self) -> Individual:
        """
        Selección por torneo.

        Elige tournament_size individuos al azar y retorna el mejor.

        Returns:
            Individual: Individuo seleccionado
        """
        tournament = random.sample(self.population, self.tournament_size)

        # Retornar el mejor del torneo
        best = max(tournament, key=lambda ind: ind.fitness_value)

        return best

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Cruzamiento de un punto.

        Args:
            parent1: Primer padre
            parent2: Segundo padre

        Returns:
            tuple: (hijo1, hijo2)
        """
        if random.random() > self.crossover_rate:
            # No hay cruzamiento, retornar copias de padres
            return parent1.clone(), parent2.clone()

        # Punto de cruzamiento aleatorio
        point = random.randint(1, len(parent1.genes) - 1)

        # Crear hijos
        child1_genes = parent1.genes[:point] + parent2.genes[point:]
        child2_genes = parent2.genes[:point] + parent1.genes[point:]

        child1 = Individual(child1_genes, self.intersection_ids)
        child2 = Individual(child2_genes, self.intersection_ids)

        return child1, child2

    def mutate(self, individual: Individual):
        """
        Mutación gaussiana.

        Modifica aleatoriamente genes del individuo.

        Args:
            individual: Individuo a mutar (modificado in-place)
        """
        for i in range(len(individual.genes)):
            if random.random() < self.mutation_rate:
                # Mutación gaussiana: agregar ruido
                sigma = 5.0  # Desviación estándar de la mutación
                individual.genes[i] += random.gauss(0, sigma)

                # Asegurar límites según tipo de gen
                gene_type = i % 3  # 0: green_ns, 1: green_ew, 2: offset

                if gene_type < 2:  # Tiempos de verde
                    individual.genes[i] = np.clip(
                        individual.genes[i], self.min_green, self.max_green
                    )
                else:  # Offset
                    individual.genes[i] = np.clip(
                        individual.genes[i], 0, self.max_offset
                    )

    def evolve(self) -> Dict[int, Dict]:
        """
        Ejecuta el proceso evolutivo completo.

        Returns:
            dict: Mejor configuración encontrada
        """
        print(f"\n{'='*70}")
        print("INICIANDO ALGORITMO GENÉTICO")
        print(f"{'='*70}")
        print(f"Población: {self.population_size}")
        print(f"Generaciones: {self.generations}")
        print(f"Tasa de mutación: {self.mutation_rate}")
        print(f"Tasa de cruzamiento: {self.crossover_rate}")

        self.start_time = timer.time()

        # Inicializar población
        self.initialize_population()

        # Evaluar población inicial
        print("\nEvaluando población inicial...")
        for individual in self.population:
            individual.fitness_value = self.fitness(individual)

        # Encontrar mejor inicial
        self.best_individual = max(self.population, key=lambda ind: ind.fitness_value)
        self.best_fitness = self.best_individual.fitness_value

        print(f"  Mejor fitness inicial: {self.best_fitness:.2f}")

        # Evolución
        print(f"\n{'='*70}")
        print("EVOLUCIÓN")
        print(f"{'='*70}")

        for generation in range(self.generations):
            # Nueva generación
            new_population = []

            # Elitismo: preservar mejores individuos
            sorted_population = sorted(
                self.population,
                key=lambda ind: ind.fitness_value,
                reverse=True
            )

            for i in range(self.elitism_count):
                new_population.append(sorted_population[i].clone())

            # Generar resto de la población
            while len(new_population) < self.population_size:
                # Selección
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()

                # Cruzamiento
                child1, child2 = self.crossover(parent1, parent2)

                # Mutación
                self.mutate(child1)
                self.mutate(child2)

                # Agregar hijos
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            # Reemplazar población
            self.population = new_population[:self.population_size]

            # Evaluar nueva población
            for individual in self.population:
                if individual.fitness_value is None:
                    individual.fitness_value = self.fitness(individual)

            # Actualizar mejor
            generation_best = max(self.population, key=lambda ind: ind.fitness_value)

            if generation_best.fitness_value > self.best_fitness:
                self.best_individual = generation_best.clone()
                self.best_fitness = generation_best.fitness_value

            # Registrar historia
            self.fitness_history.append(self.best_fitness)
            avg_fitness = np.mean([ind.fitness_value for ind in self.population])
            self.avg_fitness_history.append(avg_fitness)

            # Reportar progreso cada 10 generaciones
            if generation % 10 == 0 or generation == self.generations - 1:
                elapsed = timer.time() - self.start_time
                print(f"  Gen {generation:3d}: Mejor={self.best_fitness:.2f}, "
                      f"Promedio={avg_fitness:.2f}, "
                      f"Evaluaciones={self.evaluations_performed}, "
                      f"Tiempo={elapsed:.1f}s")

        elapsed = timer.time() - self.start_time

        print(f"\n{'='*70}")
        print("EVOLUCIÓN COMPLETADA")
        print(f"{'='*70}")
        print(f"Mejor fitness final: {self.best_fitness:.2f}")
        print(f"Evaluaciones realizadas: {self.evaluations_performed}")
        print(f"Tiempo total: {elapsed:.2f}s")

        # Retornar mejor configuración
        return self.best_individual.decode()

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
            'algorithm': 'Genetic Algorithm',
            'population_size': self.population_size,
            'generations': self.generations,
            'evaluations_performed': self.evaluations_performed,
            'best_fitness': self.best_fitness,
            'computation_time': elapsed,
            'fitness_history': self.fitness_history,
            'avg_fitness_history': self.avg_fitness_history
        }


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.simulator import TrafficNetwork, TrafficSimulator, TrafficScenario
    from src.utils.config import INTERSECTIONS_FILE, MEDIUM_FLOW_FILE

    print("="*70)
    print("EJEMPLO: Algoritmo Genético")
    print("="*70)

    # Cargar red y escenario
    network = TrafficNetwork(str(INTERSECTIONS_FILE))
    scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))

    # Crear simulador
    simulator = TrafficSimulator(network, scenario)

    # Crear optimizador genético (con parámetros reducidos para demo)
    optimizer = GeneticOptimizer(
        network=network,
        simulator=simulator,
        population_size=20,     # Pequeño para demo
        generations=30,          # Pocas generaciones para demo
        mutation_rate=0.15,
        crossover_rate=0.8,
        elitism_count=2
    )

    # Evolucionar
    best_configuration = optimizer.evolve()

    print("\n" + "="*70)
    print("MEJOR CONFIGURACIÓN ENCONTRADA")
    print("="*70)
    for int_id in sorted(best_configuration.keys()):
        config = best_configuration[int_id]
        print(f"\nIntersección {int_id}:")
        print(f"  Norte-Sur: {config['north_south']}s")
        print(f"  Este-Oeste: {config['east_west']}s")
        print(f"  Offset: {config['offset']}s")

    # Estadísticas
    print("\n" + "="*70)
    print("ESTADÍSTICAS")
    print("="*70)
    stats = optimizer.get_statistics()
    for key, value in stats.items():
        if key in ['fitness_history', 'avg_fitness_history']:
            continue  # Skip historial para no saturar salida
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
