# Arquitectura Técnica del Sistema
## Optimización de Redes de Semáforos - Montevideo

**Autor:** Franco Di Salvatore
**Fecha:** Octubre 2025
**Versión:** 1.0

---

## 1. Visión General

Este documento describe la arquitectura técnica del sistema de optimización de semáforos para Avenida Brasil en Pocitos, Montevideo.

### 1.1. Componentes Principales

```
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE OPTIMIZACIÓN                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Prog.      │  │   Branch &   │  │  Algoritmos  │      │
│  │  Dinámica    │  │    Bound     │  │   Genéticos  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                    ┌──────────────┐                          │
│                    │ Max-Pressure │                          │
│                    └──────────────┘                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  CAPA DE SIMULACIÓN                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           TrafficSimulator (Motor Principal)         │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ TrafficLight │  │    Vehicle   │  │   Network    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  CAPA DE EVALUACIÓN                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Metrics    │  │Visualization │  │    Results   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Módulo de Simulación (`src/simulator/`)

### 2.1. TrafficNetwork

**Responsabilidad:** Representar la red vial como un grafo dirigido G = (V, E)

**Clase Principal:**
```python
class TrafficNetwork:
    """
    Grafo dirigido que representa la red de calles.

    Attributes:
        graph (nx.DiGraph): Grafo de NetworkX
        intersections (dict): Diccionario de intersecciones {id: Intersection}
        edges (list): Lista de aristas con atributos
    """

    def __init__(self, intersections_file: str):
        """Carga la red desde archivo JSON"""

    def add_intersection(self, intersection_data: dict):
        """Agrega una intersección al grafo"""

    def add_edge(self, from_id: int, to_id: int, **attributes):
        """Agrega una arista (tramo de calle)"""

    def get_shortest_path(self, origin: int, destination: int):
        """Calcula ruta más corta entre dos intersecciones"""

    def get_distance(self, from_id: int, to_id: int) -> float:
        """Retorna distancia entre dos nodos conectados"""
```

**Atributos de Nodos (Intersecciones):**
- `id`: Identificador único
- `name`: Nombre completo
- `coordinates`: (lat, lon)
- `traffic_light`: Instancia de TrafficLight
- `queue`: Cola de vehículos esperando

**Atributos de Aristas (Tramos de Calle):**
- `length_m`: Longitud en metros
- `travel_time_s`: Tiempo de viaje estimado
- `lanes`: Número de carriles
- `direction`: "este", "oeste", "norte", "sur"

---

### 2.2. TrafficLight

**Responsabilidad:** Modelar el comportamiento de un semáforo

```python
class TrafficLight:
    """
    Representa un semáforo con múltiples fases.

    Attributes:
        intersection_id (int): ID de la intersección
        cycle_length (int): Duración total del ciclo (segundos)
        phases (dict): Configuración de fases {nombre: duración}
        current_phase (str): Fase actual activa
        phase_start_time (float): Tiempo de inicio de fase actual
        offset (int): Desfase respecto a t=0 (para ondas verdes)
    """

    def __init__(self, intersection_id: int, cycle_length: int = 90):
        self.phases = {
            "north_south": 30,
            "east_west": 30,
            "left_turns": 15
        }
        self.offset = 0

    def update(self, current_time: float):
        """Actualiza estado del semáforo según el tiempo"""

    def get_current_state(self, direction: str) -> str:
        """
        Retorna estado actual para una dirección.
        Returns: "green", "yellow", "red"
        """

    def set_configuration(self, phase_durations: dict, offset: int = 0):
        """Configura duraciones de fases y offset"""

    def can_vehicle_pass(self, direction: str) -> bool:
        """Determina si un vehículo puede pasar"""
```

**Estados de Semáforo:**
1. **Verde** (green): Vehículos pueden avanzar
2. **Amarillo** (yellow): Advertencia, preparar para detenerse
3. **Rojo** (red): Vehículos deben detenerse

**Fases:**
- `north_south`: Flujo norte-sur y viceversa
- `east_west`: Flujo este-oeste y viceversa
- `left_turns`: Giros a la izquierda (opcional)

---

### 2.3. Vehicle

**Responsabilidad:** Modelar el comportamiento de un vehículo individual

```python
class Vehicle:
    """
    Representa un vehículo en la simulación.

    Attributes:
        id (int): Identificador único
        origin (int): ID de intersección de origen
        destination (int): ID de intersección de destino
        route (list): Lista de IDs de intersecciones (ruta)
        current_position (tuple): Posición actual en el grafo
        speed (float): Velocidad actual (m/s)
        waiting_time (float): Tiempo total de espera acumulado
        num_stops (int): Número de paradas en semáforos
        spawn_time (float): Tiempo de generación
        arrival_time (float): Tiempo de llegada a destino
    """

    def __init__(self, vehicle_id: int, origin: int, destination: int,
                 spawn_time: float, network: TrafficNetwork):
        self.route = network.get_shortest_path(origin, destination)
        self.current_edge_index = 0
        self.position_on_edge = 0.0  # metros desde inicio de arista

    def update(self, dt: float, traffic_light_state: str):
        """
        Actualiza posición y estado del vehículo.

        Args:
            dt: Paso de tiempo (segundos)
            traffic_light_state: Estado del próximo semáforo
        """

    def accelerate(self, target_speed: float, dt: float):
        """Acelera hacia velocidad objetivo"""

    def brake(self, dt: float):
        """Frena el vehículo"""

    def has_arrived(self) -> bool:
        """Verifica si llegó a destino"""
```

---

### 2.4. TrafficSimulator

**Responsabilidad:** Motor principal de simulación

```python
class TrafficSimulator:
    """
    Motor de simulación de tráfico vehicular.

    Simula el movimiento de vehículos en la red considerando
    semáforos, colas, y comportamiento realista.
    """

    def __init__(self, network: TrafficNetwork, scenario_file: str):
        self.network = network
        self.scenario = self._load_scenario(scenario_file)
        self.vehicles = []
        self.current_time = 0.0
        self.dt = SimulatorConfig.TIME_STEP

    def run(self, duration: int, traffic_light_config: dict) -> dict:
        """
        Ejecuta simulación por duration segundos.

        Args:
            duration: Duración en segundos
            traffic_light_config: Configuración de semáforos

        Returns:
            dict: Métricas de evaluación
        """
        self._configure_traffic_lights(traffic_light_config)

        for t in range(0, duration, self.dt):
            self.step()

        return self.get_metrics()

    def step(self):
        """Ejecuta un paso de simulación (1 segundo)"""
        # 1. Generar nuevos vehículos según patrón de demanda
        self._spawn_vehicles()

        # 2. Actualizar estados de semáforos
        self._update_traffic_lights()

        # 3. Mover vehículos
        self._update_vehicles()

        # 4. Actualizar colas en intersecciones
        self._update_queues()

        # 5. Registrar métricas instantáneas
        self._record_metrics()

        self.current_time += self.dt

    def _spawn_vehicles(self):
        """Genera nuevos vehículos según distribución de Poisson"""

    def get_metrics(self) -> dict:
        """Calcula y retorna métricas de evaluación"""
        return {
            'avg_delay': self._calculate_avg_delay(),
            'avg_queue_length': self._calculate_avg_queue_length(),
            'throughput': len(self.completed_vehicles),
            'avg_stops': self._calculate_avg_stops(),
            'max_queue_length': self.max_queue_observed
        }
```

---

## 3. Módulo de Algoritmos (`src/algorithms/`)

### 3.1. Programación Dinámica

**Objetivo:** Optimizar tiempos de verde en un **solo semáforo**

```python
class DynamicProgrammingOptimizer:
    """
    Optimiza la distribución de tiempos de verde entre fases
    de un semáforo individual usando Programación Dinámica.
    """

    def optimize_single_light(self, intersection_id: int,
                              traffic_data: dict,
                              cycle_length: int = 90) -> dict:
        """
        Encuentra distribución óptima de fases para un semáforo.

        Args:
            intersection_id: ID de la intersección
            traffic_data: Datos de flujo vehicular
            cycle_length: Duración total del ciclo

        Returns:
            dict: {'north_south': 35, 'east_west': 40, 'left_turns': 12}
        """
        # Estado: dp[tiempo_restante][ultima_fase] = costo_mínimo
        dp = {}

        # Caso base: tiempo_restante = 0
        dp[(0, None)] = 0

        # Llenar tabla DP
        for time_remaining in range(1, cycle_length + 1):
            for phase in ['north_south', 'east_west', 'left_turns']:
                # Calcular costo de asignar este tiempo a esta fase
                cost = self._estimate_delay(phase, time_duration, traffic_data)
                # Actualizar tabla DP

        return self._reconstruct_solution(dp)

    def _estimate_delay(self, phase: str, duration: int,
                       traffic_data: dict) -> float:
        """
        Estima demora causada por una configuración de fase.
        Usa modelo de Webster para cálculo de demoras.
        """
```

**Complejidad:** O(L × P × L) donde L es la duración del ciclo y P el número de fases

---

### 3.2. Branch & Bound

**Objetivo:** Coordinar **múltiples semáforos** para crear ondas verdes

```python
class BranchAndBoundCoordinator:
    """
    Coordina múltiples semáforos ajustando offsets para
    lograr ondas verdes usando Branch & Bound.
    """

    def coordinate_lights(self, light_configs: dict,
                         network: TrafficNetwork,
                         simulator: TrafficSimulator) -> dict:
        """
        Encuentra offsets óptimos entre semáforos.

        Args:
            light_configs: Configuraciones individuales de semáforos
            network: Red vial
            simulator: Simulador para evaluar configuraciones

        Returns:
            dict: {intersection_id: offset}
        """
        best_solution = None
        best_cost = float('inf')

        # Inicializar cola de prioridad con configuración inicial
        queue = PriorityQueue()
        initial_node = Node(offsets={}, level=0)
        queue.put((self._lower_bound(initial_node), initial_node))

        while not queue.empty():
            _, node = queue.get()

            if node.level == len(light_configs):
                # Solución completa, evaluar
                cost = self._evaluate(node.offsets, simulator)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = node.offsets
            else:
                # Ramificar: probar diferentes offsets
                for offset in range(0, 120, 5):  # Cada 5 segundos
                    child = node.branch(offset)
                    lb = self._lower_bound(child)

                    # Poda: si cota inferior > mejor solución, descartar
                    if lb < best_cost:
                        queue.put((lb, child))

        return best_solution

    def _lower_bound(self, node: Node) -> float:
        """
        Calcula cota inferior optimista del costo.
        Asume que los semáforos restantes se configuran óptimamente.
        """

    def _evaluate(self, offsets: dict, simulator: TrafficSimulator) -> float:
        """Evalúa una configuración completa ejecutando simulación"""
```

**Estrategia de Poda:** Si cota_inferior(nodo) ≥ mejor_solución_encontrada, podar rama

---

### 3.3. Algoritmos Genéticos

**Objetivo:** Optimización **global** de toda la red simultáneamente

```python
class GeneticOptimizer:
    """
    Optimiza toda la red de semáforos usando Algoritmos Genéticos.
    Cada individuo representa una configuración completa.
    """

    def __init__(self, network: TrafficNetwork,
                 simulator: TrafficSimulator,
                 population_size: int = 100,
                 generations: int = 200):
        self.population_size = population_size
        self.generations = generations

    def optimize(self) -> dict:
        """
        Ejecuta evolución genética.

        Returns:
            dict: Mejor configuración encontrada
        """
        # Inicializar población aleatoria
        population = self._initialize_population()

        for generation in range(self.generations):
            # Evaluar fitness de cada individuo
            fitness_scores = [self.fitness(ind) for ind in population]

            # Selección
            parents = self._tournament_selection(population, fitness_scores)

            # Cruzamiento
            offspring = []
            for i in range(0, len(parents), 2):
                child1, child2 = self._crossover(parents[i], parents[i+1])
                offspring.extend([child1, child2])

            # Mutación
            for individual in offspring:
                if random.random() < self.mutation_rate:
                    self._mutate(individual)

            # Elitismo: mantener mejores individuos
            population = self._elitism(population, offspring, fitness_scores)

        # Retornar mejor individuo final
        return max(population, key=self.fitness)

    def _encode_individual(self) -> list:
        """
        Codifica un individuo como cromosoma.

        Estructura: [g1_ns, g1_ew, g1_turns, offset1, g2_ns, g2_ew, ...]
        Donde gi_* son duraciones de verde para semáforo i
        """

    def fitness(self, individual: list) -> float:
        """
        Función de fitness: ejecuta simulación y retorna -delay.
        (Negativo porque queremos minimizar delay pero maximizar fitness)
        """
        config = self._decode(individual)
        metrics = self.simulator.run(3600, config)
        return -metrics['avg_delay']

    def _crossover(self, parent1: list, parent2: list) -> tuple:
        """Cruzamiento de un punto"""
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def _mutate(self, individual: list):
        """Mutación: modifica aleatoriamente un gen"""
        index = random.randint(0, len(individual) - 1)
        individual[index] += random.gauss(0, 5)  # ±5 segundos
        individual[index] = max(10, min(60, individual[index]))  # Límites
```

**Codificación del Cromosoma:**
```
Individuo = [
    verde_ns_sem1, verde_ew_sem1, verde_giros_sem1, offset_sem1,
    verde_ns_sem2, verde_ew_sem2, verde_giros_sem2, offset_sem2,
    ...
]
```

---

### 3.4. Max-Pressure

**Objetivo:** Heurística **en tiempo real** como baseline

```python
class MaxPressureController:
    """
    Controlador heurístico que toma decisiones en tiempo real
    basándose en la presión de cada dirección.
    """

    def __init__(self, network: TrafficNetwork):
        self.min_phase_duration = 10  # segundos mínimos por fase

    def decide_phase(self, intersection_id: int,
                    current_time: float,
                    traffic_state: dict) -> str:
        """
        Decide qué fase activar según presión máxima.

        Args:
            intersection_id: ID de la intersección
            current_time: Tiempo actual de simulación
            traffic_state: Estado actual del tráfico

        Returns:
            str: Fase a activar ('north_south', 'east_west', 'left_turns')
        """
        pressures = {}

        for direction in ['north_south', 'east_west', 'left_turns']:
            pressures[direction] = self._calculate_pressure(
                intersection_id, direction, traffic_state
            )

        # Seleccionar fase con mayor presión
        return max(pressures, key=pressures.get)

    def _calculate_pressure(self, intersection_id: int,
                           direction: str,
                           traffic_state: dict) -> float:
        """
        Presión = vehículos_esperando - capacidad_downstream

        La presión representa cuánto "necesita" una dirección
        tener luz verde en este momento.
        """
        queue_length = traffic_state['queues'][intersection_id][direction]
        downstream_capacity = self._get_downstream_capacity(
            intersection_id, direction
        )

        return queue_length - downstream_capacity
```

---

## 4. Módulo de Utilidades (`src/utils/`)

### 4.1. Métricas (`metrics.py`)

```python
class MetricsCalculator:
    """Calcula métricas de evaluación del sistema."""

    @staticmethod
    def average_delay(vehicles: list) -> float:
        """
        Retraso promedio por vehículo (segundos).

        Delay = tiempo_real - tiempo_ideal_sin_semáforos
        """

    @staticmethod
    def average_queue_length(queue_history: list) -> float:
        """Longitud media de cola en intersecciones"""

    @staticmethod
    def throughput(completed_vehicles: list, duration: int) -> float:
        """Vehículos completados por hora"""

    @staticmethod
    def average_stops(vehicles: list) -> float:
        """Número promedio de paradas por vehículo"""
```

### 4.2. Visualización (`visualization.py`)

```python
class TrafficVisualizer:
    """Genera visualizaciones de resultados."""

    def plot_network(self, network: TrafficNetwork):
        """Visualiza la red vial"""

    def plot_algorithm_comparison(self, results: dict):
        """Gráfico de barras comparando algoritmos"""

    def plot_queue_evolution(self, queue_history: list):
        """Evolución temporal de colas"""

    def animate_traffic(self, simulation_history: list):
        """Animación del movimiento vehicular"""
```

---

## 5. Flujo de Ejecución

### 5.1. Flujo Típico de Experimento

```python
# 1. Cargar red vial
network = TrafficNetwork('data/montevideo/pocitos_intersections.json')

# 2. Crear simulador con escenario
simulator = TrafficSimulator(network, 'data/scenarios/high_flow.json')

# 3. Ejecutar cada algoritmo
results = {}

# Programación Dinámica
dp_optimizer = DynamicProgrammingOptimizer()
dp_config = {}
for intersection_id in network.get_all_intersections():
    dp_config[intersection_id] = dp_optimizer.optimize_single_light(
        intersection_id, network.get_traffic_data(intersection_id)
    )
results['DP'] = simulator.run(3600, dp_config)

# Branch & Bound
bb_coordinator = BranchAndBoundCoordinator()
bb_config = bb_coordinator.coordinate_lights(dp_config, network, simulator)
results['BB'] = simulator.run(3600, bb_config)

# Algoritmos Genéticos
ga_optimizer = GeneticOptimizer(network, simulator)
ga_config = ga_optimizer.optimize()
results['GA'] = simulator.run(3600, ga_config)

# Max-Pressure
mp_controller = MaxPressureController(network)
# Max-Pressure se ejecuta en tiempo real dentro del simulador
results['MP'] = simulator.run_with_realtime_control(3600, mp_controller)

# 4. Comparar resultados
visualizer = TrafficVisualizer()
visualizer.plot_algorithm_comparison(results)
```

---

## 6. Consideraciones de Diseño

### 6.1. Patrones de Diseño Utilizados

- **Strategy Pattern**: Para intercambiar diferentes algoritmos de optimización
- **Observer Pattern**: Para registrar métricas durante la simulación
- **Factory Pattern**: Para crear vehículos y configuraciones

### 6.2. Principios SOLID

- **Single Responsibility**: Cada clase tiene una responsabilidad clara
- **Open/Closed**: Extensible para nuevos algoritmos sin modificar código existente
- **Dependency Inversion**: Las clases dependen de abstracciones, no de implementaciones concretas

### 6.3. Escalabilidad

El sistema está diseñado para escalar de 5 a 50+ intersecciones:
- Uso de grafos eficientes (NetworkX)
- Algoritmos con complejidad controlada
- Posibilidad de paralelizar evaluaciones en Algoritmos Genéticos

---

## 7. Testing

### 7.1. Estrategia de Testing

```python
# tests/test_simulator.py
def test_vehicle_movement():
    """Verifica que vehículos se muevan correctamente"""

def test_traffic_light_transitions():
    """Verifica transiciones de estados de semáforos"""

def test_queue_formation():
    """Verifica formación de colas en semáforos rojos"""

# tests/test_algorithms.py
def test_dp_optimization():
    """Verifica que DP retorne configuración válida"""

def test_ga_convergence():
    """Verifica que AG converja a mejor solución"""
```

### 7.2. Cobertura Objetivo

- **Mínimo**: 80% de cobertura de código
- **Crítico**: 100% de cobertura en lógica de algoritmos

---

## 8. Próximos Pasos

1. Implementar clases base del simulador
2. Implementar Programación Dinámica (más simple)
3. Validar con casos de prueba pequeños
4. Implementar algoritmos más complejos
5. Calibrar con datos reales
6. Ejecutar experimentos completos
7. Analizar y documentar resultados

---

**Fin del Documento de Arquitectura**
