# Ejemplos de Uso

Este directorio contiene scripts de ejemplo que demuestran cómo usar el sistema de simulación de tráfico.

## Scripts Disponibles

### `run_simulation.py`

Script completo que demuestra:
- Simulación con configuración baseline (tiempos fijos)
- Simulación con configuración optimizada (ondas verdes)
- Comparación de resultados
- Evaluación en diferentes escenarios de tráfico

**Uso:**
```bash
cd examples
python run_simulation.py
```

**Salida esperada:**
- Métricas de simulación baseline
- Métricas de simulación optimizada
- Comparación de mejoras porcentuales
- Tabla comparativa de escenarios (bajo, medio, alto flujo)

## Estructura de una Simulación

Todas las simulaciones siguen este patrón básico:

```python
from src.simulator import TrafficNetwork, TrafficSimulator, TrafficScenario
from src.utils.config import INTERSECTIONS_FILE, MEDIUM_FLOW_FILE

# 1. Cargar red vial
network = TrafficNetwork(str(INTERSECTIONS_FILE))

# 2. Cargar escenario de tráfico
scenario = TrafficScenario(str(MEDIUM_FLOW_FILE))

# 3. Crear simulador
simulator = TrafficSimulator(network, scenario)

# 4. Configurar semáforos (opcional)
simulator.configure_traffic_lights({
    1: {'north_south': 35, 'east_west': 40, 'offset': 0},
    2: {'north_south': 30, 'east_west': 35, 'offset': 20}
    # ... más intersecciones
})

# 5. Ejecutar simulación
metrics = simulator.run(duration=600, verbose=True)

# 6. Analizar resultados
print(f"Retraso promedio: {metrics['avg_delay']:.2f} s")
print(f"Throughput: {metrics['throughput_per_hour']:.0f} veh/h")
```

## Configuración de Semáforos

### Tiempos Fijos Estándar

```python
config = {
    intersection_id: {
        'north_south': 30,  # segundos de verde
        'east_west': 30,
        'offset': 0  # sin coordinación
    }
}
```

### Ondas Verdes

Para crear una onda verde, calcula offsets basándose en:
- Distancia entre intersecciones
- Velocidad promedio de viaje

```python
# Ejemplo: Av. Brasil con 250m entre intersecciones, 45 km/h
# Tiempo de viaje = 250m / (45 km/h = 12.5 m/s) = 20 segundos

config = {
    1: {'north_south': 30, 'east_west': 35, 'offset': 0},
    2: {'north_south': 30, 'east_west': 35, 'offset': 20},
    3: {'north_south': 30, 'east_west': 35, 'offset': 40},
    # ...
}
```

## Escenarios Disponibles

### Flujo Bajo
- Archivo: `data/scenarios/low_flow.json`
- Tasa: ~150 veh/hora
- Uso: Hora valle (10:00-16:00)

### Flujo Medio
- Archivo: `data/scenarios/medium_flow.json`
- Tasa: ~400 veh/hora
- Uso: Horario laboral normal

### Flujo Alto
- Archivo: `data/scenarios/high_flow.json`
- Tasa: ~750 veh/hora
- Uso: Hora pico (18:00-20:00)
- Flujo asimétrico: 75% hacia rambla

## Métricas Evaluadas

Cada simulación retorna las siguientes métricas:

- **avg_delay**: Retraso promedio por vehículo (segundos)
- **avg_waiting_time**: Tiempo de espera promedio (segundos)
- **avg_queue_length**: Longitud promedio de cola (vehículos)
- **max_queue_length**: Longitud máxima de cola observada
- **throughput**: Total de vehículos completados
- **throughput_per_hour**: Vehículos procesados por hora
- **avg_stops**: Número promedio de paradas por vehículo
- **avg_speed_kmh**: Velocidad promedio (km/h)
- **vehicles_completed**: Vehículos que llegaron a destino
- **vehicles_active**: Vehículos aún en la red
- **computation_time**: Tiempo real de cómputo (segundos)

## Próximos Ejemplos

En el futuro se agregarán ejemplos de:
- Uso de algoritmos de optimización (Programación Dinámica, Branch & Bound, Algoritmos Genéticos)
- Visualización animada de la simulación
- Análisis estadístico avanzado
- Experimentos con múltiples replicas
