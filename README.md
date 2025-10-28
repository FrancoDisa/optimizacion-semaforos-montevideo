# Optimización de Redes de Semáforos en Montevideo

Proyecto Final - Algoritmos Avanzados de Búsqueda y Optimización  
**Autor:** Franco Di Salvatore  
**Fecha:** Octubre 2025  
**Universidad:** [Tu Universidad]

## 📋 Descripción

Este proyecto desarrolla un sistema de control inteligente de semáforos que se adapta dinámicamente a las condiciones del tráfico en Montevideo, aplicando algoritmos avanzados de optimización para reducir la congestión vehicular y mejorar el flujo de tránsito.

### Problema

La congestión de tráfico en horas pico en Montevideo, especialmente en avenidas principales como Av. Italia, Av. Rivera y Bulevar Artigas, genera:
- Demoras significativas en los traslados
- Aumento en el consumo de combustible
- Mayor emisión de gases contaminantes
- Incremento en la probabilidad de accidentes

Los sistemas actuales de semáforos funcionan con **tiempos fijos** que no se adaptan a las condiciones reales del tráfico, causando ineficiencias en la distribución del flujo vehicular.

## 🎯 Objetivos

- Desarrollar un sistema de optimización de semáforos adaptativo
- Implementar y comparar 4 algoritmos diferentes de optimización
- Reducir significativamente los tiempos de congestión
- Mejorar el flujo vehicular y la movilidad urbana
- Demostrar la aplicabilidad de técnicas de optimización en problemas reales

## 🧮 Modelo Formal

### Representación de la Red Vial

La red urbana se modela como un **grafo dirigido** `G = (V, E)`:

- **V**: Conjunto de nodos (intersecciones con semáforos)
- **E**: Conjunto de aristas (tramos de calle)
  - Cada arista tiene: longitud, velocidad máxima, número de carriles, flujo vehicular

### Variables de Decisión

- Duración de cada fase de verde en cada semáforo
- Orden de activación de las fases (norte-sur, este-oeste, giros)
- Offset o desfase entre semáforos (para ondas verdes coordinadas)

### Restricciones

- La suma de duraciones de todas las fases ≤ ciclo total L
- Cada fase tiene límites mínimos y máximos de duración
- Ninguna dirección puede permanecer en rojo durante todo el ciclo

### Función Objetivo

Minimizar el retraso promedio de los vehículos considerando:
- Tiempo de viaje promedio
- Tiempo de espera acumulado en intersecciones
- Longitud máxima de colas (prevenir bloqueos)

## 🔬 Algoritmos Implementados

### 1. Programación Dinámica
Optimiza los tiempos de verde dentro de un único semáforo, dividiendo cada ciclo en fases (norte-sur, este-oeste, giros) y encontrando la distribución óptima que minimiza la demora total.

### 2. Branch & Bound (Ramificación y Poda)
Coordina múltiples semáforos explorando diferentes configuraciones de desfases para lograr "ondas verdes". Utiliza cotas inferiores para podar ramas no prometedoras del árbol de búsqueda.

### 3. Algoritmos Genéticos
Optimización global de toda la red mediante evolución artificial. Cada configuración completa (tiempos y desfases) es un "individuo" que evoluciona a través de cruzamiento y mutación.

### 4. Heurística Max-Pressure (Baseline)
Método de referencia que toma decisiones en tiempo real basándose en la "presión" de cada dirección (relación entre vehículos esperando y capacidad de avance).

## 📊 Simulación y Evaluación

### Simulador de Tráfico
Desarrollado en Python con:
- Pasos discretos de simulación (cada segundo)
- Actualización de posiciones, colas y tiempos de espera
- Modelado de comportamiento vehicular realista

### Escenarios de Prueba
Red representando una zona de Montevideo (8-12 intersecciones, ej: Pocitos):

1. **Flujo bajo**: Tránsito fluido fuera de hora pico
2. **Flujo medio**: Condiciones normales de operación
3. **Flujo alto**: Hora pico con demanda saturada

### Métricas de Evaluación

- ⏱️ **Retraso promedio** por vehículo (segundos)
- 🚗 **Longitud media** de cola en cada intersección
- 📈 **Throughput**: Cantidad de vehículos procesados exitosamente
- 🛑 **Número promedio** de paradas por vehículo
- ⚡ **Tiempo de cómputo** y convergencia de cada algoritmo

## 🏆 Criterios de Éxito

El proyecto será exitoso si se logra:
- ✅ Reducción significativa de atascos vs. sistema de tiempos fijos
- ✅ Disminución en la longitud media de colas
- ✅ Aumento en el throughput de vehículos procesados
- ✅ Tiempos de cómputo razonables para aplicación en tiempo real

## 🛠️ Estructura del Proyecto

```
optimizacion-semaforos-montevideo/
│
├── src/
│   ├── algorithms/
│   │   ├── dynamic_programming.py    # Programación Dinámica
│   │   ├── branch_and_bound.py       # Branch & Bound
│   │   ├── genetic_algorithm.py      # Algoritmos Genéticos
│   │   └── max_pressure.py           # Heurística Max-Pressure
│   │
│   ├── simulator/
│   │   ├── traffic_simulator.py      # Simulador principal
│   │   ├── graph_network.py          # Modelo de grafo
│   │   └── vehicle.py                # Clase vehículo
│   │
│   └── utils/
│       ├── metrics.py                # Cálculo de métricas
│       └── visualization.py          # Visualización de resultados
│
├── data/
│   ├── intersections.json            # Datos de intersecciones
│   └── traffic_patterns.json         # Patrones de tráfico
│
├── tests/
│   └── test_algorithms.py            # Tests unitarios
│
├── docs/
│   ├── propuesta.pdf                 # Propuesta original
│   └── resultados.md                 # Documentación de resultados
│
├── notebooks/
│   └── analisis_resultados.ipynb     # Análisis y visualizaciones
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Instalación

```bash
# Clonar el repositorio
git clone https://github.com/FrancoDisa/optimizacion-semaforos-montevideo.git
cd optimizacion-semaforos-montevideo

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## 💻 Uso

```python
# Ejemplo básico de uso
from src.simulator.traffic_simulator import TrafficSimulator
from src.algorithms.genetic_algorithm import GeneticOptimizer

# Cargar red de semáforos
simulator = TrafficSimulator('data/intersections.json')

# Ejecutar optimización
optimizer = GeneticOptimizer(simulator)
best_config = optimizer.optimize(generations=100)

# Evaluar resultados
metrics = simulator.evaluate(best_config)
print(f"Retraso promedio: {metrics['avg_delay']} segundos")
```

## 📦 Dependencias

```
numpy>=1.24.0
matplotlib>=3.7.0
networkx>=3.1
scipy>=1.10.0
pandas>=2.0.0
pytest>=7.3.0
jupyter>=1.0.0
```

## 📈 Resultados Esperados

Los algoritmos se compararán en términos de:
- Calidad de la solución obtenida
- Eficiencia computacional
- Adaptabilidad a diferentes escenarios de tráfico
- Aplicabilidad en escenarios reales de Montevideo

## 🤝 Contribuciones

Este es un proyecto académico. Si tienes sugerencias o mejoras, no dudes en:
1. Hacer fork del proyecto
2. Crear una rama para tu feature (`git checkout -b feature/mejora`)
3. Commit tus cambios (`git commit -m 'Agrega mejora'`)
4. Push a la rama (`git push origin feature/mejora`)
5. Abrir un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 👨‍💻 Autor

**Franco Di Salvatore**  
Algoritmos Avanzados de Búsqueda y Optimización  
Universidad Catolica del Uruguay
GitHub: [@FrancoDisa](https://github.com/FrancoDisa)

## 📚 Referencias

- Webster, F. V. (1958). Traffic signal settings. Road Research Technical Paper.
- Roess, R. P., et al. (2004). Traffic Engineering. Pearson Education.
- Varaiya, P. (2013). Max pressure control of a network of signalized intersections.
- Intendencia de Montevideo - Datos de tráfico urbano
