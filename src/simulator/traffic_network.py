"""
Modelo de red vial como grafo dirigido.

Este módulo implementa la representación de la red de calles de Montevideo
como un grafo dirigido donde los nodos son intersecciones y las aristas son
tramos de calle.
"""

import json
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class Intersection:
    """
    Representa una intersección en la red vial.

    Una intersección es un punto donde se cruzan dos o más calles,
    típicamente controlado por un semáforo.
    """

    def __init__(self, intersection_id: int, name: str, coordinates: Tuple[float, float],
                 intersection_type: str = "residential_street", importance: str = "medium"):
        """
        Inicializa una intersección.

        Args:
            intersection_id: Identificador único de la intersección
            name: Nombre completo de la intersección (ej: "Av. Brasil y Rambla")
            coordinates: Tupla (latitud, longitud)
            intersection_type: Tipo de intersección (rambla_intersection, residential_street, etc.)
            importance: Nivel de importancia (high, medium, low)
        """
        self.id = intersection_id
        self.name = name
        self.lat, self.lon = coordinates
        self.type = intersection_type
        self.importance = importance

        # Estado dinámico de la intersección
        self.queue_north_south = []  # Cola de vehículos norte-sur
        self.queue_south_north = []  # Cola de vehículos sur-norte
        self.queue_east_west = []    # Cola de vehículos este-oeste
        self.queue_west_east = []    # Cola de vehículos oeste-este

        # Semáforo asociado (se asignará después)
        self.traffic_light = None

    def get_queue_length(self, direction: str) -> int:
        """
        Retorna la longitud de la cola en una dirección específica.

        Args:
            direction: Dirección ('north_south', 'south_north', 'east_west', 'west_east')

        Returns:
            int: Número de vehículos esperando
        """
        queue_map = {
            'north_south': self.queue_north_south,
            'south_north': self.queue_south_north,
            'east_west': self.queue_east_west,
            'west_east': self.queue_west_east
        }
        return len(queue_map.get(direction, []))

    def add_vehicle_to_queue(self, vehicle, direction: str):
        """Agrega un vehículo a la cola correspondiente."""
        queue_map = {
            'north_south': self.queue_north_south,
            'south_north': self.queue_south_north,
            'east_west': self.queue_east_west,
            'west_east': self.queue_west_east
        }
        if direction in queue_map:
            queue_map[direction].append(vehicle)

    def remove_vehicle_from_queue(self, vehicle, direction: str):
        """Remueve un vehículo de la cola correspondiente."""
        queue_map = {
            'north_south': self.queue_north_south,
            'south_north': self.queue_south_north,
            'east_west': self.queue_east_west,
            'west_east': self.queue_west_east
        }
        if direction in queue_map and vehicle in queue_map[direction]:
            queue_map[direction].remove(vehicle)

    def get_total_queue_length(self) -> int:
        """Retorna el número total de vehículos esperando en todas las direcciones."""
        return (len(self.queue_north_south) + len(self.queue_south_north) +
                len(self.queue_east_west) + len(self.queue_west_east))

    def clear_queues(self):
        """Limpia todas las colas de la intersección."""
        self.queue_north_south.clear()
        self.queue_south_north.clear()
        self.queue_east_west.clear()
        self.queue_west_east.clear()

    def __str__(self) -> str:
        return f"Intersection({self.id}: {self.name})"

    def __repr__(self) -> str:
        return f"Intersection(id={self.id}, name='{self.name}', coords=({self.lat:.4f}, {self.lon:.4f}))"


class RoadSegment:
    """
    Representa un tramo de calle entre dos intersecciones.

    Un segmento es una arista dirigida en el grafo, con atributos
    como longitud, número de carriles, y velocidad máxima.
    """

    def __init__(self, from_id: int, to_id: int, length_m: float,
                 travel_time_s: float, lanes: int = 2, direction: str = ""):
        """
        Inicializa un segmento de calle.

        Args:
            from_id: ID de la intersección de origen
            to_id: ID de la intersección de destino
            length_m: Longitud del segmento en metros
            travel_time_s: Tiempo de viaje ideal en segundos
            lanes: Número de carriles
            direction: Dirección cardinal (norte, sur, este, oeste)
        """
        self.from_id = from_id
        self.to_id = to_id
        self.length_m = length_m
        self.travel_time_s = travel_time_s
        self.lanes = lanes
        self.direction = direction

        # Calcular velocidad promedio
        self.avg_speed_ms = length_m / travel_time_s if travel_time_s > 0 else 0
        self.avg_speed_kmh = self.avg_speed_ms * 3.6

        # Vehículos actualmente en este segmento
        self.vehicles = []

    def get_capacity(self) -> int:
        """
        Calcula la capacidad aproximada del segmento.

        Usa la regla de 7 metros por vehículo (4.5m vehículo + 2.5m distancia segura).

        Returns:
            int: Número máximo de vehículos que pueden estar en el segmento
        """
        meters_per_vehicle = 7.0
        return int((self.length_m / meters_per_vehicle) * self.lanes)

    def is_full(self) -> bool:
        """Verifica si el segmento está lleno."""
        return len(self.vehicles) >= self.get_capacity()

    def get_occupancy_rate(self) -> float:
        """Retorna el porcentaje de ocupación del segmento (0.0 a 1.0)."""
        return len(self.vehicles) / max(1, self.get_capacity())

    def add_vehicle(self, vehicle):
        """Agrega un vehículo al segmento."""
        if vehicle not in self.vehicles:
            self.vehicles.append(vehicle)

    def remove_vehicle(self, vehicle):
        """Remueve un vehículo del segmento."""
        if vehicle in self.vehicles:
            self.vehicles.remove(vehicle)

    def __str__(self) -> str:
        return f"RoadSegment({self.from_id} → {self.to_id}, {self.length_m}m)"

    def __repr__(self) -> str:
        return (f"RoadSegment(from={self.from_id}, to={self.to_id}, "
                f"length={self.length_m}m, lanes={self.lanes})")


class TrafficNetwork:
    """
    Representa la red vial completa como un grafo dirigido G = (V, E).

    Esta clase encapsula la topología de la red de calles, incluyendo
    intersecciones (nodos) y tramos de calle (aristas), y proporciona
    métodos para consultar y manipular la red.
    """

    def __init__(self, intersections_file: Optional[str] = None):
        """
        Inicializa la red vial.

        Args:
            intersections_file: Ruta al archivo JSON con datos de la red.
                               Si es None, crea una red vacía.
        """
        self.graph = nx.DiGraph()
        self.intersections: Dict[int, Intersection] = {}
        self.segments: Dict[Tuple[int, int], RoadSegment] = {}

        # Metadata de la red
        self.network_name = ""
        self.location = ""
        self.description = ""

        if intersections_file:
            self.load_from_file(intersections_file)

    def load_from_file(self, filepath: str):
        """
        Carga la red desde un archivo JSON.

        Args:
            filepath: Ruta al archivo JSON con la definición de la red

        Raises:
            FileNotFoundError: Si el archivo no existe
            json.JSONDecodeError: Si el archivo no es JSON válido
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {filepath}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Cargar metadata
        self.network_name = data.get('network_name', '')
        self.location = data.get('location', '')
        self.description = data.get('description', '')

        # Cargar intersecciones
        for intersection_data in data.get('intersections', []):
            self.add_intersection(
                intersection_id=intersection_data['id'],
                name=intersection_data['name'],
                coordinates=(intersection_data['coordinates']['lat'],
                           intersection_data['coordinates']['lon']),
                intersection_type=intersection_data.get('type', 'residential_street'),
                importance=intersection_data.get('importance', 'medium')
            )

        # Cargar aristas (segmentos de calle)
        for edge_data in data.get('edges', []):
            self.add_segment(
                from_id=edge_data['from_id'],
                to_id=edge_data['to_id'],
                length_m=edge_data['length_m'],
                travel_time_s=edge_data['travel_time_s'],
                lanes=edge_data.get('lanes', 2),
                direction=edge_data.get('direction', '')
            )

        print(f"✓ Red cargada: {self.network_name}")
        print(f"  Ubicación: {self.location}")
        print(f"  Intersecciones: {len(self.intersections)}")
        print(f"  Segmentos: {len(self.segments)}")

    def add_intersection(self, intersection_id: int, name: str,
                        coordinates: Tuple[float, float],
                        intersection_type: str = "residential_street",
                        importance: str = "medium"):
        """
        Agrega una intersección a la red.

        Args:
            intersection_id: ID único de la intersección
            name: Nombre de la intersección
            coordinates: Tupla (latitud, longitud)
            intersection_type: Tipo de intersección
            importance: Nivel de importancia
        """
        intersection = Intersection(
            intersection_id, name, coordinates, intersection_type, importance
        )
        self.intersections[intersection_id] = intersection

        # Agregar nodo al grafo con sus atributos
        self.graph.add_node(
            intersection_id,
            name=name,
            lat=coordinates[0],
            lon=coordinates[1],
            type=intersection_type,
            importance=importance,
            intersection=intersection
        )

    def add_segment(self, from_id: int, to_id: int, length_m: float,
                   travel_time_s: float, lanes: int = 2, direction: str = ""):
        """
        Agrega un segmento de calle (arista dirigida) a la red.

        Args:
            from_id: ID de intersección de origen
            to_id: ID de intersección de destino
            length_m: Longitud en metros
            travel_time_s: Tiempo de viaje en segundos
            lanes: Número de carriles
            direction: Dirección cardinal
        """
        segment = RoadSegment(from_id, to_id, length_m, travel_time_s, lanes, direction)
        self.segments[(from_id, to_id)] = segment

        # Agregar arista al grafo con sus atributos
        self.graph.add_edge(
            from_id, to_id,
            length=length_m,
            travel_time=travel_time_s,
            lanes=lanes,
            direction=direction,
            weight=travel_time_s,  # Para algoritmos de ruta más corta
            segment=segment
        )

    def get_intersection(self, intersection_id: int) -> Optional[Intersection]:
        """Retorna la intersección con el ID dado."""
        return self.intersections.get(intersection_id)

    def get_segment(self, from_id: int, to_id: int) -> Optional[RoadSegment]:
        """Retorna el segmento entre dos intersecciones."""
        return self.segments.get((from_id, to_id))

    def get_all_intersection_ids(self) -> List[int]:
        """Retorna lista de todos los IDs de intersecciones."""
        return list(self.intersections.keys())

    def get_shortest_path(self, origin: int, destination: int) -> Optional[List[int]]:
        """
        Calcula la ruta más corta entre dos intersecciones.

        Usa el algoritmo de Dijkstra basado en tiempo de viaje.

        Args:
            origin: ID de intersección de origen
            destination: ID de intersección de destino

        Returns:
            Lista de IDs de intersecciones formando la ruta, o None si no hay ruta
        """
        try:
            return nx.shortest_path(self.graph, origin, destination, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_path_length(self, path: List[int]) -> float:
        """
        Calcula la longitud total de una ruta en metros.

        Args:
            path: Lista de IDs de intersecciones

        Returns:
            float: Longitud total en metros
        """
        total_length = 0.0
        for i in range(len(path) - 1):
            segment = self.get_segment(path[i], path[i + 1])
            if segment:
                total_length += segment.length_m
        return total_length

    def get_path_travel_time(self, path: List[int]) -> float:
        """
        Calcula el tiempo de viaje ideal de una ruta en segundos.

        Args:
            path: Lista de IDs de intersecciones

        Returns:
            float: Tiempo en segundos (sin considerar semáforos ni colas)
        """
        total_time = 0.0
        for i in range(len(path) - 1):
            segment = self.get_segment(path[i], path[i + 1])
            if segment:
                total_time += segment.travel_time_s
        return total_time

    def get_neighbors(self, intersection_id: int) -> List[int]:
        """
        Retorna lista de intersecciones vecinas (conectadas por salida).

        Args:
            intersection_id: ID de la intersección

        Returns:
            Lista de IDs de intersecciones vecinas
        """
        return list(self.graph.successors(intersection_id))

    def get_incoming_neighbors(self, intersection_id: int) -> List[int]:
        """
        Retorna lista de intersecciones con aristas entrantes.

        Args:
            intersection_id: ID de la intersección

        Returns:
            Lista de IDs de intersecciones que conectan a esta
        """
        return list(self.graph.predecessors(intersection_id))

    def get_network_stats(self) -> Dict:
        """
        Calcula estadísticas de la red.

        Returns:
            dict: Diccionario con estadísticas de la red
        """
        total_length = sum(seg.length_m for seg in self.segments.values())
        avg_segment_length = total_length / len(self.segments) if self.segments else 0

        return {
            'num_intersections': len(self.intersections),
            'num_segments': len(self.segments),
            'total_length_km': total_length / 1000,
            'avg_segment_length_m': avg_segment_length,
            'is_connected': nx.is_weakly_connected(self.graph),
            'network_name': self.network_name,
            'location': self.location
        }

    def visualize(self, show_labels: bool = True, figsize: Tuple[int, int] = (12, 8)):
        """
        Visualiza la red vial.

        Args:
            show_labels: Si True, muestra nombres de intersecciones
            figsize: Tamaño de la figura
        """
        plt.figure(figsize=figsize)

        # Posiciones basadas en coordenadas GPS
        pos = {node_id: (data['lon'], data['lat'])
               for node_id, data in self.graph.nodes(data=True)}

        # Dibujar nodos (intersecciones)
        node_colors = []
        for node_id in self.graph.nodes():
            intersection = self.intersections[node_id]
            if intersection.importance == 'high':
                node_colors.append('#FF6B6B')
            else:
                node_colors.append('#4ECDC4')

        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors,
                              node_size=500, alpha=0.9)

        # Dibujar aristas (calles)
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray',
                              width=2, alpha=0.6, arrows=True,
                              arrowsize=20, arrowstyle='->')

        # Etiquetas
        if show_labels:
            labels = {node_id: self.intersections[node_id].name.split(' y ')[-1]
                     for node_id in self.graph.nodes()}
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)

        plt.title(f"{self.network_name}\n{self.location}", fontsize=14, fontweight='bold')
        plt.xlabel("Longitud")
        plt.ylabel("Latitud")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return plt.gcf()

    def __str__(self) -> str:
        return f"TrafficNetwork('{self.network_name}', {len(self.intersections)} intersections)"

    def __repr__(self) -> str:
        stats = self.get_network_stats()
        return (f"TrafficNetwork(name='{self.network_name}', "
                f"intersections={stats['num_intersections']}, "
                f"segments={stats['num_segments']}, "
                f"length={stats['total_length_km']:.2f}km)")


if __name__ == "__main__":
    # Ejemplo de uso
    from src.utils.config import INTERSECTIONS_FILE

    network = TrafficNetwork(str(INTERSECTIONS_FILE))

    print("\n" + "="*60)
    print("ESTADÍSTICAS DE LA RED")
    print("="*60)
    stats = network.get_network_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "="*60)
    print("INTERSECCIONES")
    print("="*60)
    for int_id, intersection in network.intersections.items():
        print(f"  {intersection}")

    print("\n" + "="*60)
    print("EJEMPLO: Ruta más corta")
    print("="*60)
    path = network.get_shortest_path(1, 5)
    if path:
        print(f"  Ruta de intersección 1 a 5: {path}")
        print(f"  Longitud: {network.get_path_length(path):.0f} metros")
        print(f"  Tiempo: {network.get_path_travel_time(path):.0f} segundos")
