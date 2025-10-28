"""
Tests para el módulo de semáforos (TrafficLight).
"""

import pytest
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulator.traffic_light import TrafficLight, TrafficLightPhase, LightState


class TestTrafficLightPhase:
    """Tests para la clase TrafficLightPhase."""

    def test_phase_creation(self):
        """Test de creación de fase."""
        phase = TrafficLightPhase(
            name="north_south",
            green_duration=30,
            green_directions=["north", "south"]
        )

        assert phase.name == "north_south"
        assert phase.green_duration == 30
        assert "north" in phase.green_directions

    def test_phase_durations(self):
        """Test de duraciones de fase."""
        phase = TrafficLightPhase("test", 30, ["north"])

        # 30s verde + 3s amarillo + 1s todo-rojo = 34s total
        assert phase.green_duration == 30
        assert phase.yellow_duration == 3
        assert phase.all_red_duration == 1
        assert phase.total_duration == 34

    def test_phase_validation(self):
        """Test de validación de duraciones."""
        # Muy corto
        with pytest.raises(ValueError):
            TrafficLightPhase("test", 5, ["north"])

        # Muy largo
        with pytest.raises(ValueError):
            TrafficLightPhase("test", 100, ["north"])

    def test_state_at_time(self):
        """Test de estado en diferentes momentos."""
        phase = TrafficLightPhase("test", 30, ["north"])

        # En verde (0-30s)
        assert phase.get_state_at_time(15) == LightState.GREEN

        # En amarillo (30-33s)
        assert phase.get_state_at_time(31) == LightState.YELLOW

        # En rojo (33-34s y después)
        assert phase.get_state_at_time(33.5) == LightState.RED
        assert phase.get_state_at_time(40) == LightState.RED


class TestTrafficLight:
    """Tests para la clase TrafficLight."""

    def test_light_creation(self):
        """Test de creación de semáforo."""
        light = TrafficLight(
            intersection_id=1,
            intersection_name="Test Intersection"
        )

        assert light.intersection_id == 1
        assert light.intersection_name == "Test Intersection"
        assert light.is_active
        assert len(light.phases) > 0  # Configuración por defecto

    def test_default_configuration(self):
        """Test de configuración por defecto."""
        light = TrafficLight(1)

        # Debe tener al menos 2 fases
        assert len(light.phases) >= 2

        # Ciclo completo debe ser razonable
        cycle = light.get_cycle_length()
        assert 60 <= cycle <= 120

    def test_custom_configuration(self):
        """Test de configuración personalizada."""
        light = TrafficLight(1)

        light.set_configuration({
            "north_south": 35,
            "east_west": 40,
            "left_turns": 15
        })

        assert len(light.phases) == 3

        # Verificar nombres
        phase_names = [p.name for p in light.phases]
        assert "north_south" in phase_names
        assert "east_west" in phase_names
        assert "left_turns" in phase_names

    def test_update_and_phase_changes(self):
        """Test de actualización y cambios de fase."""
        light = TrafficLight(1)
        light.set_configuration({
            "north_south": 30,
            "east_west": 30
        })

        # Inicialmente en fase 0
        assert light.current_phase_index == 0

        # Actualizar por 40 segundos (más que una fase)
        for t in range(40):
            light.update(t, dt=1.0)

        # Debería haber cambiado de fase
        # Fase 1 dura 34s total, así que en t=40 estamos en fase 2
        assert light.current_phase_index > 0

    def test_can_vehicle_pass(self):
        """Test de permiso para pasar."""
        light = TrafficLight(1)
        light.set_configuration({"north_south": 30, "east_west": 30})

        # Resetear
        light.reset()

        # Al inicio, norte-sur en verde
        assert light.can_vehicle_pass("north")
        assert light.can_vehicle_pass("south")
        assert not light.can_vehicle_pass("east")
        assert not light.can_vehicle_pass("west")

    def test_time_until_green(self):
        """Test de cálculo de tiempo hasta verde."""
        light = TrafficLight(1)
        light.set_configuration({"north_south": 30, "east_west": 30})

        light.reset()

        # Norte está en verde, tiempo = 0
        assert light.get_time_until_green("north") == 0.0

        # Este está en rojo, debe esperar
        time_east = light.get_time_until_green("east")
        assert time_east > 0

    def test_efficiency_ratio(self):
        """Test de ratio de eficiencia."""
        light = TrafficLight(1)
        light.set_configuration({"north_south": 30, "east_west": 30})

        # Cada dirección tiene 30s de verde
        # Ciclo = 68s (34s + 34s)
        ratio_north = light.get_efficiency_ratio("north")
        ratio_east = light.get_efficiency_ratio("east")

        # Ambos deberían tener ~44% (30/68)
        assert 0.4 < ratio_north < 0.5
        assert 0.4 < ratio_east < 0.5

    def test_offset_configuration(self):
        """Test de configuración con offset."""
        light = TrafficLight(1)
        light.set_configuration({"north_south": 30}, offset=10)

        assert light.offset == 10

        # Con offset de 10s, el semáforo no debería estar activo antes de t=10
        light.reset(0)
        light.update(5, dt=1.0)

        # El tiempo en fase debería ser 0 aún
        assert light.time_in_current_phase == 0

    def test_cycle_completion(self):
        """Test de conteo de ciclos completos."""
        light = TrafficLight(1)
        light.set_configuration({"north_south": 20, "east_west": 20})

        light.reset()

        cycle_length = light.get_cycle_length()
        assert light.total_cycles_completed == 0

        # Simular un ciclo completo + un poco más
        for t in range(cycle_length + 5):
            light.update(t, dt=1.0)

        # Debería haber completado al menos 1 ciclo
        assert light.total_cycles_completed >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
