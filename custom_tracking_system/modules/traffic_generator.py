"""
Traffic Generator Module
Spawns and manages vehicles and pedestrians in CARLA simulation
"""

import carla
import random
import logging

logger = logging.getLogger(__name__)

class TrafficGenerator:
    """
    Generates traffic actors (vehicles and pedestrians) for testing
    """

    def __init__(self, world, num_vehicles=10, num_pedestrians=5):
        """
        Initialize traffic generator

        Args:
            world: CARLA world object
            num_vehicles: Number of vehicles to spawn
            num_pedestrians: Number of pedestrians to spawn
        """
        self.world = world
        self.num_vehicles = num_vehicles
        self.num_pedestrians = num_pedestrians
        self.vehicle_list = []
        self.pedestrian_list = []
        self.walker_controllers = []

        logger.info(f"TrafficGenerator initialized: {num_vehicles} vehicles, {num_pedestrians} pedestrians")

    def spawn_actors(self):
        """
        Spawn vehicles and pedestrians at random spawn points.
        In synchronous mode, controllers must be started AFTER a world.tick().
        """
        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()

        if len(spawn_points) == 0:
            logger.warning("No spawn points available on map")
            return

        # Spawn vehicles
        self._spawn_vehicles(blueprint_library, spawn_points)

        # Spawn pedestrians (actors only — controllers not started yet)
        self._spawn_pedestrians(blueprint_library, spawn_points)

        # Tick once so CARLA registers all spawned actors before starting AI
        self.world.tick()

        # Now it is safe to start walker AI controllers
        self._start_walker_controllers(spawn_points)

        logger.info(f"Spawned {len(self.vehicle_list)} vehicles and {len(self.pedestrian_list)} pedestrians")

    def _spawn_vehicles(self, blueprint_library, spawn_points):
        """Spawn vehicles with autopilot enabled"""
        vehicle_blueprints = blueprint_library.filter('vehicle.*')

        for i in range(self.num_vehicles):
            try:
                # Random vehicle blueprint
                bp = random.choice(vehicle_blueprints)

                # Set random color
                if bp.has_attribute('color'):
                    color = random.choice(bp.get_attribute('color').recommended_values)
                    bp.set_attribute('color', color)

                # Random spawn point
                spawn_point = random.choice(spawn_points)

                # Spawn vehicle
                vehicle = self.world.try_spawn_actor(bp, spawn_point)

                if vehicle is not None:
                    # Enable autopilot
                    vehicle.set_autopilot(True)
                    self.vehicle_list.append(vehicle)
                    logger.debug(f"Spawned vehicle at {spawn_point.location}")
                else:
                    logger.warning(f"Failed to spawn vehicle at {spawn_point.location}")

            except Exception as e:
                logger.error(f"Error spawning vehicle: {e}")

    def _spawn_pedestrians(self, blueprint_library, spawn_points):
        """Spawn pedestrian actors only — do NOT start AI controllers here.
        Controllers must be started after world.tick() in synchronous mode."""
        walker_blueprints = blueprint_library.filter('walker.pedestrian.*')
        walker_controller_bp = blueprint_library.find('controller.ai.walker')

        for i in range(self.num_pedestrians):
            try:
                bp = random.choice(walker_blueprints)
                spawn_point = random.choice(spawn_points)
                pedestrian = self.world.try_spawn_actor(bp, spawn_point)

                if pedestrian is not None:
                    self.pedestrian_list.append(pedestrian)

                    # Spawn controller attached to pedestrian (do NOT call start() yet)
                    controller = self.world.spawn_actor(walker_controller_bp,
                                                       carla.Transform(),
                                                       attach_to=pedestrian)
                    if controller is not None:
                        self.walker_controllers.append(controller)
                        logger.debug(f"Spawned pedestrian at {spawn_point.location}")
                else:
                    logger.warning(f"Failed to spawn pedestrian at {spawn_point.location}")

            except Exception as e:
                logger.error(f"Error spawning pedestrian: {e}")

    def _start_walker_controllers(self, spawn_points):
        """Start walker AI controllers — must be called after world.tick()."""
        for controller in self.walker_controllers:
            try:
                destination = random.choice(spawn_points).location
                controller.start()
                controller.go_to_location(destination)
                controller.set_max_speed(1.0 + random.random() * 2.0)
            except Exception as e:
                logger.error(f"Error starting walker controller: {e}")

    def update_pedestrians(self):
        """Update pedestrian destinations periodically"""
        for controller in self.walker_controllers:
            try:
                # Check if pedestrian reached destination
                if controller.is_at_goal():
                    # Set new random destination
                    spawn_points = self.world.get_map().get_spawn_points()
                    if spawn_points:
                        destination = random.choice(spawn_points).location
                        controller.go_to_location(destination)
            except Exception as e:
                logger.error(f"Error updating pedestrian: {e}")

    def get_actor_info(self):
        """
        Get information about spawned actors

        Returns:
            dict: Actor statistics
        """
        return {
            'vehicles': len(self.vehicle_list),
            'pedestrians': len(self.pedestrian_list),
            'controllers': len(self.walker_controllers)
        }

    def cleanup(self):
        """Destroy all spawned actors"""
        logger.info("Cleaning up traffic actors...")

        # Destroy vehicles
        for vehicle in self.vehicle_list:
            try:
                vehicle.destroy()
            except Exception as e:
                logger.error(f"Error destroying vehicle: {e}")

        # Destroy pedestrians and controllers
        for pedestrian in self.pedestrian_list:
            try:
                pedestrian.destroy()
            except Exception as e:
                logger.error(f"Error destroying pedestrian: {e}")

        for controller in self.walker_controllers:
            try:
                controller.destroy()
            except Exception as e:
                logger.error(f"Error destroying controller: {e}")

        self.vehicle_list.clear()
        self.pedestrian_list.clear()
        self.walker_controllers.clear()

        logger.info("Traffic cleanup completed")