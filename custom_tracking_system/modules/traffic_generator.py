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

        logger.info(f"TrafficGenerator initialized: {num_vehicles} vehicles, {num_pedestrians} pedestrians")

    def spawn_actors(self):
        """
        Spawn vehicles and pedestrians at random spawn points.
        """
        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()

        if len(spawn_points) == 0:
            logger.warning("No spawn points available on map")
            return

        # Spawn vehicles
        self._spawn_vehicles(blueprint_library, spawn_points)

        # Spawn pedestrians va cho di bo bang WalkerControl truc tiep
        self._spawn_pedestrians(blueprint_library, spawn_points)

        # Tick once so CARLA registers all spawned actors
        self.world.tick()

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
                    vehicle.set_autopilot(True, 8050)
                    self.vehicle_list.append(vehicle)
                    logger.debug(f"Spawned vehicle at {spawn_point.location}")
                else:
                    logger.warning(f"Failed to spawn vehicle at {spawn_point.location}")

            except Exception as e:
                logger.error(f"Error spawning vehicle: {e}")

    def _spawn_pedestrians(self, blueprint_library, spawn_points):
        """Spawn pedestrian actors va cho di bo bang WalkerControl truc tiep.

        KHONG dung controller.ai.walker: controller nay goi RPC
        'set_actor_collisions' khong ton tai o server CARLA 0.9.14
        (client la 0.9.15) -> world.tick() loi RPC nay vinh vien
        moi frame sau do, lam camera dung hinh.
        """
        walker_blueprints = blueprint_library.filter('walker.pedestrian.*')

        for i in range(self.num_pedestrians):
            try:
                bp = random.choice(walker_blueprints)
                spawn_point = random.choice(spawn_points)
                pedestrian = self.world.try_spawn_actor(bp, spawn_point)

                if pedestrian is not None:
                    self.pedestrian_list.append(pedestrian)
                    self._walk_to_random_target(pedestrian, spawn_points)
                    logger.debug(f"Spawned pedestrian at {spawn_point.location}")
                else:
                    logger.warning(f"Failed to spawn pedestrian at {spawn_point.location}")

            except Exception as e:
                logger.error(f"Error spawning pedestrian: {e}")

    def _walk_to_random_target(self, pedestrian, spawn_points):
        """Cho pedestrian di bo ve 1 huong ngau nhien bang WalkerControl truc tiep."""
        try:
            loc = pedestrian.get_location()
            target = random.choice(spawn_points).location
            dx = target.x - loc.x
            dy = target.y - loc.y
            dist = max(1e-3, (dx ** 2 + dy ** 2) ** 0.5)
            control = carla.WalkerControl(
                direction=carla.Vector3D(x=dx / dist, y=dy / dist, z=0.0),
                speed=1.0 + random.random() * 1.5,
            )
            pedestrian.apply_control(control)
        except Exception as e:
            logger.error(f"Error setting pedestrian walk control: {e}")

    def update_pedestrians(self):
        """Periodically retarget pedestrians to a new random direction.

        Khong dung WalkerAIController (xem _spawn_pedestrians) — ap lai
        WalkerControl voi huong moi moi ~10s (100 frames @ 10fps).
        """
        self._pedestrian_tick = getattr(self, '_pedestrian_tick', 0) + 1
        if self._pedestrian_tick % 100 != 0:
            return

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            return

        for pedestrian in self.pedestrian_list:
            self._walk_to_random_target(pedestrian, spawn_points)

    def get_actor_info(self):
        """
        Get information about spawned actors

        Returns:
            dict: Actor statistics
        """
        return {
            'vehicles': len(self.vehicle_list),
            'pedestrians': len(self.pedestrian_list),
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

        self.vehicle_list.clear()
        self.pedestrian_list.clear()

        logger.info("Traffic cleanup completed")