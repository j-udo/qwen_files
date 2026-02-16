import os
import time
import random
from queue import Queue

import carla
import cv2
import numpy as np

from PCLA import PCLA, route_maker, location_to_waypoint
from dotenv import load_dotenv

load_dotenv()

# ===================== CONFIG =====================

HOST = os.getenv("CARLA_HOST", "127.0.0.1")
PORT = int(os.getenv("CARLA_PORT", 2000))

MAP_NAME = "Town03"
FIXED_DELTA = 0.05
FPS = int(1 / FIXED_DELTA)
DURATION_SECONDS = 15.0

VIDEO_PATH = "/home/joshua/CPX/scene.mp4"
IMAGE_W = 1280
IMAGE_H = 720
FOV = 90

EGO_SPAWN_INDEX = 31
EGO_END_INDEX = 42
ROUTE_PATH = "/tmp/pcla_route.xml"

# Pedestrian jaywalk event tuning
EVENT_SPAWN_AHEAD = 28.0              # spawn the crossing some distance ahead of ego
EVENT_TRIGGER_AHEAD_MIN = 20.0        # start crossing when the crossing point is within this forward distance
EVENT_TRIGGER_AHEAD_MAX = 24.0        # ...and still ahead (so it starts while ego approaches)
CROSSING_HALF_WIDTH_PAD = 0.6         # extra pad beyond lane half-width so it starts clearly outside the lane
WALK_SPEED = 2.2                      # brisk walk
RUN_SPEED = 3.8                       # if we want to force it more "urgent"
USE_RUN_PROB = 0.35

# Camera chase
CAM_BACK = 7.0
CAM_UP = 3.5
CAM_PITCH = -12.0

# ==================================================


def setup_world(client: carla.Client) -> carla.World:
    world = client.load_world(MAP_NAME)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    world.tick()
    return world


def _dot2(a: carla.Vector3D, b: carla.Vector3D) -> float:
    return a.x * b.x + a.y * b.y


def _try_spawn_jaywalker(world: carla.World, blueprints: carla.BlueprintLibrary, ego: carla.Vehicle):
    m = world.get_map()
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location

    wp_ego = m.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp_ego is None:
        return None

    # Find a waypoint ahead on ego lane (more robust than ego forward vector in curves)
    next_wps = wp_ego.next(EVENT_SPAWN_AHEAD)
    if not next_wps:
        return None
    wp_cross = next_wps[0]

    # Avoid spawning inside junction/crosswalk-like areas; keep it "jaywalk" midblock
    if wp_cross.is_junction:
        for dist in [35.0, 45.0, 55.0]:
            nxt = wp_ego.next(dist)
            if nxt and (not nxt[0].is_junction):
                wp_cross = nxt[0]
                break

    cross_tf = wp_cross.transform
    lane_center = cross_tf.location
    lane_fwd = cross_tf.get_forward_vector()
    lane_right = cross_tf.get_right_vector()

    # Use lane width to cross the full ego lane (and a bit more)
    lane_width = float(getattr(wp_cross, "lane_width", 3.5))
    half = 0.5 * lane_width

    start_loc = lane_center + lane_right * (half + CROSSING_HALF_WIDTH_PAD)
    end_loc = lane_center - lane_right * (half + CROSSING_HALF_WIDTH_PAD)

    start_loc.z += 0.8
    end_loc.z += 0.8

    walker_bps = blueprints.filter("walker.pedestrian.*")
    if not walker_bps:
        return None
    walker_bp = random.choice(walker_bps)
    if walker_bp.has_attribute("is_invincible"):
        walker_bp.set_attribute("is_invincible", "true")

    # Face roughly towards the destination (across the road)
    yaw = cross_tf.rotation.yaw - 90.0  # from right to left across lane (approx)
    walker_tf = carla.Transform(start_loc, carla.Rotation(yaw=yaw))
    walker = world.try_spawn_actor(walker_bp, walker_tf)
    if walker is None:
        return None

    controller_bp = blueprints.find("controller.ai.walker")
    controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)

    event = {
        "walker": walker,
        "controller": controller,
        "wp_cross": wp_cross,
        "cross_center": lane_center,
        "start_loc": start_loc,
        "end_loc": end_loc,
        "started": False,
        "done": False,
        "speed": (RUN_SPEED if random.random() < USE_RUN_PROB else WALK_SPEED),
    }
    return event


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)

    world = setup_world(client)
    blueprints = world.get_blueprint_library()
    spectator = world.get_spectator()

    ego = None
    pcla = None
    camera = None
    video = None
    event = None

    image_queue: Queue = Queue()

    try:
        # ---------- Spawn Ego ----------
        ego_bp = blueprints.filter("vehicle.tesla.model3")[0]
        spawn_points = world.get_map().get_spawn_points()
        ego_spawn = spawn_points[min(EGO_SPAWN_INDEX, len(spawn_points) - 1)]
        ego = world.spawn_actor(ego_bp, ego_spawn)
        world.tick()

        # ---------- Route for PCLA ----------
        start_loc = ego_spawn.location
        end_loc = spawn_points[min(EGO_END_INDEX, len(spawn_points) - 1)].location
        waypoints = location_to_waypoint(client, start_loc, end_loc)
        route_maker(waypoints, ROUTE_PATH)

        # ---------- PCLA agent ----------
        pcla = PCLA("carl_carlv11", ego, ROUTE_PATH, client)

        # ---------- Camera Sensor attached to spectator (spectator is moved as chase cam) ----------
        cam_bp = blueprints.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMAGE_W))
        cam_bp.set_attribute("image_size_y", str(IMAGE_H))
        cam_bp.set_attribute("fov", str(FOV))

        camera = world.spawn_actor(cam_bp, carla.Transform(), attach_to=spectator)

        def camera_callback(image: carla.Image) -> None:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            image_queue.put(array[:, :, :3])

        camera.listen(camera_callback)

        video = cv2.VideoWriter(
            VIDEO_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (IMAGE_W, IMAGE_H),
        )

        sim_start = world.get_snapshot().timestamp.elapsed_seconds

        # Not strictly required for our single scripted walker, but harmless.
        world.set_pedestrians_cross_factor(1.0)

        # Spawn the jaywalker immediately in a deterministic way ahead of ego; if it fails, retry a few times
        for _ in range(20):
            event = _try_spawn_jaywalker(world, blueprints, ego)
            world.tick()
            if event is not None:
                break

        while True:
            now = world.get_snapshot().timestamp.elapsed_seconds
            if now - sim_start >= DURATION_SECONDS:
                break

            # ===== Ego control via PCLA =====
            ego_control = pcla.get_action()
            ego.apply_control(ego_control)

            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            ego_rot = ego_tf.rotation
            ego_fwd = ego_tf.get_forward_vector()

            # ===== Start pedestrian crossing while ego approaches =====
            if event is not None and (not event["started"]):
                cross_center = event["cross_center"]
                to_center = carla.Vector3D(
                    x=cross_center.x - ego_loc.x,
                    y=cross_center.y - ego_loc.y,
                    z=0.0,
                )
                forward_dist = _dot2(to_center, ego_fwd)

                # Start once it's clearly in front of ego but close enough to create interaction
                if EVENT_TRIGGER_AHEAD_MIN <= forward_dist <= EVENT_TRIGGER_AHEAD_MAX:
                    event["controller"].start()
                    event["controller"].set_max_speed(event["speed"])
                    event["controller"].go_to_location(event["end_loc"])
                    event["started"] = True

            # Keep issuing destination to avoid occasional AI hesitation
            if event is not None and event["started"] and (not event["done"]):
                wloc = event["walker"].get_location()
                if wloc.distance(event["end_loc"]) < 1.2:
                    event["done"] = True
                else:
                    event["controller"].go_to_location(event["end_loc"])

            # ===== Chase camera: slightly above and behind ego, aligned forward =====
            cam_loc = ego_loc - ego_fwd * CAM_BACK + carla.Location(z=CAM_UP)
            cam_rot = carla.Rotation(pitch=CAM_PITCH, yaw=ego_rot.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

            # ===== Tick & record =====
            world.tick()

            while not image_queue.empty():
                video.write(image_queue.get())

    finally:
        try:
            if camera is not None:
                camera.stop()
        except Exception:
            pass

        try:
            if video is not None:
                video.release()
        except Exception:
            pass

        try:
            if pcla is not None:
                pcla.cleanup()
        except Exception:
            pass

        if event is not None:
            try:
                if event.get("controller") is not None:
                    event["controller"].stop()
            except Exception:
                pass

            for actor in [event.get("controller"), event.get("walker")]:
                try:
                    if actor is not None:
                        actor.destroy()
                except Exception:
                    pass

        for actor in [ego, camera]:
            try:
                if actor is not None:
                    actor.destroy()
            except Exception:
                pass

        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = 0.0
            world.apply_settings(settings)
        except Exception:
            pass


if __name__ == "__main__":
    main()
