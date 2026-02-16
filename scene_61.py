import os
import time
from queue import Queue, Empty

import carla
import cv2
import numpy as np
from dotenv import load_dotenv

from PCLA import PCLA

load_dotenv()

# ===================== CONFIG =====================

# Networking
HOST = os.getenv("CARLA_HOST", "127.0.0.1")
PORT = int(os.getenv("CARLA_PORT", 2000))
CLIENT_TIMEOUT = 20.0
TM_PORT = 8000

# Simulation
MAP_NAME = "Town03"
FIXED_DELTA = 0.05
FPS = int(1.0 / FIXED_DELTA)
TOTAL_DURATION_SEC = 15.0

# Video
VIDEO_PATH = "/home/joshua/CPX/scene.mp4"
IMAGE_W = 1280
IMAGE_H = 720
FOV = 90

# PCLA
AGENT_NAME = "carl_carlv11"
ROUTE_XML = "./sample_route.xml"

# Ego spawn
EGO_SPAWN_INDEX = 31

# Scenario behavior
LEAD_TRUCK_DISTANCE_AHEAD = 12.0
STOPPED_VEHICLE_DISTANCE_AHEAD = 32.0
LANE_CHANGE_TIME = 6.0
PREFER_LANE_CHANGE_RIGHT = True

# Chase camera (spectator follows ego; RGB camera attached to spectator)
CAM_BACK = 7.0
CAM_UP = 3.2
CAM_PITCH = -12.0

# Stopped vehicle
STOPPED_BRAKE = 1.0

# ==================================================


def setup_world(client: carla.Client) -> carla.World:
    world = client.load_world(MAP_NAME)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA
    settings.no_rendering_mode = False  # must render for viewers
    world.apply_settings(settings)

    world.tick()
    return world


def get_driving_waypoint(carla_map: carla.Map, loc: carla.Location) -> carla.Waypoint:
    return carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)


def advance_waypoint(wp: carla.Waypoint, distance_m: float, step_m: float = 1.0) -> carla.Waypoint:
    if wp is None:
        return None
    step_m = max(0.5, float(step_m))
    steps = max(1, int(distance_m / step_m))
    cur = wp
    for _ in range(steps):
        nxt = cur.next(step_m)
        if not nxt:
            break
        cur = nxt[0]
    return cur


def choose_lane_change_direction(lead_wp: carla.Waypoint, prefer_right: bool) -> bool:
    probe = lead_wp
    for _ in range(20):
        if probe is None:
            break
        right = probe.get_right_lane()
        left = probe.get_left_lane()

        if prefer_right:
            if right and right.lane_type == carla.LaneType.Driving:
                return True
            if left and left.lane_type == carla.LaneType.Driving:
                return False
        else:
            if left and left.lane_type == carla.LaneType.Driving:
                return False
            if right and right.lane_type == carla.LaneType.Driving:
                return True

        nxt = probe.next(2.0)
        if not nxt:
            break
        probe = nxt[0]
    return True


def pick_lead_truck_bp(blueprints: carla.BlueprintLibrary) -> carla.ActorBlueprint:
    patterns = [
        "vehicle.carlamotors.carlacola",
        "vehicle.*carlacola*",
        "vehicle.*truck*",
        "vehicle.*",
    ]
    candidates = []
    seen = set()
    for pat in patterns:
        for bp in blueprints.filter(pat):
            if bp.id not in seen:
                candidates.append(bp)
                seen.add(bp.id)

    def score(bp):
        tid = bp.id.lower()
        if "carlacola" in tid:
            return 0
        if "truck" in tid:
            return 1
        return 10

    candidates.sort(key=lambda b: (score(b), b.id))
    return candidates[0] if candidates else None


def try_spawn_vehicle(world: carla.World, bp: carla.ActorBlueprint, tf: carla.Transform, z_offset: float = 0.5) -> carla.Actor:
    t = carla.Transform(
        carla.Location(tf.location.x, tf.location.y, tf.location.z + z_offset),
        tf.rotation,
    )
    return world.try_spawn_actor(bp, t)


def spawn_lead_truck(world: carla.World, blueprints: carla.BlueprintLibrary, start_wp: carla.Waypoint) -> carla.Vehicle:
    if start_wp is None:
        return None
    bp = pick_lead_truck_bp(blueprints)
    if bp is None:
        return None

    probe = start_wp
    for _ in range(30):
        actor = try_spawn_vehicle(world, bp, probe.transform, z_offset=0.8)
        if actor and isinstance(actor, carla.Vehicle):
            return actor
        if actor:
            try:
                actor.destroy()
            except Exception:
                pass
        nxt = probe.next(2.0)
        if not nxt:
            break
        probe = nxt[0]
    return None


def spawn_stopped_vehicle(world: carla.World, blueprints: carla.BlueprintLibrary, start_wp: carla.Waypoint) -> carla.Vehicle:
    if start_wp is None:
        return None

    pref_patterns = ["vehicle.audi.a2", "vehicle.toyota.prius", "vehicle.tesla.model3", "vehicle.*"]
    pref = []
    for pat in pref_patterns:
        pref = list(blueprints.filter(pat))
        if pref:
            break

    probe = start_wp
    for _ in range(35):
        for bp in pref[:40]:
            actor = try_spawn_vehicle(world, bp, probe.transform, z_offset=0.6)
            if actor and isinstance(actor, carla.Vehicle):
                actor.set_autopilot(False)
                actor.set_simulate_physics(True)
                actor.apply_control(carla.VehicleControl(throttle=0.0, brake=STOPPED_BRAKE, hand_brake=True))
                return actor
            if actor:
                try:
                    actor.destroy()
                except Exception:
                    pass
        nxt = probe.next(2.0)
        if not nxt:
            break
        probe = nxt[0]
    return None


def drain_image_queue(image_queue: Queue, video: cv2.VideoWriter, max_frames: int = 10) -> None:
    count = 0
    while count < max_frames:
        try:
            frame = image_queue.get_nowait()
        except Empty:
            break
        video.write(frame)
        count += 1


def safe_destroy(client: carla.Client, actors) -> None:
    ids = []
    for a in actors:
        try:
            if a is not None:
                ids.append(a.id)
        except Exception:
            pass
    if not ids:
        return
    try:
        client.apply_batch([carla.command.DestroyActor(x) for x in ids])
    except Exception:
        # Fallback best-effort
        for a in actors:
            try:
                if a is not None:
                    a.destroy()
            except Exception:
                pass


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(CLIENT_TIMEOUT)

    world = setup_world(client)
    carla_map = world.get_map()
    blueprints = world.get_blueprint_library()

    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(7)

    ego = None
    lead_truck = None
    stopped_vehicle = None
    camera = None
    pcla = None
    video = None

    spectator = world.get_spectator()
    image_queue: Queue = Queue()

    try:
        # ---------- Spawn ego ----------
        ego_bp_list = blueprints.filter("vehicle.tesla.model3")
        ego_bp = ego_bp_list[0] if ego_bp_list else blueprints.filter("vehicle.*")[0]
        if ego_bp.has_attribute("role_name"):
            ego_bp.set_attribute("role_name", "hero")

        spawn_points = carla_map.get_spawn_points()
        ego_spawn = spawn_points[EGO_SPAWN_INDEX] if len(spawn_points) > EGO_SPAWN_INDEX else spawn_points[0]
        ego = world.spawn_actor(ego_bp, ego_spawn)
        world.tick()

        # ---------- PCLA ----------
        pcla = PCLA(AGENT_NAME, ego, ROUTE_XML, client)

        # ---------- Camera (attach to spectator, as in sample) ----------
        cam_bp = blueprints.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMAGE_W))
        cam_bp.set_attribute("image_size_y", str(IMAGE_H))
        cam_bp.set_attribute("fov", str(FOV))

        camera = world.spawn_actor(cam_bp, carla.Transform(), attach_to=spectator)

        def camera_callback(image: carla.Image) -> None:
            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            arr = arr.reshape((image.height, image.width, 4))
            image_queue.put(arr[:, :, :3])

        camera.listen(camera_callback)

        video = cv2.VideoWriter(
            VIDEO_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (IMAGE_W, IMAGE_H),
        )

        # ---------- Plan and spawn scenario actors ----------
        ego_wp0 = get_driving_waypoint(carla_map, ego.get_location())
        if ego_wp0 is None:
            raise RuntimeError("Could not find ego waypoint on a driving lane")

        lead_wp = advance_waypoint(ego_wp0, LEAD_TRUCK_DISTANCE_AHEAD, step_m=1.0)
        stopped_wp = advance_waypoint(ego_wp0, STOPPED_VEHICLE_DISTANCE_AHEAD, step_m=1.0)

        lane_change_right = choose_lane_change_direction(lead_wp, PREFER_LANE_CHANGE_RIGHT)

        lead_truck = spawn_lead_truck(world, blueprints, lead_wp)
        if lead_truck is None:
            raise RuntimeError("Failed to spawn lead truck")

        lead_truck.set_simulate_physics(True)
        lead_truck.set_autopilot(True, tm.get_port())
        tm.auto_lane_change(lead_truck, False)
        tm.distance_to_leading_vehicle(lead_truck, 2.0)
        tm.vehicle_percentage_speed_difference(lead_truck, 10.0)  # slightly slower than limit
        world.tick()

        stopped_vehicle = spawn_stopped_vehicle(world, blueprints, stopped_wp)
        if stopped_vehicle is None:
            raise RuntimeError("Failed to spawn stopped vehicle")
        world.tick()

        # ---------- Main loop (15 seconds) ----------
        start_t = world.get_snapshot().timestamp.elapsed_seconds
        lane_change_forced = False

        # Put spectator behind ego immediately to ensure viewers see the correct framing
        ego_tf = ego.get_transform()
        forward = ego_tf.get_forward_vector()
        cam_loc = ego_tf.location - forward * CAM_BACK + carla.Location(z=CAM_UP)
        cam_rot = carla.Rotation(pitch=CAM_PITCH, yaw=ego_tf.rotation.yaw, roll=0.0)
        spectator.set_transform(carla.Transform(cam_loc, cam_rot))
        world.tick()

        while True:
            now_t = world.get_snapshot().timestamp.elapsed_seconds
            elapsed = now_t - start_t
            if elapsed >= TOTAL_DURATION_SEC:
                break

            # Ego control via PCLA
            ego_control = pcla.get_action()
            ego.apply_control(ego_control)

            # Keep stopped vehicle stopped
            if stopped_vehicle is not None:
                stopped_vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, brake=STOPPED_BRAKE, hand_brake=True)
                )

            # Force truck lane change at set time to reveal stopped vehicle ahead
            if (not lane_change_forced) and (elapsed >= LANE_CHANGE_TIME):
                tm.auto_lane_change(lead_truck, True)
                tm.force_lane_change(lead_truck, lane_change_right)  # True=right, False=left
                lane_change_forced = True

            # Chase camera: above/behind ego, forward-facing aligned to ego yaw
            ego_tf = ego.get_transform()
            forward = ego_tf.get_forward_vector()
            cam_loc = ego_tf.location - forward * CAM_BACK + carla.Location(z=CAM_UP)
            cam_rot = carla.Rotation(pitch=CAM_PITCH, yaw=ego_tf.rotation.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

            world.tick()
            drain_image_queue(image_queue, video, max_frames=50)

        # Final drain to capture last frames
        drain_image_queue(image_queue, video, max_frames=200)

    finally:
        # Stop sensor first to avoid callbacks during destruction
        try:
            if camera is not None:
                camera.stop()
        except Exception:
            pass

        # Flush a couple ticks to let sensor pipeline settle (helps avoid aborts)
        try:
            for _ in range(2):
                world.tick()
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

        # Batch-destroy actors to reduce "not found" / race issues
        safe_destroy(client, [camera, stopped_vehicle, lead_truck, ego])

        # Restore async settings
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            settings.no_rendering_mode = False
            world.apply_settings(settings)
        except Exception:
            pass

        try:
            tm.set_synchronous_mode(False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
