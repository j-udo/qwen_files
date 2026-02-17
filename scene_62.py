import os
import time
from queue import Queue

import carla
import cv2
import numpy as np

from PCLA import PCLA
from dotenv import load_dotenv

load_dotenv()

# ===================== CONFIG =====================

# Networking
HOST = os.getenv("CARLA_HOST", "127.0.0.1")
PORT = int(os.getenv("CARLA_PORT", 2000))

# Simulation
MAP_NAME = "Town03"
FIXED_DELTA = 0.05
FPS = int(1 / FIXED_DELTA)
TERMINATE_AFTER = 15.0  # seconds

# Video
VIDEO_PATH = "/home/joshua/CPX/scene.mp4"
IMAGE_W = 1280
IMAGE_H = 720
FOV = 90

# Scenario timing (relative to scenario start)
LEAD_SPAWN_AFTER = 1.0
OBSTACLE_SPAWN_AFTER = 4.0
LEAD_BRAKE_AFTER = 6.5
LEAD_SLOW_DURATION = 3.0

# Spacing (meters)
LEAD_DISTANCE_AHEAD = 22.0
OBSTACLE_DISTANCE_AHEAD = 45.0

# Lead speeds (m/s)
LEAD_CRUISE_SPEED = 9.0
LEAD_SLOW_SPEED = 2.0
LEAD_RESUME_SPEED = 8.0

# Turning vehicle behavior
TURN_CROSS_SPEED = 6.0
TURN_CROSS_OFFSET = 8.0
TURN_TRAVEL_TIME = 4.0

# Chase camera (spectator)
CAM_BACK = 7.0
CAM_UP = 3.5
CAM_PITCH = -10.0

# Dusk + glare weather
SUN_ALTITUDE = 4.0
CLOUDINESS = 15.0
FOG_DENSITY = 10.0
FOG_DISTANCE = 35.0
SCATTERING_INTENSITY = 1.2
MIE_SCATTERING_SCALE = 8.0
RAYLEIGH_SCATTERING_SCALE = 0.06

# ==================================================


def setup_world(client: carla.Client) -> carla.World:
    available = []
    try:
        available = client.get_available_maps()
    except Exception:
        available = []

    desired_paths = {MAP_NAME, f"/Game/Carla/Maps/{MAP_NAME}"}
    should_load = True
    if available:
        should_load = any(m in desired_paths for m in available)

    if should_load:
        print(f"[INFO] Loading map: {MAP_NAME}")
        world = client.load_world(MAP_NAME)
    else:
        print(f"[WARN] Map {MAP_NAME} not listed by server; using current world instead")
        world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA
    settings.no_rendering_mode = False  # must render
    world.apply_settings(settings)

    # If a TrafficManager is running, keep it in sync mode to avoid sync warnings/freezes
    try:
        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(True)
    except Exception:
        pass

    world.tick()
    return world


def set_dusk_glare_weather(world: carla.World, ego_yaw: float) -> None:
    # Put sun "in front" of ego by setting azimuth close to ego yaw (degrees).
    az = ego_yaw % 360.0
    weather = carla.WeatherParameters(
        cloudiness=float(CLOUDINESS),
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=5.0,
        sun_azimuth_angle=float(az),
        sun_altitude_angle=float(SUN_ALTITUDE),
        fog_density=float(FOG_DENSITY),
        fog_distance=float(FOG_DISTANCE),
        wetness=0.0,
        scattering_intensity=float(SCATTERING_INTENSITY),
        mie_scattering_scale=float(MIE_SCATTERING_SCALE),
        rayleigh_scattering_scale=float(RAYLEIGH_SCATTERING_SCALE),
    )
    world.set_weather(weather)


def pick_vehicle_bp(blueprints: carla.BlueprintLibrary, preferred_ids, fallback_filter="vehicle.*"):
    for vid in preferred_ids:
        bps = blueprints.filter(vid)
        if bps:
            return bps[0]
    fb = blueprints.filter(fallback_filter)
    return fb[0] if fb else None


def try_find_blueprint(blueprints: carla.BlueprintLibrary, blueprint_id: str):
    try:
        return blueprints.find(blueprint_id)
    except Exception:
        return None


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(60.0)  # loading a world can take longer than 20s on some systems

    world = setup_world(client)
    blueprints = world.get_blueprint_library()
    carla_map = world.get_map()

    ego = None
    lead = None
    turning = None
    obstacle = None
    camera = None
    video = None
    pcla = None

    image_queue: Queue = Queue()

    try:
        # ---------- Spawn Ego ----------
        ego_bp = None
        for q in ["vehicle.tesla.model3", "model3", "vehicle.*model3*"]:
            bps = blueprints.filter(q)
            if bps:
                ego_bp = bps[0]
                break
        if ego_bp is None:
            ego_bp = blueprints.filter("vehicle.*")[0]

        if ego_bp.has_attribute("role_name"):
            ego_bp.set_attribute("role_name", "hero")

        spawn_points = carla_map.get_spawn_points()
        ego_spawn = spawn_points[31] if len(spawn_points) > 31 else spawn_points[0]
        ego = world.try_spawn_actor(ego_bp, ego_spawn)
        if ego is None:
            ego = world.spawn_actor(ego_bp, ego_spawn)

        world.tick()

        # Set dusk weather with sun in front to create glare
        ego_yaw = ego.get_transform().rotation.yaw
        set_dusk_glare_weather(world, ego_yaw)

        # ---------- PCLA ----------
        pcla = PCLA("carl_carlv11", ego, "./sample_route.xml", client)
        print("[INFO] Ego spawned, PCLA running")

        # ---------- Spectator + RGB camera (attached to ego with rigid mount) ----------
        spectator = world.get_spectator()

        cam_bp = blueprints.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMAGE_W))
        cam_bp.set_attribute("image_size_y", str(IMAGE_H))
        cam_bp.set_attribute("fov", str(FOV))
        if cam_bp.has_attribute("sensor_tick"):
            cam_bp.set_attribute("sensor_tick", str(FIXED_DELTA))

        # Camera should be slightly above/behind ego, capturing rear with forward-facing view aligned to ego
        cam_rel_tf = carla.Transform(
            carla.Location(x=-CAM_BACK, z=CAM_UP),
            carla.Rotation(pitch=CAM_PITCH, yaw=0.0, roll=0.0),
        )
        camera = world.spawn_actor(cam_bp, cam_rel_tf, attach_to=ego, attachment_type=carla.AttachmentType.Rigid)

        def camera_callback(image: carla.Image) -> None:
            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            arr = arr.reshape((image.height, image.width, 4))
            image_queue.put(arr[:, :, :3])

        camera.listen(camera_callback)

        os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
        video = cv2.VideoWriter(
            VIDEO_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (IMAGE_W, IMAGE_H),
        )

        # ---------- Scenario state ----------
        start_ts = world.get_snapshot().timestamp.elapsed_seconds
        lead_slow_start = None
        turning_start_ts = None

        # ---------- Main loop ----------
        while True:
            snapshot = world.get_snapshot()
            now_ts = snapshot.timestamp.elapsed_seconds
            t = now_ts - start_ts
            if t >= TERMINATE_AFTER:
                print("[INFO] Terminating scenario (15s reached)")
                break

            # Ego control by PCLA agent
            ego_control = pcla.get_action()
            ego.apply_control(ego_control)

            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            ego_rot = ego_tf.rotation
            forward = ego_tf.get_forward_vector()
            right = ego_tf.get_right_vector()

            # Keep glare roughly aligned with ego heading (update occasionally)
            if int(t * 2) != int((t - FIXED_DELTA) * 2):
                set_dusk_glare_weather(world, ego_rot.yaw)

            # Keep spectator aligned with ego perspective (viewer camera in simulator window)
            spec_loc = ego_loc - forward * CAM_BACK + carla.Location(z=CAM_UP)
            spec_rot = carla.Rotation(pitch=CAM_PITCH, yaw=ego_rot.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(spec_loc, spec_rot))

            # Spawn lead vehicle ahead (dark-colored)
            if lead is None and t >= LEAD_SPAWN_AFTER:
                lead_bp = pick_vehicle_bp(
                    blueprints,
                    preferred_ids=[
                        "vehicle.lincoln.mkz_2020",
                        "vehicle.audi.a2",
                        "vehicle.audi.tt",
                        "vehicle.bmw.grandtourer",
                        "vehicle.mercedes.coupe",
                        "vehicle.toyota.prius",
                    ],
                )
                if lead_bp is not None and lead_bp.has_attribute("color"):
                    lead_bp.set_attribute("color", "10,10,10")
                if lead_bp is not None and lead_bp.has_attribute("role_name"):
                    lead_bp.set_attribute("role_name", "scenario_lead")

                spawn_loc = ego_loc + forward * LEAD_DISTANCE_AHEAD
                spawn_loc.z += 0.3
                lead_tf = carla.Transform(spawn_loc, carla.Rotation(yaw=ego_rot.yaw))
                lead = world.try_spawn_actor(lead_bp, lead_tf) if lead_bp else None
                if lead:
                    lead.set_autopilot(False)
                    lead.set_simulate_physics(True)
                    print("[EVENT] Lead vehicle spawned")

            # Spawn an obstacle ahead to justify braking (static prop or parked vehicle)
            if obstacle is None and t >= OBSTACLE_SPAWN_AFTER:
                prop_bp = None
                for prop_id in [
                    "static.prop.trafficcone01",
                    "static.prop.constructioncone",
                    "static.prop.streetbarrier",
                    "static.prop.barrierwork05",
                ]:
                    bp = try_find_blueprint(blueprints, prop_id)
                    if bp is not None:
                        prop_bp = bp
                        break

                if prop_bp is None:
                    veh_bps = blueprints.filter("vehicle.*")
                    prop_bp = veh_bps[0] if veh_bps else None

                if prop_bp is not None and prop_bp.has_attribute("role_name"):
                    prop_bp.set_attribute("role_name", "scenario_obstacle")

                obs_loc = ego_loc + forward * OBSTACLE_DISTANCE_AHEAD
                obs_loc.z += 0.2
                obs_tf = carla.Transform(obs_loc, carla.Rotation(yaw=ego_rot.yaw))
                obstacle = world.try_spawn_actor(prop_bp, obs_tf) if prop_bp else None
                if obstacle:
                    try:
                        obstacle.set_simulate_physics(False)
                    except Exception:
                        pass
                    print("[EVENT] Obstacle spawned")

            # Spawn a crossing/turning vehicle near the obstacle to add distraction
            if turning is None and t >= OBSTACLE_SPAWN_AFTER and lead is not None:
                turn_bp = pick_vehicle_bp(
                    blueprints,
                    preferred_ids=[
                        "vehicle.nissan.patrol",
                        "vehicle.jeep.wrangler_rubicon",
                        "vehicle.audi.etron",
                    ],
                )
                if turn_bp is not None and turn_bp.has_attribute("color"):
                    turn_bp.set_attribute("color", "220,220,220")
                if turn_bp is not None and turn_bp.has_attribute("role_name"):
                    turn_bp.set_attribute("role_name", "scenario_turning")

                turn_spawn = ego_loc + forward * (OBSTACLE_DISTANCE_AHEAD - 8.0) + right * TURN_CROSS_OFFSET
                turn_spawn.z += 0.3
                turning_tf = carla.Transform(turn_spawn, carla.Rotation(yaw=ego_rot.yaw - 90.0))
                turning = world.try_spawn_actor(turn_bp, turning_tf) if turn_bp else None
                if turning:
                    turning.set_autopilot(False)
                    turning.set_simulate_physics(True)
                    turning_start_ts = now_ts
                    print("[EVENT] Turning/crossing vehicle spawned")

            # Turning vehicle motion (crossing leftwards across road)
            if turning and turning_start_ts is not None:
                dt = now_ts - turning_start_ts
                if dt <= TURN_TRAVEL_TIME:
                    cross_dir = carla.Vector3D(x=-right.x, y=-right.y, z=0.0)
                    turning.set_target_velocity(cross_dir * TURN_CROSS_SPEED)
                else:
                    turning.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))

            # Lead sudden slowdown (harder to judge distance due to glare + dark lead car)
            if lead and t >= LEAD_BRAKE_AFTER and lead_slow_start is None:
                lead_slow_start = now_ts
                print("[EVENT] Lead sudden slowdown")

            if lead:
                if lead_slow_start is None:
                    lead.set_target_velocity(forward * LEAD_CRUISE_SPEED)
                else:
                    slow_t = now_ts - lead_slow_start
                    if slow_t <= LEAD_SLOW_DURATION:
                        lead.set_target_velocity(forward * LEAD_SLOW_SPEED)
                    else:
                        lead.set_target_velocity(forward * LEAD_RESUME_SPEED)

            # Tick & record
            world.tick()

            while not image_queue.empty():
                frame = image_queue.get()
                video.write(frame)

    finally:
        print("[INFO] Cleaning up")

        try:
            if camera:
                camera.stop()
                camera.destroy()
        except Exception:
            pass

        try:
            if video:
                video.release()
                print(f"[INFO] Video saved: {VIDEO_PATH}")
        except Exception:
            pass

        try:
            if pcla:
                pcla.cleanup()
        except Exception:
            pass

        for actor in [turning, obstacle, lead, ego]:
            try:
                if actor:
                    actor.destroy()
            except Exception:
                pass

        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            settings.no_rendering_mode = False
            world.apply_settings(settings)
        except Exception:
            pass

        try:
            tm = client.get_trafficmanager()
            tm.set_synchronous_mode(False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
