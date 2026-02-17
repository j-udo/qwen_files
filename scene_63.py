import os
from queue import Queue

import carla
import cv2
import numpy as np

from PCLA import PCLA
from dotenv import load_dotenv

load_dotenv()

# ===================== CONFIG =====================

HOST = os.getenv("CARLA_HOST", "127.0.0.1")
PORT = int(os.getenv("CARLA_PORT", 2000))

MAP_NAME = "Town03"
FIXED_DELTA = 0.05
FPS = int(1 / FIXED_DELTA)
TERMINATE_AFTER = 15.0

VIDEO_PATH = "/home/joshua/CPX/scene.mp4"
IMAGE_W = 1280
IMAGE_H = 720
FOV = 90

# Camera placement: slightly above and behind ego, forward-facing aligned with ego
CAM_BACK = 7.0
CAM_UP = 3.2
CAM_PITCH = -10.0

# Scenario timing
LEAD_SPAWN_AFTER = 0.8
TURNER_SPAWN_AFTER = 3.0
LEAD_BRAKE_AFTER = 5.5
LEAD_SLOW_DURATION = 3.0

# Distances (meters)
LEAD_DISTANCE_AHEAD = 28.0
TURNER_DISTANCE_AHEAD = 55.0
TURNER_RIGHT_OFFSET = 10.0

# Speeds (m/s)
LEAD_CRUISE_SPEED = 11.0
LEAD_SLOW_SPEED = 1.8
LEAD_RESUME_SPEED = 9.0

TURNER_CROSS_SPEED = 5.0
TURNER_TRAVEL_TIME = 3.8

# Dusk + glare tuning (avoid heavy fog; use haze/halo via mie scattering)
SUN_ALTITUDE = 2.0
CLOUDINESS = 10.0
FOG_DENSITY = 0.0
FOG_DISTANCE = 0.0
WETNESS = 0.0
SCATTERING_INTENSITY = 1.8
MIE_SCATTERING_SCALE = 12.0
RAYLEIGH_SCATTERING_SCALE = 0.035

# ==================================================


def setup_world(client: carla.Client) -> carla.World:
    world = client.load_world(MAP_NAME)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA
    settings.no_rendering_mode = False  # must render
    world.apply_settings(settings)

    # Keep TM in sync if present
    try:
        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(True)
    except Exception:
        pass

    world.tick()
    return world


def set_dusk_glare_weather(world: carla.World, sun_azimuth: float) -> None:
    weather = carla.WeatherParameters(
        cloudiness=float(CLOUDINESS),
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=5.0,
        sun_azimuth_angle=float(sun_azimuth % 360.0),
        sun_altitude_angle=float(SUN_ALTITUDE),
        fog_density=float(FOG_DENSITY),
        fog_distance=float(FOG_DISTANCE),
        wetness=float(WETNESS),
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


def safe_set_attr(bp: carla.ActorBlueprint, key: str, value: str) -> None:
    try:
        if bp.has_attribute(key):
            bp.set_attribute(key, value)
    except Exception:
        pass


def spawn_vehicle(world: carla.World, bp: carla.ActorBlueprint, tf: carla.Transform):
    actor = world.try_spawn_actor(bp, tf)
    if actor is None:
        actor = world.spawn_actor(bp, tf)
    return actor


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(60.0)

    world = setup_world(client)
    carla_map = world.get_map()
    blueprints = world.get_blueprint_library()

    ego = None
    lead = None
    turner = None
    camera = None
    video = None
    pcla = None

    spectator = world.get_spectator()
    image_queue: Queue = Queue()

    lead_slow_start_ts = None
    turner_start_ts = None

    try:
        # ---------- Spawn Ego ----------
        ego_bp = pick_vehicle_bp(blueprints, ["vehicle.tesla.model3", "vehicle.*model3*"])
        safe_set_attr(ego_bp, "role_name", "hero")

        spawn_points = carla_map.get_spawn_points()
        ego_spawn = spawn_points[31] if len(spawn_points) > 31 else spawn_points[0]
        ego = spawn_vehicle(world, ego_bp, ego_spawn)
        world.tick()

        # Dusk weather; put sun directly in front of ego (toward camera view)
        ego_yaw = ego.get_transform().rotation.yaw
        set_dusk_glare_weather(world, ego_yaw)

        # ---------- PCLA ----------
        pcla = PCLA("carl_carlv11", ego, "./sample_route.xml", client)

        # ---------- RGB camera attached to ego (rear chase, forward-facing aligned) ----------
        cam_bp = blueprints.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMAGE_W))
        cam_bp.set_attribute("image_size_y", str(IMAGE_H))
        cam_bp.set_attribute("fov", str(FOV))
        if cam_bp.has_attribute("sensor_tick"):
            cam_bp.set_attribute("sensor_tick", str(FIXED_DELTA))

        cam_rel_tf = carla.Transform(
            carla.Location(x=-CAM_BACK, z=CAM_UP),
            carla.Rotation(pitch=CAM_PITCH, yaw=0.0, roll=0.0),
        )
        camera = world.spawn_actor(
            cam_bp,
            cam_rel_tf,
            attach_to=ego,
            attachment_type=carla.AttachmentType.Rigid,
        )

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

        start_ts = world.get_snapshot().timestamp.elapsed_seconds

        # ---------- Main loop ----------
        while True:
            snapshot = world.get_snapshot()
            now_ts = snapshot.timestamp.elapsed_seconds
            t = now_ts - start_ts
            if t >= TERMINATE_AFTER:
                break

            # Ego controlled by PCLA
            ego_control = pcla.get_action()
            ego.apply_control(ego_control)

            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            ego_rot = ego_tf.rotation
            forward = ego_tf.get_forward_vector()
            right = ego_tf.get_right_vector()

            # Keep sun aligned with ego heading for persistent "driving into the sun" glare
            if int(t * 2) != int((t - FIXED_DELTA) * 2):
                set_dusk_glare_weather(world, ego_rot.yaw)

            # Spectator follows same chase viewpoint (for simulator window viewers)
            spec_loc = ego_loc - forward * CAM_BACK + carla.Location(z=CAM_UP)
            spec_rot = carla.Rotation(pitch=CAM_PITCH, yaw=ego_rot.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(spec_loc, spec_rot))

            # ---------- Spawn dark lead vehicle in same lane ahead ----------
            if lead is None and t >= LEAD_SPAWN_AFTER:
                lead_bp = pick_vehicle_bp(
                    blueprints,
                    preferred_ids=[
                        "vehicle.lincoln.mkz_2020",
                        "vehicle.audi.tt",
                        "vehicle.bmw.grandtourer",
                        "vehicle.mercedes.coupe",
                        "vehicle.toyota.prius",
                        "vehicle.audi.a2",
                    ],
                )
                safe_set_attr(lead_bp, "role_name", "scenario_lead")
                safe_set_attr(lead_bp, "color", "5,5,5")  # dark-colored

                # Place on ego's current driving lane center to avoid intersection weirdness
                ego_wp = carla_map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                lead_wp = ego_wp.next(LEAD_DISTANCE_AHEAD)[0] if ego_wp is not None else None
                lead_tf = lead_wp.transform if lead_wp is not None else carla.Transform(
                    ego_loc + forward * LEAD_DISTANCE_AHEAD, carla.Rotation(yaw=ego_rot.yaw)
                )
                lead_tf.location.z += 0.3

                lead = world.try_spawn_actor(lead_bp, lead_tf)
                if lead:
                    lead.set_autopilot(False)
                    lead.set_simulate_physics(True)
                    # Reduce chance of chaotic collisions by making it a bit stable
                    try:
                        phys = lead.get_physics_control()
                        phys.mass = max(phys.mass, 1600.0)
                        lead.apply_physics_control(phys)
                    except Exception:
                        pass

            # ---------- Spawn a turning/crossing vehicle near the future conflict point ----------
            # This vehicle crosses ahead of the lead (not the ego) to justify lead braking.
            if turner is None and t >= TURNER_SPAWN_AFTER and lead is not None:
                turner_bp = pick_vehicle_bp(
                    blueprints,
                    preferred_ids=[
                        "vehicle.nissan.patrol",
                        "vehicle.jeep.wrangler_rubicon",
                        "vehicle.audi.etron",
                        "vehicle.citroen.c3",
                    ],
                )
                safe_set_attr(turner_bp, "role_name", "scenario_turner")
                safe_set_attr(turner_bp, "color", "230,230,230")

                ego_wp = carla_map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                target_wp = ego_wp.next(TURNER_DISTANCE_AHEAD)[0] if ego_wp is not None else None
                target_loc = target_wp.transform.location if target_wp is not None else (ego_loc + forward * TURNER_DISTANCE_AHEAD)

                # Spawn offset to the right, oriented to cross the lane (right-to-left across ego heading)
                spawn_loc = target_loc + right * TURNER_RIGHT_OFFSET
                spawn_loc.z += 0.3
                turner_tf = carla.Transform(spawn_loc, carla.Rotation(yaw=ego_rot.yaw - 90.0))

                turner = world.try_spawn_actor(turner_bp, turner_tf)
                if turner:
                    turner.set_autopilot(False)
                    turner.set_simulate_physics(True)
                    turner_start_ts = now_ts

                    # Reduce collision chaos: ignore collision between lead and turner (still visually plausible slowing)
                    try:
                        tm = client.get_trafficmanager()
                        tm.collision_detection(lead, turner, False)
                    except Exception:
                        pass

            # Turner motion: crosses the road in front of lead's path
            if turner and turner_start_ts is not None:
                dt = now_ts - turner_start_ts
                if dt <= TURNER_TRAVEL_TIME:
                    cross_dir = carla.Vector3D(x=-right.x, y=-right.y, z=0.0)
                    turner.set_target_velocity(cross_dir * TURNER_CROSS_SPEED)
                else:
                    turner.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))

            # ---------- Lead behavior: sudden slowdown in-lane (no turning/crash) ----------
            if lead:
                if lead_slow_start_ts is None and t >= LEAD_BRAKE_AFTER:
                    lead_slow_start_ts = now_ts

                if lead_slow_start_ts is None:
                    lead.set_target_velocity(forward * LEAD_CRUISE_SPEED)
                else:
                    slow_t = now_ts - lead_slow_start_ts
                    if slow_t <= LEAD_SLOW_DURATION:
                        lead.set_target_velocity(forward * LEAD_SLOW_SPEED)
                    else:
                        lead.set_target_velocity(forward * LEAD_RESUME_SPEED)

            # Tick & record
            world.tick()
            while not image_queue.empty():
                video.write(image_queue.get())

    finally:
        try:
            if camera:
                camera.stop()
                camera.destroy()
        except Exception:
            pass

        try:
            if video:
                video.release()
        except Exception:
            pass

        try:
            if pcla:
                pcla.cleanup()
        except Exception:
            pass

        for actor in [turner, lead, ego]:
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
