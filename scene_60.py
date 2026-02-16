import os
from queue import Queue

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

# Simulation
MAP_NAME = "Town03"
FIXED_DELTA = 0.05
FPS = int(1 / FIXED_DELTA)
TOTAL_DURATION_SEC = 15.0

# Video
VIDEO_PATH = "/home/joshua/CPX/scene.mp4"
IMAGE_W = 1280
IMAGE_H = 720
FOV = 90

# PCLA
AGENT_NAME = "carl_carlv11"
ROUTE_XML = "./sample_route.xml"

# Scenario: ego follows a large truck that blocks visibility; truck changes lane revealing stopped car
EGO_SPAWN_INDEX = 31

TRUCK_DISTANCE_AHEAD = 10.0          # close enough to occlude view
TRUCK_MIN_EXTENT_X = 4.0             # ensure "large" (bounding box extent x ~ half-length)
TRUCK_MIN_EXTENT_Z = 1.4             # ensure tall enough

TRUCK_LANE_CHANGE_TIME = 7.0
TRUCK_LANE_CHANGE_DURATION = 2.0

# The revealed hazard car (stopped) is ahead in ego lane, initially occluded by truck
STOPPED_DISTANCE_AHEAD = 22.0
STOPPED_BRAKE = 1.0

# Camera (slightly above and behind ego), must show rear of ego and forward-aligned view
CAM_REL_X = -7.0
CAM_REL_Z = 3.0
CAM_PITCH = -10.0

# ==================================================


def setup_world(client: carla.Client) -> carla.World:
    world = client.load_world(MAP_NAME)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA
    settings.no_rendering_mode = False  # MUST render for viewers
    world.apply_settings(settings)
    world.tick()
    return world


def pick_large_truck_blueprint(blueprints: carla.BlueprintLibrary) -> carla.ActorBlueprint:
    # Prefer known big vehicles, then fall back to any "*truck*".
    preferred_patterns = [
        "vehicle.carlamotors.carlacola",
        "vehicle.carlamotors.firetruck",
        "vehicle.*truck*",
    ]

    candidates = []
    for pat in preferred_patterns:
        candidates.extend(list(blueprints.filter(pat)))

    if not candidates:
        candidates = list(blueprints.filter("vehicle.*"))

    def is_large(bp: carla.ActorBlueprint) -> bool:
        # Spawn a temp? Not possible. We'll rely on id hints and then validate after spawn using actor.bounding_box.
        tid = bp.id.lower()
        return ("truck" in tid) or ("carlacola" in tid) or ("firetruck" in tid)

    # Sort to prefer likely large trucks first
    candidates.sort(key=lambda b: (0 if is_large(b) else 1, b.id))
    return candidates[0]


def try_spawn_on_waypoint(world: carla.World, bp: carla.ActorBlueprint, wp: carla.Waypoint, z_offset: float = 0.35) -> carla.Actor:
    tf = wp.transform
    tf.location.z += z_offset
    return world.try_spawn_actor(bp, tf)


def find_adjacent_driving_lane(wp: carla.Waypoint) -> carla.Waypoint:
    # Try right lane first, then left lane
    right = wp.get_right_lane()
    if right and right.lane_type == carla.LaneType.Driving:
        return right
    left = wp.get_left_lane()
    if left and left.lane_type == carla.LaneType.Driving:
        return left
    return None


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(CLIENT_TIMEOUT)

    world = setup_world(client)
    m = world.get_map()
    blueprints = world.get_blueprint_library()

    ego = None
    pcla = None
    camera = None
    video = None
    truck = None
    stopped = None

    image_queue: Queue = Queue()

    try:
        # ---------- Spawn Ego ----------
        ego_bp_list = blueprints.filter("vehicle.tesla.model3")
        ego_bp = ego_bp_list[0] if ego_bp_list else blueprints.filter("vehicle.*")[0]

        spawn_points = m.get_spawn_points()
        ego_spawn = spawn_points[EGO_SPAWN_INDEX] if len(spawn_points) > EGO_SPAWN_INDEX else spawn_points[0]

        ego = world.spawn_actor(ego_bp, ego_spawn)
        world.tick()

        start_sim_t = world.get_snapshot().timestamp.elapsed_seconds

        # ---------- PCLA ----------
        pcla = PCLA(AGENT_NAME, ego, ROUTE_XML, client)

        # ---------- Camera attached behind/above ego ----------
        cam_bp = blueprints.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMAGE_W))
        cam_bp.set_attribute("image_size_y", str(IMAGE_H))
        cam_bp.set_attribute("fov", str(FOV))

        cam_rel_tf = carla.Transform(
            carla.Location(x=CAM_REL_X, z=CAM_REL_Z),
            carla.Rotation(pitch=CAM_PITCH, yaw=0.0, roll=0.0),
        )
        camera = world.spawn_actor(cam_bp, cam_rel_tf, attach_to=ego)

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

        # ---------- Compute lane-based spawn points ----------
        ego_wp0 = m.get_waypoint(ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        if ego_wp0 is None:
            raise RuntimeError("Could not find ego waypoint on driving lane")

        # Truck waypoint ahead in ego lane
        truck_wp = ego_wp0
        for _ in range(max(1, int(TRUCK_DISTANCE_AHEAD))):
            nxt = truck_wp.next(1.0)
            if not nxt:
                break
            truck_wp = nxt[0]

        # Stopped car waypoint farther ahead in ego lane
        stopped_wp = ego_wp0
        for _ in range(max(1, int(STOPPED_DISTANCE_AHEAD))):
            nxt = stopped_wp.next(1.0)
            if not nxt:
                break
            stopped_wp = nxt[0]

        # Adjacent lane waypoint for the truck after lane-change (same longitudinal position as truck_wp)
        adjacent_lane_wp = find_adjacent_driving_lane(truck_wp)

        # ---------- Spawn lead "large truck" ----------
        truck_bp = pick_large_truck_blueprint(blueprints)
        truck = try_spawn_on_waypoint(world, truck_bp, truck_wp, z_offset=0.4)

        # If spawn failed, try a few meters further ahead
        if truck is None:
            tmp_wp = truck_wp
            for _ in range(10):
                nxt = tmp_wp.next(2.0)
                if not nxt:
                    break
                tmp_wp = nxt[0]
                truck = try_spawn_on_waypoint(world, truck_bp, tmp_wp, z_offset=0.4)
                if truck:
                    truck_wp = tmp_wp
                    adjacent_lane_wp = find_adjacent_driving_lane(truck_wp)
                    break

        if truck is None:
            raise RuntimeError("Failed to spawn the lead truck")

        truck.set_autopilot(False)
        truck.set_simulate_physics(False)  # we will teleport along waypoints for deterministic lane change

        # Validate "large" by bounding box; if too small, try other truck-ish blueprints
        if truck.bounding_box.extent.x < TRUCK_MIN_EXTENT_X or truck.bounding_box.extent.z < TRUCK_MIN_EXTENT_Z:
            truck.destroy()
            truck = None
            truck_candidates = list(blueprints.filter("vehicle.*truck*")) + list(blueprints.filter("vehicle.carlamotors.*"))
            spawned = False
            for bp in truck_candidates:
                tr = try_spawn_on_waypoint(world, bp, truck_wp, z_offset=0.4)
                if tr:
                    tr.set_autopilot(False)
                    tr.set_simulate_physics(False)
                    if tr.bounding_box.extent.x >= TRUCK_MIN_EXTENT_X and tr.bounding_box.extent.z >= TRUCK_MIN_EXTENT_Z:
                        truck = tr
                        spawned = True
                        break
                    tr.destroy()
            if not spawned:
                # Last resort: keep the first big-ish actor that spawns
                for bp in truck_candidates:
                    tr = try_spawn_on_waypoint(world, bp, truck_wp, z_offset=0.4)
                    if tr:
                        tr.set_autopilot(False)
                        tr.set_simulate_physics(False)
                        truck = tr
                        break
            if truck is None:
                raise RuntimeError("Failed to spawn a sufficiently large truck-like lead vehicle")

        # ---------- Spawn stopped vehicle in ego lane ----------
        stopped_bp = blueprints.filter("vehicle.*")[0]
        stopped = try_spawn_on_waypoint(world, stopped_bp, stopped_wp, z_offset=0.35)
        if stopped is None:
            # Try a few meters further ahead
            tmp_wp = stopped_wp
            for _ in range(10):
                nxt = tmp_wp.next(2.0)
                if not nxt:
                    break
                tmp_wp = nxt[0]
                stopped = try_spawn_on_waypoint(world, stopped_bp, tmp_wp, z_offset=0.35)
                if stopped:
                    stopped_wp = tmp_wp
                    break

        if stopped is None:
            raise RuntimeError("Failed to spawn the stopped vehicle")

        stopped.set_autopilot(False)
        stopped.set_simulate_physics(True)
        stopped.apply_control(carla.VehicleControl(throttle=0.0, brake=STOPPED_BRAKE, hand_brake=True))

        # ---------- Prepare truck lane change trajectory ----------
        lane_change_started = False
        lane_change_start_t = None
        truck_origin_tf = truck.get_transform()
        truck_dest_tf = None

        if adjacent_lane_wp is None:
            # If no adjacent driving lane at that exact spot, try ahead a little
            tmp_wp = truck_wp
            for _ in range(10):
                nxt = tmp_wp.next(2.0)
                if not nxt:
                    break
                tmp_wp = nxt[0]
                adj = find_adjacent_driving_lane(tmp_wp)
                if adj is not None:
                    truck_wp = tmp_wp
                    truck.set_transform(truck_wp.transform)
                    adjacent_lane_wp = adj
                    break

        if adjacent_lane_wp is not None:
            truck_dest_tf = adjacent_lane_wp.transform
        else:
            # Fallback: lateral shift based on lane width-ish (less ideal, but still reveals)
            ego_tf0 = ego.get_transform()
            right0 = ego_tf0.get_right_vector()
            truck_dest_tf = carla.Transform(
                carla.Location(
                    x=truck_origin_tf.location.x + right0.x * 3.6,
                    y=truck_origin_tf.location.y + right0.y * 3.6,
                    z=truck_origin_tf.location.z,
                ),
                truck_origin_tf.rotation,
            )

        # ---------- Main loop ----------
        while True:
            now_t = world.get_snapshot().timestamp.elapsed_seconds
            elapsed = now_t - start_sim_t
            if elapsed >= TOTAL_DURATION_SEC:
                break

            # Ego control via PCLA
            ego_control = pcla.get_action()
            ego.apply_control(ego_control)

            # Keep stopped car stopped
            if stopped:
                stopped.apply_control(carla.VehicleControl(throttle=0.0, brake=STOPPED_BRAKE, hand_brake=True))

            # Keep truck fixed in ego lane until lane change begins (so ego "follows" it and visibility is blocked)
            if truck and not lane_change_started:
                # Re-anchor to lane center (prevents drift)
                truck.set_transform(truck_origin_tf)

            # Trigger lane change for truck
            if truck and (elapsed >= TRUCK_LANE_CHANGE_TIME):
                if not lane_change_started:
                    lane_change_started = True
                    lane_change_start_t = now_t
                    truck_origin_tf = truck.get_transform()

                phase = (now_t - lane_change_start_t) / max(TRUCK_LANE_CHANGE_DURATION, 1e-3)
                if phase > 1.0:
                    phase = 1.0

                # Smooth lateral shift from origin to destination
                ox, oy, oz = truck_origin_tf.location.x, truck_origin_tf.location.y, truck_origin_tf.location.z
                dx, dy, dz = truck_dest_tf.location.x, truck_dest_tf.location.y, truck_dest_tf.location.z

                new_loc = carla.Location(
                    x=ox + (dx - ox) * phase,
                    y=oy + (dy - oy) * phase,
                    z=oz + (dz - oz) * phase,
                )

                # Keep original yaw to avoid odd rotations during teleport
                new_rot = carla.Rotation(
                    pitch=truck_origin_tf.rotation.pitch,
                    yaw=truck_origin_tf.rotation.yaw,
                    roll=truck_origin_tf.rotation.roll,
                )
                truck.set_transform(carla.Transform(new_loc, new_rot))

            # Tick & record
            world.tick()

            while not image_queue.empty():
                frame = image_queue.get()
                video.write(frame)

    finally:
        try:
            if camera:
                camera.stop()
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

        for actor in [camera, truck, stopped, ego]:
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


if __name__ == "__main__":
    main()
