import os
import math
from queue import Queue

import carla
import cv2
import numpy as np
from dotenv import load_dotenv

from PCLA import PCLA, location_to_waypoint, route_maker

load_dotenv()

HOST = os.getenv("CARLA_HOST", "127.0.0.1")
PORT = int(os.getenv("CARLA_PORT", 2000))

MAP_NAME = "Town03"
TM_PORT = 8000

FIXED_DELTA = 0.05
FPS = int(1.0 / FIXED_DELTA)

VIDEO_PATH = "/home/joshua/CPX/scene.mp4"
IMAGE_W = 1280
IMAGE_H = 720
FOV = 90

DURATION_SECONDS = 15.0

# Ego placement: approaching a junction within this distance range
EGO_TO_JUNCTION_MIN = 35.0
EGO_TO_JUNCTION_MAX = 90.0

# Attacker: waits at side-road stop, then creeps forward across stop line toward ego lane center
TRIGGER_EGO_TO_JUNCTION = 26.0  # meters
ATTACKER_SPEED = 2.0            # m/s
ATTACKER_MAX_TIME_ACTIVE = 8.0  # seconds safety

# Stop line approximation: attacker starts behind this many meters from junction entry
ATTACKER_STOP_BEHIND_JUNCTION = 8.0  # meters

# How close to ego lane center we want the attacker nose to get (approx)
INTRUDE_DISTANCE_TO_EGO_LANE_CENTER = 1.2  # meters

# Chase camera (spectator) parameters
CAM_BACK = 7.5
CAM_UP = 3.2
CAM_PITCH = -10.0


def setup_world(client: carla.Client) -> carla.World:
    world = client.load_world(MAP_NAME)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DELTA
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)

    world.tick()
    return world


def v2(x, y):
    return np.array([float(x), float(y)], dtype=np.float32)


def norm2(v):
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return v, 0.0
    return v / n, n


def angle_deg(a, b):
    a, _ = norm2(a)
    b, _ = norm2(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def advance_to_junction(wp: carla.Waypoint, max_dist=220.0, step=6.0):
    dist = 0.0
    cur = wp
    while dist < max_dist and cur is not None:
        nxts = cur.next(step)
        if not nxts:
            return None, None
        cur = nxts[0]
        dist += step
        if cur.is_junction:
            return cur, dist
    return None, None


def is_roundabout_like(junc: carla.Junction) -> bool:
    pairs = junc.get_waypoints(carla.LaneType.Driving)
    if not pairs:
        return True
    if len(pairs) > 28:
        return True
    yaws = [p[0].transform.rotation.yaw for p in pairs[: min(len(pairs), 20)]]
    if len(yaws) >= 8:
        yaws = np.deg2rad(np.array(yaws, dtype=np.float32))
        spread = float(np.std(np.unwrap(yaws)))
        if spread > 0.9:
            return True
    return False


def find_t_junction_candidate(world: carla.World):
    cmap = world.get_map()
    sps = cmap.get_spawn_points()
    best = None

    for i, sp in enumerate(sps):
        wp0 = cmap.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp0 is None:
            continue

        j_wp, d_to_j = advance_to_junction(wp0, max_dist=220.0, step=7.0)
        if j_wp is None or j_wp.get_junction() is None:
            continue

        if not (EGO_TO_JUNCTION_MIN <= d_to_j <= EGO_TO_JUNCTION_MAX):
            continue

        junc = j_wp.get_junction()
        if is_roundabout_like(junc):
            continue

        pairs = junc.get_waypoints(carla.LaneType.Driving)
        if not pairs:
            continue

        # ego should be able to go straight-ish through
        ego_fwd = wp0.transform.get_forward_vector()
        ego_dir = v2(ego_fwd.x, ego_fwd.y)
        ego_dir, _ = norm2(ego_dir)

        wp_before = wp0
        for _ in range(16):
            nxts = wp_before.next(5.0)
            if not nxts:
                break
            if nxts[0].is_junction:
                break
            wp_before = nxts[0]

        options = wp_before.next(15.0)
        if not options:
            continue

        best_straight = 999.0
        for opt in options:
            f = opt.transform.get_forward_vector()
            best_straight = min(best_straight, angle_deg(ego_dir, v2(f.x, f.y)))
        if best_straight > 20.0:
            continue

        # must have a perpendicular incoming approach (side road)
        perp_found = False
        for w_in, _w_out in pairs:
            f = w_in.transform.get_forward_vector()
            in_dir = v2(f.x, f.y)
            in_dir, _ = norm2(in_dir)
            dp = abs(float(np.dot(in_dir, ego_dir)))
            if dp < 0.35:
                perp_found = True
                break
        if not perp_found:
            continue

        complexity = len(pairs)
        if complexity > 22:
            continue

        score = (complexity, abs(d_to_j - 60.0), best_straight)
        if best is None or score < best[0]:
            best = (score, i, sp, wp0, j_wp, junc)

    if best is None:
        i = min(31, len(world.get_map().get_spawn_points()) - 1)
        sp = world.get_map().get_spawn_points()[i]
        wp0 = world.get_map().get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        j_wp, _ = advance_to_junction(wp0, max_dist=220.0, step=7.0) if wp0 else (None, None)
        junc = j_wp.get_junction() if j_wp else None
        return i, sp, wp0, j_wp, junc

    _, i, sp, wp0, j_wp, junc = best
    return i, sp, wp0, j_wp, junc


def pick_side_incoming_lane(junc: carla.Junction, ego_wp0: carla.Waypoint):
    pairs = junc.get_waypoints(carla.LaneType.Driving)
    if not pairs:
        return None

    ego_fwd = ego_wp0.transform.get_forward_vector()
    ego_dir = v2(ego_fwd.x, ego_fwd.y)
    ego_dir, _ = norm2(ego_dir)
    ego_road_id = ego_wp0.road_id

    best = None
    jc = junc.bounding_box.location

    for w_in, w_out in pairs:
        # side road should not be the same road as ego's approach road
        if w_in.road_id == ego_road_id:
            continue

        f = w_in.transform.get_forward_vector()
        in_dir = v2(f.x, f.y)
        in_dir, _ = norm2(in_dir)

        dp = abs(float(np.dot(in_dir, ego_dir)))
        if dp > 0.45:
            continue

        # closer to junction center is better
        d_center = float(w_in.transform.location.distance(jc))
        # also prefer lanes that actually enter the junction (w_in should be close to boundary)
        score = (dp, d_center)
        if best is None or score < best[0]:
            best = (score, w_in, w_out)

    return None if best is None else best[1]


def ensure_route_file(client: carla.Client, start_loc: carla.Location, end_loc: carla.Location, route_path: str) -> str:
    if os.path.exists(route_path):
        return route_path
    wps = location_to_waypoint(client, start_loc, end_loc)
    route_maker(wps, route_path)
    return route_path


def draw_stop_line(world: carla.World, wp: carla.Waypoint, life: float):
    dbg = world.debug
    tf = wp.transform
    loc = tf.location
    right = tf.get_right_vector()
    lane_w = float(max(3.0, min(5.0, wp.lane_width)))
    a = loc + carla.Location(x=right.x * (lane_w * 0.65), y=right.y * (lane_w * 0.65), z=0.12)
    b = loc - carla.Location(x=right.x * (lane_w * 0.65), y=right.y * (lane_w * 0.65), z=0.12)
    dbg.draw_line(a, b, thickness=0.14, color=carla.Color(255, 255, 255), life_time=life)
    dbg.draw_string(loc + carla.Location(z=0.6), "STOP (approx)", draw_shadow=True,
                    color=carla.Color(255, 255, 255), life_time=life)


def lane_center_from_waypoint(wp: carla.Waypoint) -> carla.Location:
    return wp.transform.location


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(30.0)

    world = setup_world(client)
    cmap = world.get_map()
    blueprints = world.get_blueprint_library()
    original_settings = world.get_settings()

    ego = None
    attacker = None
    camera = None
    video = None
    pcla = None

    image_queue: Queue = Queue()

    try:
        sp_idx, ego_spawn, ego_wp0, ego_j_wp, junc = find_t_junction_candidate(world)
        print(f"[INFO] Ego spawn idx={sp_idx} map={MAP_NAME}")

        ego_bp = blueprints.filter("vehicle.tesla.model3")[0]
        ego = world.try_spawn_actor(ego_bp, ego_spawn)
        if ego is None:
            ego = world.spawn_actor(ego_bp, ego_spawn)
        world.tick()

        # Route for ego: build a forward route along its lane to go through junction
        wp_cursor = cmap.get_waypoint(ego_spawn.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        end_wp = wp_cursor
        for _ in range(34):
            nxts = end_wp.next(12.0) if end_wp else []
            if not nxts:
                break
            if len(nxts) == 1:
                end_wp = nxts[0]
            else:
                cur_fwd = end_wp.transform.get_forward_vector()
                cur_dir = v2(cur_fwd.x, cur_fwd.y)
                cur_dir, _ = norm2(cur_dir)
                best_opt = nxts[0]
                best_ang = 999.0
                for opt in nxts:
                    f = opt.transform.get_forward_vector()
                    ang = angle_deg(cur_dir, v2(f.x, f.y))
                    if ang < best_ang:
                        best_ang = ang
                        best_opt = opt
                end_wp = best_opt

        route_path = "./_pcla_route.xml"
        ensure_route_file(client, ego_spawn.location, end_wp.transform.location, route_path)

        pcla = PCLA("carl_carlv11", ego, route_path, client)
        print("[INFO] PCLA running")

        spectator = world.get_spectator()

        cam_bp = blueprints.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMAGE_W))
        cam_bp.set_attribute("image_size_y", str(IMAGE_H))
        cam_bp.set_attribute("fov", str(FOV))
        camera = world.spawn_actor(cam_bp, carla.Transform(), attach_to=spectator)

        def camera_callback(image: carla.Image):
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
            image_queue.put(arr[:, :, :3])

        camera.listen(camera_callback)

        video = cv2.VideoWriter(
            VIDEO_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (IMAGE_W, IMAGE_H),
        )

        # ---- Attacker placement: spawn on side-road incoming lane and stop behind junction ----
        attacker_bp = blueprints.filter("vehicle.audi.*")[0]

        attacker_in_wp = None
        attacker_stop_wp = None
        attacker_target_loc = None

        if ego_wp0 is not None and ego_j_wp is not None and junc is not None:
            attacker_in_wp = pick_side_incoming_lane(junc, ego_wp0)

        if attacker_in_wp is not None:
            # stop a bit before junction boundary on the side road (approx stop line)
            prevs = attacker_in_wp.previous(ATTACKER_STOP_BEHIND_JUNCTION)
            attacker_stop_wp = prevs[0] if prevs else attacker_in_wp

            spawn_tf = attacker_stop_wp.transform
            spawn_tf.location.z += 0.35

            attacker = world.try_spawn_actor(attacker_bp, spawn_tf)
            if attacker is None:
                # try a bit further back
                prevs2 = attacker_stop_wp.previous(4.0)
                if prevs2:
                    spawn_tf2 = prevs2[0].transform
                    spawn_tf2.location.z += 0.35
                    attacker = world.try_spawn_actor(attacker_bp, spawn_tf2)
        else:
            # hard fallback: spawn near ego's upcoming junction area on the right, facing left
            ego_tf0 = ego.get_transform()
            fwd0 = ego_tf0.get_forward_vector()
            right0 = ego_tf0.get_right_vector()
            spawn_loc = ego_tf0.location + fwd0 * 55.0 + right0 * 14.0
            spawn_loc.z += 0.35
            spawn_rot = carla.Rotation(yaw=ego_tf0.rotation.yaw - 90.0)
            attacker = world.try_spawn_actor(attacker_bp, carla.Transform(spawn_loc, spawn_rot))

        world.tick()

        if attacker is None:
            print("[WARN] Attacker could not be spawned; scenario will still run but without interaction.")
        else:
            attacker.set_autopilot(False)
            attacker.set_simulate_physics(True)
            attacker.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            print("[INFO] Attacker spawned")

        if attacker_stop_wp is not None:
            draw_stop_line(world, attacker_stop_wp, life=DURATION_SECONDS)

        # Intrusion target: ego lane center near the junction entry (slightly before/at junction)
        if ego_j_wp is not None and attacker is not None:
            ego_lane_center = lane_center_from_waypoint(ego_j_wp)
            attacker_target_loc = carla.Location(ego_lane_center.x, ego_lane_center.y, ego_lane_center.z)

        sim_start = world.get_snapshot().timestamp.elapsed_seconds
        attacker_started = False
        attacker_start_time = None
        attacker_done = False

        while True:
            snap = world.get_snapshot()
            now = snap.timestamp.elapsed_seconds
            if now - sim_start >= DURATION_SECONDS:
                break

            # Ego control from PCLA
            ego_control = pcla.get_action()
            ego.apply_control(ego_control)

            ego_tf = ego.get_transform()
            ego_loc = ego_tf.location
            ego_rot = ego_tf.rotation

            ego_to_j = None
            if ego_j_wp is not None:
                ego_to_j = ego_loc.distance(ego_j_wp.transform.location)

            # Trigger attacker to start rolling when ego approaches
            if attacker is not None and not attacker_started and not attacker_done:
                if (ego_to_j is not None and ego_to_j < TRIGGER_EGO_TO_JUNCTION) or (now - sim_start > 6.0):
                    attacker_started = True
                    attacker_start_time = now
                    attacker.apply_control(carla.VehicleControl(throttle=0.3, brake=0.0))
                    print("[EVENT] Attacker begins rolling past stop line")

            # Attacker creep control: set target velocity towards ego lane center, then stop when close
            if attacker is not None and attacker_started and not attacker_done:
                if attacker_start_time is not None and (now - attacker_start_time) > ATTACKER_MAX_TIME_ACTIVE:
                    attacker_done = True
                    attacker.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                    attacker.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                    print("[EVENT] Attacker stopped (timeout)")
                else:
                    a_loc = attacker.get_location()
                    if attacker_target_loc is not None:
                        vec = v2(attacker_target_loc.x - a_loc.x, attacker_target_loc.y - a_loc.y)
                        vdir, vnorm = norm2(vec)

                        if vnorm <= INTRUDE_DISTANCE_TO_EGO_LANE_CENTER:
                            attacker_done = True
                            attacker.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                            attacker.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                            print("[EVENT] Attacker intruded into/near ego lane and stopped")
                        else:
                            attacker.set_target_velocity(
                                carla.Vector3D(float(vdir[0] * ATTACKER_SPEED), float(vdir[1] * ATTACKER_SPEED), 0.0)
                            )
                    else:
                        tf = attacker.get_transform()
                        f = tf.get_forward_vector()
                        attacker.set_target_velocity(carla.Vector3D(f.x * ATTACKER_SPEED, f.y * ATTACKER_SPEED, 0.0))

            # Chase camera: slightly above and behind ego, aligned with ego yaw, forward-facing
            fwd = ego_tf.get_forward_vector()
            cam_loc = ego_loc - fwd * CAM_BACK + carla.Location(z=CAM_UP)
            cam_rot = carla.Rotation(pitch=CAM_PITCH, yaw=ego_rot.yaw, roll=0.0)
            spectator.set_transform(carla.Transform(cam_loc, cam_rot))

            world.tick()

            while not image_queue.empty():
                video.write(image_queue.get())

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

        for actor in [attacker, ego]:
            try:
                if actor:
                    actor.destroy()
            except Exception:
                pass

        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = 0.0
            settings.no_rendering_mode = original_settings.no_rendering_mode
            world.apply_settings(settings)
        except Exception:
            pass

        try:
            tm = client.get_trafficmanager(TM_PORT)
            tm.set_synchronous_mode(False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
