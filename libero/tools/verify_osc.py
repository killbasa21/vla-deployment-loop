"""Standalone checks on libero/osc_controller.py, before it is trusted in a rollout.

All CPU, no GPU, no Modal, no inference -- so this is free to re-run and should be, after
any change to the controller or to panda_libero_osc.xml.

Checks, in order of what they would catch:
  1. VELOCITY CONVENTION. `compute_torque` derives site velocity as J @ qvel; robosuite
     reads it from the sim. mj_objectVelocity returns (rot, lin) -- the OPPOSITE order to
     the (lin, rot) used here. A swap there is invisible while the arm translates and
     only shows up as instability once it rotates, so it is checked explicitly.
  2. STATION KEEPING / DROOP. The headline claim for this port: a torque controller with
     gravity compensation has no standing sag, unlike the position servo whose droop is
     the subject of the whole `docs/SERVO_DROOP.md`. Measured against that file's numbers.
  3. STEP RESPONSE. How much of a commanded delta is realised in one 20 Hz tick. README
     sec.3 measures the position servo at ~33% (stock gains). PROGRESS sec.12 predicts OSC
     also under-travels, since kp=150 critically damped cannot traverse a full delta in
     50 ms either -- so a number near 100% would mean the port is WRONG, not good.
  4. CONTACT COMPLIANCE. Drive the hand into the table and measure penetration and force.
     PROGRESS sec.19 measured 2.9 mm / ~70 N for the position servo. This is the property
     `--min-clearance` was invented to fake.

Usage:  MUJOCO_GL=egl uv run python libero/tools/verify_osc.py
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from libero.osc_controller import OSCController  # noqa: E402

SCENE = "mujoco_menagerie/franka_emika_panda/scene_libero_osc.xml"
LIBERO_INIT_QPOS = np.array(
    [0.0, -0.16103739, 0.0, -2.44459747, 0.0, 2.2267522, 0.78539816])
TABLE_TOP_Z = -0.012
DECIMATION = 25          # 20 Hz at a 0.002 s timestep


def build():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home != -1:
        mujoco.mj_resetDataKeyframe(model, data, home)
    data.qpos[:7] = LIBERO_INIT_QPOS
    data.ctrl[:] = 0.0
    data.ctrl[7] = 255.0                      # gripper open
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
    osc = OSCController(model, site_id)
    osc.reset(data)
    return model, data, osc, site_id


def settle(model, data, osc, steps=200):
    """Hold station under OSC while contacts settle. With torque actuators ctrl=0 would
    let the arm fall, so 'do nothing' has to be actively commanded."""
    for _ in range(steps):
        data.ctrl[0:7] = osc.compute_torque(model, data)
        mujoco.mj_step(model, data)


def tick(model, data, osc, dpos, drot, n=DECIMATION):
    osc.set_goal(data, dpos, drot)
    sat = 0
    for _ in range(n):
        data.ctrl[0:7] = osc.compute_torque(model, data)
        sat += osc.last_saturated
        mujoco.mj_step(model, data)
    return sat


def check_velocity_convention():
    print("\n[1] SITE VELOCITY CONVENTION  (J @ qvel  vs  mj_objectVelocity)")
    model, data, osc, site_id = build()
    # Give the arm a nonzero, non-symmetric velocity so both linear and angular parts
    # are populated -- at qvel = 0 every convention agrees and the check proves nothing.
    data.qvel[osc.dof_idx] = np.array([0.3, -0.2, 0.15, 0.4, -0.1, 0.25, -0.35])
    mujoco.mj_forward(model, data)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    lin_j = jacp[:, osc.dof_idx] @ data.qvel[osc.dof_idx]
    ang_j = jacr[:, osc.dof_idx] @ data.qvel[osc.dof_idx]

    res = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, res, 0)
    ang_mj, lin_mj = res[0:3], res[3:6]      # mj_objectVelocity is (rot, lin)

    print(f"  linear   J@qvel {np.array2string(lin_j, precision=6)}")
    print(f"           mujoco {np.array2string(lin_mj, precision=6)}   "
          f"max|diff| {np.max(np.abs(lin_j - lin_mj)):.2e}")
    print(f"  angular  J@qvel {np.array2string(ang_j, precision=6)}")
    print(f"           mujoco {np.array2string(ang_mj, precision=6)}   "
          f"max|diff| {np.max(np.abs(ang_j - ang_mj)):.2e}")
    ok = (np.max(np.abs(lin_j - lin_mj)) < 1e-9) and (np.max(np.abs(ang_j - ang_mj)) < 1e-9)
    # Show what the swapped reading would have looked like, so the check is legible.
    print(f"  if (lin,rot) had been assumed: linear error would be "
          f"{np.max(np.abs(lin_j - ang_mj)):.4f} (i.e. the bug is NOT subtle once seen)")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_station_keeping():
    print("\n[2] STATION KEEPING / DROOP  (the headline claim)")
    model, data, osc, site_id = build()
    mujoco.mj_forward(model, data)
    fk_pos = data.site_xpos[site_id].copy()   # pure kinematics, dynamics off
    settle(model, data, osc, steps=1000)
    settled = data.site_xpos[site_id].copy()
    sag = settled - fk_pos
    print(f"  grip_site, pure FK        {np.array2string(fk_pos, precision=5)}")
    print(f"  grip_site, settled (OSC)  {np.array2string(settled, precision=5)}")
    print(f"  sag                       {np.array2string(sag * 1000, precision=3)} mm "
          f"(norm {np.linalg.norm(sag) * 1000:.3f} mm)")
    print(f"  height above table        {settled[2] - TABLE_TOP_Z:.4f} m "
          f"(LIBERO measured 0.2733)")
    print("  docs/SERVO_DROOP.md sec.1: position servo sag was 4.84 mm stock, 2.44 mm stiffened")
    ok = np.linalg.norm(sag) < 5e-4
    print(f"  -> {'PASS' if ok else 'FAIL'} (want < 0.5 mm)")
    return ok


def check_step_response():
    print("\n[3] STEP RESPONSE  (realised / commanded per 20 Hz tick)")
    model, data, osc, site_id = build()
    settle(model, data, osc)
    for label, action_scale in (("full -z (a=-1.0)", -1.0), ("half -z (a=-0.5)", -0.5)):
        m2, d2, o2, s2 = build()
        settle(m2, d2, o2)
        start = d2.site_xpos[s2].copy()
        dpos = np.array([0.0, 0.0, action_scale * 0.05])
        tick(m2, d2, o2, dpos, np.zeros(3))
        moved = d2.site_xpos[s2] - start
        realised = moved[2] / dpos[2]
        print(f"  {label}: commanded {dpos[2] * 1000:+.1f} mm, "
              f"realised {moved[2] * 1000:+.2f} mm  ->  {realised * 100:.1f}%")
    # ANALYTIC CHECK, and this is the real test in this function -- "does it move a
    # plausible amount" is not falsifiable, but this is. OSC with uncoupled lambda makes
    # each Cartesian axis a unit-mass second-order system: wn = sqrt(kp) = sqrt(150) =
    # 12.247 rad/s, critically damped (damping_ratio = 1). The critically-damped step
    # response is  x(t)/x_goal = 1 - (1 + wn*t) * exp(-wn*t).
    wn = np.sqrt(150.0)
    t = DECIMATION * 0.002
    predicted = 1.0 - (1.0 + wn * t) * np.exp(-wn * t)
    print(f"  ANALYTIC: wn=sqrt(kp)={wn:.3f} rad/s, critically damped, t={t:.3f} s")
    print(f"            1-(1+wn*t)*exp(-wn*t) = {predicted * 100:.1f}%  <- expected")
    print("  docs/SERVO_DROOP.md sec.3: position servo realises ~33% (stock) / ~72% (stiffened),")
    print("  so OSC moves LESS per tick than Route A did. That is not a regression: the")
    print("  policy was TRAINED through this exact response, so its commanded magnitudes")
    print("  already account for it. A value near 100% would mean the port is WRONG.")
    ok = abs(realised - predicted) < 0.02
    print(f"  -> {'PASS' if ok else 'FAIL'} (measured {realised * 100:.1f}% vs "
          f"predicted {predicted * 100:.1f}%)")
    return ok


def check_contact_compliance():
    print("\n[4] CONTACT COMPLIANCE  (drive into the table)")
    model, data, osc, site_id = build()
    settle(model, data, osc)
    table_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_geom")

    # Enough ticks to actually ARRIVE. At ~12% realised per tick (check 3) a full-scale
    # descent covers ~6 mm/tick, and the eef starts 273 mm above the table -- an earlier
    # version of this check ran 12 ticks, stopped 126 mm short, and reported "0 N, no
    # penetration" as though that were compliance. It was just a test that never touched
    # anything. Same class of error as PROGRESS sec.7 (counting the green bin as the ball):
    # the probe was wrong, not the system.
    worst_dist, max_fn, contact_steps = 0.0, 0.0, 0
    for _ in range(120):
        tick(model, data, osc, np.array([0.0, 0.0, -0.05]), np.zeros(3))
        touched = False
        for i in range(data.ncon):
            c = data.contact[i]
            if table_gid not in (c.geom1, c.geom2):
                continue
            touched = True
            worst_dist = min(worst_dist, c.dist)
            f = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, f)
            max_fn = max(max_fn, abs(f[0]))
        contact_steps += int(touched)

    eef_z = data.site_xpos[site_id][2]
    print(f"  after 120 ticks of full-scale descent "
          f"({contact_steps} ticks in table contact):")
    print(f"    grip_site z          {eef_z:+.4f}  (table top {TABLE_TOP_Z:+.4f}, "
          f"{(eef_z - TABLE_TOP_Z) * 1000:+.1f} mm above)")
    print(f"    worst contact.dist   {worst_dist * 1000:+.2f} mm  (negative = penetration)")
    print(f"    max normal force     {max_fn:.1f} N")
    print("  PROGRESS sec.19, position servo: -2.9 mm penetration at ~70 N")
    print("  NOTE --min-clearance is OFF here; this is the unclamped behaviour, which is")
    print("  the whole point -- the clamp existed only to fake what this does natively.")
    ok = contact_steps > 0 and worst_dist > -0.0029
    print(f"  -> {'PASS' if ok else 'FAIL'} (want contact to actually happen, and less "
          f"penetration than the position servo's -2.9 mm)")
    return ok


def main():
    results = [
        ("velocity convention", check_velocity_convention()),
        ("station keeping", check_station_keeping()),
        ("step response", check_step_response()),
        ("contact compliance", check_contact_compliance()),
    ]
    print("\n" + "=" * 64)
    for name, ok in results:
        print(f"  {name:24s} {'PASS' if ok else 'FAIL'}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
