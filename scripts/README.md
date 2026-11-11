# What is in `scripts/` - a guide to the code the labs use

The notebooks use **two kinds of code**:

1. **Peter Corke's toolboxes** - `spatialmath` and `roboticstoolbox`. These are the toolboxes of the course textbook,
   *Robotics, Vision and Control* (RVC3, Python edition). They are general: they know nothing about our arm.
2. **The course's own modules** - the `omx_*.py` files in this folder. They describe *our* arm (the OpenMANIPULATOR-X,
   "OMX"), drive it, simulate it and read its wrist camera. They are written using the toolboxes.

There is no separate "OMX toolbox" to install or look up online: the course modules are documented here and in their own
docstrings (in a notebook, `help(omx_ik)` or `omx_ik?`). **Do not edit this folder**: staff replace it every week.

## 1. The textbook toolboxes

Both come with the `elec3901` conda environment (`environment.yml`). The book's code examples use the same functions, so
anything you read in RVC3 runs in the notebooks.

### spatialmath (`spatialmath-python`) - poses and rotations

Rotations and rigid-body transforms as objects you can multiply, invert and plot. Used every week, from Lab 1.

| Class / function | What it is | RVC3 |
|---|---|---|
| `SE2(x, y, theta)` | 2D pose (a 3x3 homogeneous transform) | Ch. 2 |
| `SO3.Rx(a)`, `SO3.Ry`, `SO3.Rz`, `SO3.RPY(...)`, `SO3.AngleAxis(theta, v)` | 3D rotation (a 3x3 rotation matrix) | Ch. 2 |
| `SE3.Trans(x, y, z)`, `SE3.Tx`/`Ty`/`Tz`, `SE3.Rx`/`Ry`/`Rz`, `SE3.Rt(R, t)` | 3D pose (a 4x4 homogeneous transform) | Ch. 2 |
| `UnitQuaternion` | rotation as a unit quaternion | Ch. 2 |
| `T1 * T2`, `T.inv()`, `T * p` | compose, invert, transform a point | Ch. 2 |
| `T.A` (the numpy array), `T.R`, `T.t`, `T.rpy()`, `T.eul()`, `T.angvec()` | take a pose apart | Ch. 2 |
| `T.plot()`, `trplot(T)`, `tranimate(T)` | draw a frame | Ch. 2 |
| `spatialmath.base`: `rotx`, `roty`, `rotz`, `transl`, `tr2rpy`, `skew`, ... | the same ideas on plain numpy arrays | Ch. 2 |

```python
from spatialmath import SE3
T_BE = SE3.Trans(0.2, 0, 0.1) * SE3.Ry(30, unit='deg')   # pose of {E} in {B}: Corke's notation ^B T_E, code T_BE
p_B = T_BE * [0.05, 0, 0]                                 # a point given in {E}, expressed in {B}
T_EB = T_BE.inv()
print(T_BE.t, T_BE.rpy(unit='deg'))
```

Angles are in **radians** unless you pass `unit='deg'`.

- Documentation: <https://spatialmath-python.rai-inst.com/> (class reference: `spatialmath.html`; plain-array functions:
  `func_3d.html`)
- Source code and examples: <https://github.com/bdaiinstitute/spatialmath-python>

### roboticstoolbox (`roboticstoolbox-python`) - robot arms

Robot models built from links, with forward and inverse kinematics, Jacobians, dynamics and trajectories. Used from Lab 3.

| Class / function | What it is | RVC3 |
|---|---|---|
| `ET.tz(d)`, `ET.Rz()`, ... joined with `*` | an elementary transform sequence (ETS) | Ch. 7.1 |
| `rtb.Robot(ets)`; `rtb.DHRobot([rtb.RevoluteDH(...), ...])` | a robot from an ETS; a robot from DH parameters | Ch. 7 |
| `robot.fkine(q)` | forward kinematics: the end-effector pose, an `SE3` | Ch. 7.1 |
| `robot.ikine_LM(T, mask=...)` | numerical inverse kinematics | Ch. 7.3 |
| `robot.jacob0(q)`, `robot.jacobe(q)` | Jacobian in the base frame / end-effector frame | Ch. 8 |
| `robot.manipulability(q)` | how well the arm can move in every direction | Ch. 8 |
| `robot.gravload(q)`, `robot.rne(q, qd, qdd)` | gravity torques; full inverse dynamics | Ch. 9 |
| `rtb.jtraj`, `rtb.ctraj`, `rtb.mstraj`, `rtb.quintic`, `rtb.trapezoidal` | joint-space, Cartesian and multi-segment trajectories | Ch. 3 |

```python
import roboticstoolbox as rtb
from omx_model import omx_robot, HOME_Q
robot = omx_robot()                       # our arm as a roboticstoolbox Robot (4 joints)
T = robot.fkine(HOME_Q)                   # SE3 pose of the fingertips
J = robot.jacob0(HOME_Q)                  # 6 x 4
m = robot.manipulability(HOME_Q, axes='trans')
traj = rtb.jtraj(HOME_Q, [0, 0, 0, 0], 50)    # traj.q, traj.qd, traj.qdd: 50 x 4
```

The OMX has only 4 joints, so it cannot reach an arbitrary 6-DOF pose. Ask for the parts it can do:
`manipulability(q, axes='trans')`, and `ikine_LM(T, mask=[1, 1, 1, 0, 0, 0])` for position only. For the OMX, the analytic
`omx_ik` below is usually the better choice.

- Documentation: <https://petercorke.github.io/robotics-toolbox-python/> (arms: `arm.html`; ETS: `ets.html`; inverse
  kinematics: `IK/ik.html`; trajectories: `arm_trajectory.html`)
- Source code and examples: <https://github.com/petercorke/robotics-toolbox-python>

### How this relates to `rvc3python` and the book's own code

`rvc3python` (<https://github.com/petercorke/RVC3-python>) is the book's companion package. `pip install rvc3python`
installs the toolboxes behind the book: the two above, plus the Machine Vision Toolbox and `bdsim` (block diagrams). It also
adds `rvctool`, an interactive Python prompt with everything already imported, as the book's examples are written. The
repository's `notebooks/` folder has the code of every chapter as Jupyter notebooks.

You do **not** need it for this course: the `elec3901` environment installs `spatialmath` and `roboticstoolbox` directly, and
the camera work uses OpenCV (through `omx_camera.py`) instead of the Machine Vision Toolbox. Install it if you want to run the
book's own examples, preferably in a separate conda environment so the course environment stays as issued.

## 2. The course modules in this folder

| Module | What it does | Main names |
|---|---|---|
| `omx_model.py` | The arm's geometry and kinematics, built on the toolboxes | `omx_robot()`, `omx_ets()`, `omx_dh()`, `omx_ik(p, pitch, elbow)`, `Q_LIM`, `HOME_Q`, `REST_Q`, `T_EC`, `P_TIP_E`, `mount_tilt`, `look_down_pitch` |
| `omx.py` | Drives the **real arm** over its USB servo bus | `OMX()`, `torque`, `read_q`, `move_q`, `stream`, `set_profile`, `set_gripper`, `read_torque`, `home`, `rest`, `close` |
| `omx_mujoco.py` | The **digital twin**: a MuJoCo simulation with the same methods as the real arm and camera | `OMXSim(view=True)`, `SimCamera(arm)` |
| `omx_camera.py` | The **wrist camera** as a measuring instrument: images, ArUco markers on the mat, the mat's pose | `Camera()`, `calibrate_from_mat`, `mat_pose`, `mat_pose_avg`, `tip_in_mat`, `predicted_T_CM`, `pose_error` |
| `omx_autocal.py` | The arm moves the camera over the mat and calibrates it (the notebooks' camera cell) | `auto_calibrate(arm, cam, ARM_ID)` |
| `omx_calib.py` | Kinematic and camera-mount calibration (Lab 9) | `small_T`, ... |
| `omx_ref.py` | **Completed lab functions**: the reference solutions of your Tasks from earlier labs, so a late or wrong answer last week does not block this week. It grows each week | `make_T`, `omx_fk`, `omx_ik`, ... (more each week) |

Also here: `mat_markers.json` (where every mat marker is, in mm), the mat image and PDF, and `assets/omx_f/` (the arm's
meshes for the twin, from ROBOTIS).

### Conventions used by every module

- **Units**: metres and radians. Joint vectors `q` have 4 entries (joints 1-4; the wrist roll, joint 5, is held at 0).
- **Frames**: {B} the arm's base, on the mat at the joint-1 axis, x forward, z up; {E} the end effector at the closed
  fingertips, x along the last link; {C} the camera (OpenCV: z out of the lens, x right, y down the image); {M} the mat.
  `T_BE` is the pose of {E} in {B}, Corke's ^B T_E.
- **Pitch**: positive = tool pointing down (as `SE3.Ry` with a positive angle).
- **`HOME_Q`** is the home pose; `q = 0` is the kinematic zero pose, not home. Always `arm.rest()` before `arm.torque(False)`.

### Real arm or twin

The notebooks' connect cell chooses with `SIMULATE = True / False`. Both give you `arm` and `cam` with the same methods, so the
same code runs on either:

```python
from omx_mujoco import OMXSim, SimCamera     # or: from omx import OMX; from omx_camera import Camera
from omx_model import omx_ik, HOME_Q
arm = OMXSim(view=True)
cam = SimCamera(arm)
arm.torque(True)
arm.home()
q = omx_ik((0.20, 0.0, 0.10), pitch=0.5)    # tip 200 mm forward, 100 mm up, tool pitched 0.5 rad down
arm.move_q(q, wait=True)
print(arm.read_q())
T_CM, n = cam.mat_pose()                     # pose of the mat in the camera frame, and how many markers it used
arm.rest()
arm.close()
```

### Finding out more

- `help(name)` or `name?` in a notebook shows a function's docstring: the arguments, units and what it returns.
- The top of each `.py` file explains what the module is for and how it is meant to be used.
