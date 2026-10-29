"""Reference implementations of the ELEC3901 lab task functions (generated from tasks.py).
Each lab imports the functions of PREVIOUS labs from here so nobody is blocked by last week; your own definition in
the current notebook takes precedence. Do not read ahead: the point of the tasks is to write them yourself."""
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as _Rot
NOM = {'d1': 0.0975, 'l2z': 0.11315, 'l2x': 0.0415, 'l3': 0.162, 'l4': 0.123}
Q_LIM = np.array([[-2.618, 2.618], [-1.868, 1.658], [-1.571, 1.676], [-1.745, 1.745]])
def small_T(v):
    T = np.eye(4); T[:3, :3] = _Rot.from_rotvec(v[:3]).as_matrix(); T[:3, 3] = v[3:]; return T

# ---- Lab 1 Task 1: Log a joint step (open task)
def step_log(arm, joint, targets_deg, samples=50, period=0.02):
    """Command joint `joint` through targets_deg; return (t, q) arrays of the measured joint angle."""
    q_cmd = arm.read_q().copy(); t0 = time.time(); log_t, log_q = [], []
    for target in np.radians(targets_deg):
        q_cmd[joint] = target; arm.move_q(q_cmd)
        for _ in range(samples):
            log_t.append(time.time() - t0); log_q.append(arm.read_q()[joint]); time.sleep(period)
    return np.array(log_t), np.array(log_q)

# ---- Lab 1 Task 2: 2D pose from scratch
def rot2(theta):
    """2x2 rotation matrix for angle theta (rad)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def se2(x, y, theta):
    """3x3 homogeneous transform: translate (x, y) then rotate theta. Returns a NumPy array."""
    T = np.eye(3); T[:2, :2] = rot2(theta); T[:2, 2] = [x, y]
    return T

def transform_point(T, p):
    """Apply a 3x3 transform T to a 2D point p (length-2 array). Returns a length-2 array."""
    return (T @ np.append(np.asarray(p, float), 1.0))[:2]

# ---- Lab 1 Task 3: Combining transforms
def transl2(x, y):
    """3x3 homogeneous transform: a pure translation by (x, y)."""
    return np.array([[1.0, 0, x], [0, 1.0, y], [0, 0, 1.0]])

def trot2(theta):
    """3x3 homogeneous transform: a pure rotation by theta (rad) about the origin."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0, 0, 1.0]])

def rot_about(theta, c):
    """3x3 homogeneous transform: rotation by theta (rad) about the point c = (cx, cy)."""
    return transl2(c[0], c[1]) @ trot2(theta) @ transl2(-c[0], -c[1])

# ---- Lab 2 Task 1: Rotations in 3D and roll-pitch-yaw angles
def rotx(a):
    """3x3 rotation about x by a (rad)."""
    c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def roty(a):
    """3x3 rotation about y by a (rad)."""
    c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotz(a):
    """3x3 rotation about z by a (rad). GIVEN - the worked example: the columns are where x and y go, z stays put."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0],      # column 0: x turns to (cos a, sin a, 0)
                     [s, c, 0],       # column 1: y turns to (-sin a, cos a, 0)
                     [0, 0, 1]])      # column 2: z is the axis - it does not move

def rpy_to_R(roll, pitch, yaw):
    """zyx roll-pitch-yaw (rad) -> 3x3 rotation matrix."""
    return rotz(yaw) @ roty(pitch) @ rotx(roll)

def R_to_rpy(R):
    """3x3 rotation matrix -> [roll, pitch, yaw] (rad), zyx convention. Assume |pitch| < 90 deg."""
    pitch = np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))
    roll = np.arctan2(R[2, 1], R[2, 2]); yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])

# ---- Lab 2 Task 2: Pose in 3D: build, invert and apply a transform
def make_T(R, t):
    """3x3 rotation, 3-vector -> 4x4 homogeneous transform."""
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T

def inv_T(T):
    """Inverse of a 4x4 homogeneous transform, from R^T and -R^T t (no general matrix inverse)."""
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4); Ti[:3, :3] = R.T; Ti[:3, 3] = -R.T @ t
    return Ti

def transform_points(T, P):
    """Map point(s) P ((3,) or N x 3) through the 4x4 transform T; returns the same shape."""
    P = np.asarray(P, float)
    out = (T[:3, :3] @ np.atleast_2d(P).T).T + T[:3, 3]
    return out[0] if P.ndim == 1 else out

# ---- Lab 2 Task 3: Chaining a camera measurement to the base frame - and measuring the camera mount
def chain_T_BM(T_BE, T_EC, T_CM):
    """All 4x4 NumPy arrays. Returns T_BM."""
    return T_BE @ T_EC @ T_CM

def mount_from_views(T_BEs, T_CMs, T_BM=np.eye(4)):
    """Lists of 4x4 arrays (one pair per view) -> 4x4 camera mount T_EC averaged over the views."""
    Ts = [np.linalg.inv(T_BE) @ T_BM @ np.linalg.inv(T_CM) for T_BE, T_CM in zip(T_BEs, T_CMs)]
    U, _, Vt = np.linalg.svd(np.mean([T[:3, :3] for T in Ts], axis=0))
    T = np.eye(4); T[:3, :3] = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt; T[:3, 3] = np.mean([T[:3, 3] for T in Ts], axis=0)
    return T

# ---- Lab 3 Task 1: Forward kinematics of the OMX in NumPy
def roty(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def rotz(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
def omx_fk(q, p=NOM):
    """Joint angles q (4,) rad, link parameters p -> 4x4 end-effector pose in the base frame."""
    R1 = rotz(q[0]); R2 = R1 @ roty(q[1]); R3 = R2 @ roty(q[2]); R4 = R3 @ roty(q[3])
    t = np.array([0, 0, p['d1']]) + R2 @ [p['l2x'], 0, p['l2z']] + R3 @ [p['l3'], 0, 0] + R4 @ [p['l4'], 0, 0]
    T = np.eye(4); T[:3, :3] = R4; T[:3, 3] = t; return T

# ---- Lab 3 Task 2: Denavit-Hartenberg table and its forward kinematics
def dh_table(p=NOM):
    """Standard DH rows [theta_offset, d, a, alpha] for joints 1..4 (4x4 array)."""
    a2 = np.hypot(p['l2x'], p['l2z']); beta = np.arctan2(p['l2x'], p['l2z'])
    return np.array([[0.0, p['d1'], 0.0, -np.pi / 2],
                     [beta - np.pi / 2, 0.0, a2, 0.0],
                     [np.pi / 2 - beta, 0.0, p['l3'], 0.0],
                     [0.0, 0.0, p['l4'], 0.0]])

def fk_dh(q, dh):
    """Standard DH forward kinematics -> 4x4 pose."""
    T = np.eye(4)
    for qi, (off, d, a, al) in zip(q, dh):
        th = qi + off; ct, st, ca, sa = np.cos(th), np.sin(th), np.cos(al), np.sin(al)
        A = np.array([[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st], [0, sa, ca, d], [0, 0, 0, 1]])
        T = T @ A
    return T

# ---- Lab 3 Task 3: Validation statistics and the error field
def error_stats(pred, meas):
    """N x 3 arrays (m) -> [mean_mm, max_mm, bias_x_mm, bias_y_mm]."""
    d = np.asarray(meas) - np.asarray(pred); n = 1e3 * np.linalg.norm(d, axis=1)
    return np.array([n.mean(), n.max(), 1e3 * d[:, 0].mean(), 1e3 * d[:, 1].mean()])

def plot_error_field(pred, meas, scale=10):
    """Top view: arrows from predicted to measured, exaggerated by `scale`."""
    pred, meas = np.asarray(pred), np.asarray(meas); d = meas - pred
    plt.figure(figsize=(7, 5)); plt.quiver(pred[:, 0], pred[:, 1], d[:, 0] * scale, d[:, 1] * scale, angles='xy', scale_units='xy', scale=1, color='r')
    plt.plot(pred[:, 0], pred[:, 1], 'k.'); plt.axis('equal'); plt.grid(); plt.xlabel('x (m)'); plt.ylabel('y (m)')
    plt.title(f'error field (x{scale}); mean {error_stats(pred, meas)[0]:.1f} mm'); plt.show()

# ---- Lab 4 Task 1: Planar two-link inverse kinematics
def planar_2r(rw, zw, a2, l3, elbow='up'):
    """Planar 2R IK -> [phi2, phi3] (rad) or None if out of reach."""
    # the elbow, equation (3): the law of cosines
    c3 = (rw ** 2 + zw ** 2 - a2 ** 2 - l3 ** 2) / (2 * a2 * l3)
    if abs(c3) > 1:
        return None                      # out of reach: no triangle
    # equation (4): the sign of the sine is the elbow choice
    s3 = np.sqrt(1 - c3 ** 2)
    if elbow == 'up':
        s3 = -s3                         # elbow up: the relative angle is negative
    phi3 = np.arctan2(s3, c3)
    # the shoulder, equation (5): towards the wrist point, less the angle the bent elbow adds
    phi2 = np.arctan2(zw, rw) - np.arctan2(l3 * s3, a2 + l3 * c3)
    return np.array([phi2, phi3])

# ---- Lab 4 Task 2: Full inverse kinematics of the OMX
def omx_ik(p, pitch, elbow='up', par=NOM):
    """Base-frame position p (3,), tool pitch below horizontal (rad) -> q (4,) or None. elbow: 'up', 'down' or 'any'."""
    if elbow == 'any':                   # try elbow up first, then elbow down
        q = omx_ik(p, pitch, 'up', par)
        if q is None:
            q = omx_ik(p, pitch, 'down', par)
        return q
    x = p[0]
    y = p[1]
    z = p[2]
    # step 1, equation (6): joint 1 turns the arm's plane onto the target
    q1 = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)
    # step 2, equation (7): the wrist point, l4 back along the tool, measured from the shoulder
    rw = r - par['l4'] * np.cos(pitch)
    zw = z - par['d1'] + par['l4'] * np.sin(pitch)
    # step 3: the two-link problem (your Task 1)
    a2 = np.sqrt(par['l2x'] ** 2 + par['l2z'] ** 2)
    beta = np.arctan2(par['l2x'], par['l2z'])
    sol = planar_2r(rw, zw, a2, par['l3'], elbow)
    if sol is None:
        return None                      # out of reach
    phi2 = sol[0]
    phi3 = sol[1]
    # step 4, equation (8): the OMX's own angles (zero with link 2 at beta, positive = nose down)
    q2 = (np.pi / 2 - beta) - phi2
    q3 = (beta - np.pi / 2) - phi3
    q4 = pitch - q2 - q3
    q = np.array([q1, q2, q3, q4])
    # an answer outside the joint limits is no answer
    if np.any(q < Q_LIM[:, 0]) or np.any(q > Q_LIM[:, 1]):
        return None
    return q

# ---- Lab 4 Task 3: Placing the camera with IK
def camera_ik(p_C, view_pitch, T_EC, elbow='up', par=NOM):
    """Lens position p_C (3,) in {B}, optical axis view_pitch (rad) below horizontal, mount T_EC -> q (4,) or None."""
    if hasattr(T_EC, 'A'):
        T = T_EC.A                       # an SE3: take its 4 x 4 matrix
    else:
        T = np.array(T_EC)
    # equation (13): the tool pitch is the view pitch less the mount's tilt
    alpha = np.arctan2(-T[2, 2], T[0, 2])
    pitch = view_pitch - alpha
    # equation (14): step back from the lens to the tool point, in the arm's plane
    ox = T[0, 3]
    oz = T[2, 3]
    x = p_C[0]
    y = p_C[1]
    z_C = p_C[2]
    q1 = np.arctan2(y, x)
    r_C = np.sqrt(x ** 2 + y ** 2)
    r = r_C - ox * np.cos(pitch) - oz * np.sin(pitch)
    z = z_C + ox * np.sin(pitch) - oz * np.cos(pitch)
    if r <= 0:
        return None                      # the tool point would be behind the joint-1 axis
    p_tool = [r * np.cos(q1), r * np.sin(q1), z]
    return omx_ik(p_tool, pitch, elbow, par)   # your Task 2
