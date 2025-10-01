import matplotlib.pyplot as plt
import time
import cvxpy
import math
import numpy as np

from angle import angle_mod
import cubic_spline_planner

NX = 4  # x, y, yaw, v
NU = 2  # a, omega (acceleration and angular velocity)
T = 5  # horizon length

# mpc parameters
R = np.diag([0.1, 0.1])  # input cost matrix
Rd = np.diag([0.5, 1.0])  # input difference cost matrix
Q = np.diag([1000.0, 1000.0, 20.0, 1.0])  # x, y, yaw, v
Qf = Q  # state final matrix
GOAL_DIS = 3.0  # goal distance
MAX_TIME = 200.0  # max simulation time

# iterative paramter
MAX_ITER = 5  # Max iteration
DU_TH = 0.01  # iteration finish param

N_IND_SEARCH = 10  # Search index number

DT = 0.05  # [s] time tick

MAX_V = 5.0 # m/s
MAX_OMEGA = np.deg2rad(180) # rad/s
MAX_DOMEGA = 10.0
MAX_ACCEL = 7.0  # m/s^2

TARGET_SPEED = 3.0

FORMATION_OFFSETS = [
    np.array([ 0.25,  0.25]),  # R0 - front-left
    np.array([ 0.25, -0.25]),  # R1 - front-right
    np.array([-0.25,  0.25]),  # R2 - back-left
    np.array([-0.25, -0.25]),  # R3 - back-right
]

show_animation = True

class State:
    """
    Vehicle state class to keep track of vehicles

    """
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=1.0): #start vehicle at 0s for everything
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

def pi_2_pi(angle):
    """
    Normalize angle to [-pi, pi], supports scalar or NumPy array input
    """
    angle = np.asarray(angle)
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_linear_model_matrix(v, yaw):
    """
    Description:
        Generates the linearized discrete-time system matrices A, B, C for the differential drive model at a given velocity and yaw.

    Inputs:
        v   : float - linear velocity of the robot
        yaw : float - current orientation (heading) of the robot in radians

    Outputs:
        A, B, C : np.ndarray - Discrete-time linear system matrices

    Where the function is called:
        - `linear_mpc_control()`

    Why we need the function:
        Required for formulating the dynamics constraints in the MPC optimization problem using a linearized model around the current state.
    """

    A = np.eye(NX)
    A[0, 2] = -v * math.sin(yaw) * DT
    A[1, 2] =  v * math.cos(yaw) * DT
    A[0, 3] = math.cos(yaw) * DT
    A[1, 3] = math.sin(yaw) * DT
    A[3, 3] = 1.0  # velocity dynamics (v[k+1] = v[k] + a*dt)

    B = np.zeros((NX, NU))
    B[2, 1] = DT  # omega
    B[3, 0] = DT  # acceleration

    C = np.zeros(NX)
    return A, B, C

def plot_car(x, y, yaw, size=0.2, color='b'):
    """
    Description:
        Plots the robot as a square indicating its position and orientation.

    Inputs:
        x     : float - x-coordinate of the robot
        y     : float - y-coordinate of the robot
        yaw   : float - heading angle of the robot (radians)
        size  : float - half the length of the robot’s square side (optional)
        color : str   - color for the plot (optional)

    Outputs:
        None (generates plot)

    Where the function is called:
        - `do_simulation()`

    Why we need the function:
        Useful for visualizing the robot’s trajectory and heading during simulation.
    """

    # Define square corners in robot's local frame
    half = size
    corners = np.array([
        [ half,  half],
        [-half,  half],
        [-half, -half],
        [ half, -half],
        [ half,  half]  # Close the square
    ]).T  # Shape: (2, 5)

    # Rotate and translate to global frame
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    global_corners = rot @ corners
    global_corners[0, :] += x
    global_corners[1, :] += y

    plt.plot(global_corners[0, :], global_corners[1, :], color)
    plt.plot(x, y, "o", label="robot center")

def update_state(state, a, omega):
    """
    Description:
        Updates the robot’s state using forward Euler integration based on the given control inputs.

    Inputs:
        state : State - current state of the robot
        v     : float - linear velocity input
        omega : float - angular velocity input

    Outputs:
        state : State - updated robot state

    Where the function is called:
        - `predict_motion()`
        - `do_simulation()`

    Why we need the function:
        Simulates robot motion and is used to predict future states over the MPC horizon.
    """

    state.x += state.v * math.cos(state.yaw) * DT
    state.y += state.v * math.sin(state.yaw) * DT
    state.yaw += omega * DT
    state.yaw = pi_2_pi(state.yaw)
    state.v += a * DT
    return state

def calc_nearest_index(state, cx, cy, cyaw, pind):
    """
    Description:
        Finds the index of the closest point on the reference trajectory to the robot’s current position.

    Inputs:
        state : State - current state of the robot
        cx, cy: list[float] - reference x and y positions
        cyaw  : list[float] - reference yaw angles
        pind  : int - previous closest index (to reduce search window)

    Outputs:
        ind   : int - index of nearest reference point
        mind  : float - signed distance to that point

    Where the function is called:
        - `calc_ref_trajectory()`
        - `do_simulation()`

    Why we need the function:
        Aligns robot with its closest reference point, needed to compute the local trajectory for MPC.
    """

    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind) + pind

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind

def predict_motion(x0, oa, od, xref):
    """
    Description:
        Simulates the robot’s trajectory over the horizon using current control inputs for use as a linearization point.

    Inputs:
        x0   : list[float] - initial robot state [x, y, yaw]
        oa   : list[float] - predicted linear velocities
        od   : list[float] - predicted angular velocities
        xref : np.ndarray  - reference trajectory

    Outputs:
        xbar : np.ndarray - predicted state trajectory

    Where the function is called:
        - `iterative_linear_mpc_control()`

    Why we need the function:
        Provides updated operating points for linearization in the iterative MPC routine.
    """

    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[2], v=x0[3])
    state.yaw = pi_2_pi(state.yaw)

    for (ai, di, i) in zip(oa, od, range(1, T + 1)):
        state = update_state(state, ai, di)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.yaw

    return xbar

def iterative_linear_mpc_control(xref, x0, oa, od, other_predicted_trajs=None, formation_constraints=None):
    """
    Runs an iterative linear MPC routine with fallback in case of solver failure.
    """
    ox, oy, oyaw, ov = None, None, None, None

    if oa is None or od is None:
        oa = [0.0] * T
        od = [0.0] * T

    for _ in range(MAX_ITER):
        xbar = predict_motion(x0, oa, od, xref)
        poa, pod = oa[:], od[:]

        oa_new, od_new, ox, oy, oyaw, ov = linear_mpc_control(
            xref, xbar, x0, other_predicted_trajs, formation_constraints
        )

        if oa_new is None or od_new is None:
            print("Skipping iteration due to MPC failure.")
            return oa, od, ox, oy, oyaw, ov  # return last known good inputs

        du = sum(abs(oa_new[i] - oa[i]) for i in range(T)) + sum(abs(od_new[i] - od[i]) for i in range(T))
        oa = oa_new
        od = od_new

        if du <= DU_TH:
            break
    else:
        print("Iterative is max iter")

    return oa, od, ox, oy, oyaw, ov

def linear_mpc_control(xref, xbar, x0, other_predicted_trajs=None, formation_constraints=None):
    x = cvxpy.Variable((NX, T + 1))  # [x, y, yaw, v]
    u = cvxpy.Variable((NU, T))      # [a, omega]

    cost = 0.0
    constraints = []

    FORMATION_WEIGHT = 2000.0

    apply_speed_matching = False
    if formation_constraints:
        for dx_target, dy_target in [(c[2], c[3]) for c in formation_constraints]:
            if dx_target < 0:
                apply_speed_matching = True
                break

    for t in range(T):
        cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        e_lat = -(x[0, t] - xref[0, t]) * np.sin(xref[2, t]) + (x[1, t] - xref[1, t]) * np.cos(xref[2, t])
        cost += 300.0 * cvxpy.square(e_lat)

        yaw_error = x[2, t] - xref[2, t]
        cost += 1000.0 * cvxpy.square(yaw_error)

        target_speed = xref[3, t]
        apply_speed_matching = False
        if apply_speed_matching:
            for pred_x, pred_y, dx_target, dy_target in formation_constraints:
                if pred_x is None or len(pred_x) <= t:
                    continue
                delta_x = x[0, t] - pred_x[t]
                delta_y = x[1, t] - pred_y[t]
                dist_error = delta_x * dx_target + delta_y * dy_target
                target_speed += 0.1 * (-dist_error)

        v_error = x[3, t] - target_speed
        cost += (10.0 if apply_speed_matching else 5.0) * cvxpy.square(v_error)

        v_guess = xbar[3, t]
        yaw_guess = xbar[2, t]
        A, B, C = get_linear_model_matrix(v_guess, yaw_guess)
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

        if t < (T - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
            constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= MAX_DOMEGA * DT]

        if formation_constraints:
            yaw = xref[2, t]
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)

            for pred_x, pred_y, dx_target, dy_target in formation_constraints:
                if pred_x is None or len(pred_x) <= t:
                    continue

                kappa = 0.0
                if t < len(xref[2, :]) - 1:
                    dyaw = xref[2, t + 1] - xref[2, t]
                    ds = xref[3, t] * DT + 1e-3
                    kappa = dyaw / ds

                scale_x = 0.3
                scale_y = 0.5
                dx_mod = dx_target * (1 + kappa * scale_x)
                dy_mod = dy_target * (1 + kappa * scale_y)

                dx_world = cos_yaw * dx_mod - sin_yaw * dy_mod
                dy_world = sin_yaw * dx_mod + cos_yaw * dy_mod

                rel_x_error = x[0, t] - pred_x[t] - dx_world
                rel_y_error = x[1, t] - pred_y[t] - dy_world
                heading_error = x[2, t] - yaw
                cost += 1.0 * cvxpy.square(heading_error)
                tight_weight_x = 12000.0  # tighter tracking in x (forward direction)
                tight_weight_y = 3000.0  # less strict on y (lateral drift)

                cost += tight_weight_x * cvxpy.square(rel_x_error)
                cost += tight_weight_y * cvxpy.square(rel_y_error)

    for t in range(T - 2):
        jerk = u[0, t + 2] - 2 * u[0, t + 1] + u[0, t]
        cost += 100.0 * cvxpy.square(jerk)

    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

    constraints += [x[:, 0] == x0]
    constraints += [x[3, :] <= MAX_V]
    constraints += [x[3, :] >= -MAX_V]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_OMEGA]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.CLARABEL, verbose=False)

    if prob.status in [cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE]:
        ox = np.array(x.value[0, :]).flatten()
        oy = np.array(x.value[1, :]).flatten()
        oyaw = np.array(x.value[2, :]).flatten()
        ov = np.array(u.value[0, :]).flatten()
        od = np.array(u.value[1, :]).flatten()
    else:
        print("Error: Cannot solve mpc..")
        ox = oy = oyaw = ov = od = None

    return ov, od, ox, oy, oyaw, ov

def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, pind, unlock_index_limit=None):
    xref = np.zeros((NX, T + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)
    if pind >= ind:
        ind = pind

    for i in range(T + 1):
        travel = abs(state.v) * i * DT
        dind = int(round(travel / dl))
        dind = min(dind, 5)  # <-- relaxed to allow more forward points

        index = ind + dind
        if unlock_index_limit is not None:
            index = min(index, unlock_index_limit)

        if index < ncourse:
            xref[0, i] = cx[index]
            xref[1, i] = cy[index]
            xref[2, i] = cyaw[index]
            # fallback speed clamp to avoid zero-speed startup
            xref[3, i] = max(sp[index], 1.0) if index < ncourse - 5 else 0.0
        else:
            xref[0, i] = cx[-1]
            xref[1, i] = cy[-1]
            xref[2, i] = cyaw[-1]
            xref[3, i] = 0.0

    xref[2, :] = pi_2_pi(xref[2, :])
    return xref, ind

def check_goal(state, goal, tind, nind):
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    dist = math.hypot(dx, dy)

    is_goal = dist <= GOAL_DIS
    is_slow = abs(state.v) <= 0.2  # make sure robot is nearly stopped
    index_close = tind >= nind - 5

    return is_goal and is_slow and index_close

def do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state):
    """
    Simulation
    cx: course x position list
    cy: course y position list
    cy: course yaw position list
    ck: course curvature list
    dl: course tick [m]
    """
    goal = [cx[-1], cy[-1]]

    state = initial_state

    # initial yaw compensation
    if state.yaw - cyaw[0] >= math.pi:
        state.yaw -= math.pi * 2.0
    elif state.yaw - cyaw[0] <= -math.pi:
        state.yaw += math.pi * 2.0

    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    d = [0.0]
    a = [0.0]

    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)

    odelta, oa = None, None

    cyaw = smooth_yaw(cyaw)

    while MAX_TIME >= time:
        xref, target_ind = calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, target_ind)

        di, ai = 0.0, 0.0
        if odelta is not None:
            di, ai = odelta[0], oa[0]
            state = update_state(state, ai, di)

        x0 = [state.x, state.y, state.yaw, state.v]  # Reset current state

        oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(xref, x0, oa, odelta)

        time = time + DT

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        d.append(di)
        a.append(ai)

        if check_goal(state, goal, target_ind, len(cx)):
            print("Goal")
            break

        if show_animation:
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            if ox is not None and len(ox) > 0:
                # Align the red dots with the current robot center for visualization
                dx = state.x - ox[0]
                dy = state.y - oy[0]
                ox_aligned = [x + dx for x in ox]
                oy_aligned = [y + dy for y in oy]
                plt.plot(ox_aligned, oy_aligned, "xr", label="MPC (centered)")
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plot_car(state.x, state.y, state.yaw)
            plt.axis("equal")
            plt.grid(True)
            plt.title(f"Time[s]: {round(time, 2)}, speed[km/h]: {round(state.v, 2)}")
            plt.pause(0.0001)

    return t, x, y, yaw, v, d, a

def smooth_yaw(yaw):
    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw

def get_straight_course(dl):
    ax = [0.0, 5.0, 10.0, 20.0]
    ay = [0.0, 0.0, 0.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck

def get_straight_course2(dl):
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    return cx, cy, cyaw, ck

def get_straight_course3(dl):
    ax = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    ay = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=dl)

    cyaw = [i - math.pi for i in cyaw]

    return cx, cy, cyaw, ck

def get_aisle_turn_course(dl):
    # Move down an aisle, then make a sharp right turn
    ax = [0.0, 5.0, 10.0, 15.0, 15.0, 15.0]
    ay = [0.0, 0.0, 0.0, 0.0, 2.0, 5.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=dl)
    return cx, cy, cyaw, ck

def get_zigzag_course(dl):
    ax = [0.0, 5.0, 10.0, 15.0, 20.0]
    ay = [0.0, 2.0, 0.0, 2.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=dl)
    return cx, cy, cyaw, ck

def get_figure_eight_course(dl):
    ax = [0.0, 5.0, 10.0, 15.0, 10.0, 5.0, 0.0]
    ay = [0.0, 5.0, 5.0, 0.0, -5.0, -5.0, 0.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=dl)
    return cx, cy, cyaw, ck

def get_u_turn_course(dl):
    # Robot goes forward, loops around, and comes back
    ax = [0.0, 10.0, 10.0, 0.0]
    ay = [0.0, 0.0, 5.0, 5.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=dl)
    return cx, cy, cyaw, ck

def get_forward_course(dl):
    ax = [0.0, 60.0, 125.0, 50.0, 75.0, 30.0, -10.0]
    ay = [0.0, 0.0, 50.0, 65.0, 30.0, 50.0, -20.0]
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=dl)

    return cx, cy, cyaw, ck

def calc_speed_profile(cx, cy, cyaw, ck, target_speed):
    """
    Generates a speed profile that slows down in sharp turns based on curvature.

    ck : list of curvature values from spline
    target_speed : maximum target speed
    """
    speed_profile = []
    K_CURV = 0.5  # Tuning parameter (increase to go slower in curves)
    EPS = 1e-4    # Prevent divide-by-zero
    MIN_SPEED = 0.5  # m/s (avoid stalling)
    for k in ck:
        if abs(k) < EPS:
            speed = target_speed
        else:
            speed = min(target_speed, K_CURV / (abs(k) + EPS))
        speed_profile.append(max(speed, MIN_SPEED))
    N = 10
    for i in range(1, N + 1):
        index = -i
        if abs(index) <= len(speed_profile):
            taper_ratio = 0.5 * (1 + math.cos(math.pi * i / N))  # from 1 to 0
            speed_profile[index] *= taper_ratio

    return speed_profile

def get_offset_path(cx, cy, cyaw, offset_dist):
    """
    Generates an offset path (left/right) from a centerline.
    Positive offset_dist = left; Negative = right.
    """
    offset_cx, offset_cy = [], []
    for x, y, yaw in zip(cx, cy, cyaw):
        dx = -np.sin(yaw) * offset_dist
        dy =  np.cos(yaw) * offset_dist
        offset_cx.append(x + dx)
        offset_cy.append(y + dy)
    return offset_cx, offset_cy

def run_formation_simulation(cx, cy, cyaw, ck, sp, dl, robot_states, formation_offsets):
    num_robots = len(robot_states)
    oa_list = [[0.0] * T for _ in range(num_robots)]
    od_list = [[0.0] * T for _ in range(num_robots)]
    pred_trajectories = [None] * num_robots
    target_inds = [0] * num_robots
    xrefs = [None] * num_robots

    time = 0.0
    cyaw = smooth_yaw(cyaw)
    offset_distance = 0.25

    while time <= MAX_TIME:
        next_states = [None] * num_robots  # buffer for next step states

        for i in range(num_robots):
            state = robot_states[i]
            offset = formation_offsets[i]
            dx, dy = offset[0], offset[1]

            cx_i, cy_i = [], []
            for x, y, yaw in zip(cx, cy, cyaw):
                x_offset = math.cos(yaw) * dx - math.sin(yaw) * dy
                y_offset = math.sin(yaw) * dx + math.cos(yaw) * dy
                cx_i.append(x + x_offset)
                cy_i.append(y + y_offset)

            offset_mag = np.linalg.norm(offset)
            ck_i = [k / (1 - k * offset_mag) if abs(1 - k * offset_mag) > 1e-3 else k for k in ck]
            sp_i = calc_speed_profile(cx_i, cy_i, cyaw, ck_i, TARGET_SPEED)

            unlock_limit = None
            if i in [2, 3]:
                leader_idx = 0 if i == 2 else 1
                unlock_limit = max(0, target_inds[leader_idx] - 4)
                kappa = ck[target_inds[leader_idx]] if target_inds[leader_idx] < len(ck) else 0.0
                shift = int(round(offset_mag * kappa * 4))
                unlock_limit = min(len(cx) - 1, unlock_limit + shift)

            xref, target_inds[i] = calc_ref_trajectory(
                state, cx_i, cy_i, cyaw, ck_i, sp_i, dl, target_inds[i],
                unlock_index_limit=unlock_limit
            )

            if i in [0, 1]:
                xref[3, :] = np.maximum(xref[3, :], 2.5)  # ensure minimum speed

            xrefs[i] = xref
            x0 = [state.x, state.y, state.yaw, state.v]

            formation_constraint = []

            # Add symmetric lateral constraint between front robots
            if i == 0 and pred_trajectories[1] is not None:
                dx_target = formation_offsets[0][0] - formation_offsets[1][0]
                dy_target = formation_offsets[0][1] - formation_offsets[1][1]
                formation_constraint.append((pred_trajectories[1][0], pred_trajectories[1][1], dx_target, dy_target))
            elif i == 1 and pred_trajectories[0] is not None:
                dx_target = formation_offsets[1][0] - formation_offsets[0][0]
                dy_target = formation_offsets[1][1] - formation_offsets[0][1]
                formation_constraint.append((pred_trajectories[0][0], pred_trajectories[0][1], dx_target, dy_target))

            # Add rear-following logic (same as before)
            if i in [2, 3]:
                leader_idx = 0 if i == 2 else 1
                pred_leader = pred_trajectories[leader_idx]
                if pred_leader is not None:
                    dx_target = formation_offsets[i][0] - formation_offsets[leader_idx][0]
                    dy_target = formation_offsets[i][1] - formation_offsets[leader_idx][1]
                    formation_constraint.append((pred_leader[0], pred_leader[1], dx_target, dy_target))

            oa, od, ox, oy, oyaw, ov = iterative_linear_mpc_control(
                xrefs[i], x0, oa_list[i], od_list[i],
                other_predicted_trajs=pred_trajectories,
                formation_constraints=formation_constraint
            )

            if oa is None or od is None:
                print(f"[{round(time, 2)}s] Robot {i} failed to solve MPC.")
                continue

            oa_list[i] = oa
            od_list[i] = od
            pred_trajectories[i] = (ox, oy)
            next_states[i] = update_state(state, oa[0], od[0])  # defer state update

        # Commit all next states
        for i in range(num_robots):
            robot_states[i] = next_states[i]

        # Print distance between R0 and R1 for debug
        dx = robot_states[0].x - robot_states[1].x
        dy = robot_states[0].y - robot_states[1].y
        dist_r0_r1 = math.hypot(dx, dy)
        print(f"[{round(time, 2)}s] Distance between R0 and R1: {dist_r0_r1:.3f} m")

        time += DT

        if show_animation:
            plt.cla()
            for pred in pred_trajectories:
                if pred and pred[0] is not None:
                    plt.plot(pred[0], pred[1], "xr", alpha=0.3)
            for i, state in enumerate(robot_states):
                color = 'r' if i == 0 else 'b'
                plot_car(state.x, state.y, state.yaw, color=color)
            for i in range(num_robots):
                for j in range(i + 1, num_robots):
                    xi, yi = robot_states[i].x, robot_states[i].y
                    xj, yj = robot_states[j].x, robot_states[j].y
                    plt.plot([xi, xj], [yi, yj], "--k", linewidth=0.5)
            cx_left, cy_left = get_offset_path(cx, cy, cyaw, -offset_distance)
            cx_right, cy_right = get_offset_path(cx, cy, cyaw, offset_distance)
            plt.plot(cx_left, cy_left, "--b", linewidth=1.0, label="Left offset path")
            plt.plot(cx_right, cy_right, "--g", linewidth=1.0, label="Right offset path")
            plt.plot(cx, cy, "-r", label="Centerline")
            plt.axis("equal")
            plt.grid(True)
            plt.title(f"Time[s]: {round(time, 2)}")
            plt.pause(0.001)

def main():
    print(__file__ + " start!!")
    start = time.time()

    dl = 0.1  # course tick
    #Use these two courses for testing
    #cx, cy, cyaw, ck = get_straight_course(dl)
    cx, cy, cyaw, ck = get_zigzag_course(dl)

    #Not these
    #cx, cy, cyaw, ck = get_straight_course2(dl)
    #cx, cy, cyaw, ck = get_straight_course3(dl)
    #cx, cy, cyaw, ck = get_forward_course(dl)
    #cx, cy, cyaw, ck = get_aisle_turn_course(dl)
    #cx, cy, cyaw, ck = get_figure_eight_course(dl)
    #cx, cy, cyaw, ck = get_u_turn_course(dl)

    cyaw = np.array(cyaw)  # convert from list to NumPy array
    cyaw = smooth_yaw(cyaw)
    cyaw = pi_2_pi(cyaw)

    sp = calc_speed_profile(cx, cy, cyaw, ck, TARGET_SPEED)

    robot_states = []
    yaw0 = cyaw[0]
    for offset in FORMATION_OFFSETS:
        dx = math.cos(yaw0) * offset[0] - math.sin(yaw0) * offset[1]
        dy = math.sin(yaw0) * offset[0] + math.cos(yaw0) * offset[1]
        init_x = cx[0] + dx
        init_y = cy[0] + dy
        robot_states.append(State(x=init_x, y=init_y, yaw=yaw0, v=0.0))


    run_formation_simulation(cx, cy, cyaw, ck, sp, dl, robot_states, FORMATION_OFFSETS)

    #t, x, y, yaw, v, d, a = do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state)

    elapsed_time = time.time() - start
    print(f"calc time:{elapsed_time:.6f} [sec]")

if __name__ == '__main__':
    main()