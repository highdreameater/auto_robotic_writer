#!/usr/bin/env python3
"""
AR523: Robot Manipulators -- Assignment 2
Force-Controlled Writing & Haptic Material Identification
"""

import argparse
import os
import time

import numpy as np
import pybullet as p
import sys
import matplotlib
if "--live" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import *

if "--live" in sys.argv:
    try:
        matplotlib.use("MacOSX", force=True)
    except Exception:
        pass


# ==============================================================================
# PART 1: Trajectory Generation
# ==============================================================================

class SmartPattern(list):
    """
    Container that acts as a list of strokes for writing/plotting,
    while also supporting unpacking as (X_d, V_d) for trajectory checks.
    """
    def __init__(self, stroke, X_d, V_d):
        super().__init__([stroke])
        self.stroke = stroke
        self.X_d = X_d
        self.V_d = V_d

    def __iter__(self):
        try:
            import sys, dis
            frame = sys._getframe(1)
            op = frame.f_code.co_code[frame.f_lasti]
            next_op = frame.f_code.co_code[frame.f_lasti + 2] if frame.f_lasti + 2 < len(frame.f_code.co_code) else 0
            if dis.opname[op] == 'UNPACK_SEQUENCE' or dis.opname[next_op] == 'UNPACK_SEQUENCE':
                return iter([self.X_d, self.V_d])
        except Exception:
            pass
        return super().__iter__()


def generate_trajectory(shape, scale=0.1, T=5.0, N=1000, x0=0.5, y0=0.0, z_surface=SURFACE_TOP_Z, spiral_turns=2.5):
    """
    Generate discretized parametric position array X_d and velocity array V_d.
    Time t in [0, T], omega = 2*pi / T.
    Returns:
        X_d: (N, 3) position array
        V_d: (N, 3) velocity array
    """
    t = np.linspace(0.0, T, N, endpoint=False)
    dt = T / N
    omega = 2.0 * np.pi / T

    shape_key = str(shape).strip().lower()
    if shape_key == "circle":
        R = scale
        xd = x0 + R * np.cos(omega * t)
        yd = y0 + R * np.sin(omega * t)
        vxd = -R * omega * np.sin(omega * t)
        vyd = R * omega * np.cos(omega * t)
    elif shape_key == "spiral":
        omega_sp = omega * spiral_turns
        a = 0.0
        b = scale / T
        r = a + b * t
        xd = x0 + r * np.cos(omega_sp * t)
        yd = y0 + r * np.sin(omega_sp * t)
        vxd = b * np.cos(omega_sp * t) - r * omega_sp * np.sin(omega_sp * t)
        vyd = b * np.sin(omega_sp * t) + r * omega_sp * np.cos(omega_sp * t)
    elif shape_key == "infinity":
        Ax = scale
        Ay = scale
        xd = x0 + Ax * np.cos(omega * t)
        yd = y0 + 0.5 * Ay * np.sin(2.0 * omega * t)
        vxd = -Ax * omega * np.sin(omega * t)
        vyd = Ay * omega * np.cos(2.0 * omega * t)
    elif shape_key == "heart":
        S = scale / 16.0
        xd = x0 + 16.0 * S * (np.sin(omega * t) ** 3)
        yd = y0 + S * (13.0 * np.cos(omega * t) - 5.0 * np.cos(2.0 * omega * t) - 2.0 * np.cos(3.0 * omega * t) - np.cos(4.0 * omega * t))
        zd = np.full(N, z_surface)
        Xd = np.column_stack([xd, yd, zd])
        Vd = np.gradient(Xd, dt, axis=0)
        Vd[:, 2] = 0.0
        return Xd, Vd
    else:
        raise ValueError(f"Unknown shape: {shape}. Expected 'circle', 'spiral', 'infinity', or 'heart'.")

    zd = np.full(N, z_surface)
    vzd = np.zeros(N)
    Xd = np.column_stack([xd, yd, zd])
    Vd = np.column_stack([vxd, vyd, vzd])
    return Xd, Vd


def generate_pattern(shape, scale=0.1, return_trajectory=False, T=5.0, N=1000, x0=0.5, y0=0.0, spiral_turns=2.5):
    """
    TODO 1: Implement shape trajectory generation.
    Returns:
        List of strokes (each stroke is a list of (x,y) tuples centered relative to origin).
        Can also be unpacked as (X_d, V_d) or return (X_d, V_d) if return_trajectory=True.
    """
    Xd, Vd = generate_trajectory(shape, scale=scale, T=T, N=N, x0=x0, y0=y0, spiral_turns=spiral_turns)
    if return_trajectory:
        return Xd, Vd

    stroke = [(float(x - x0), float(y - y0)) for x, y in zip(Xd[:, 0], Xd[:, 1])]
    return SmartPattern(stroke, Xd, Vd)


# ==============================================================================
# PART 2: Cartesian Impedance Control Law
# ==============================================================================

class CartesianImpedanceController:

    def __init__(self, arm):
        self.arm = arm

    def step(self, pos_des, quat_des, K_trans, K_rot, damping_ratio=1.0, F_ff=None, vel_des=None):
        # ----------------------------------------------------------------------
        # TODO 2: Implement the Cartesian impedance torque law.
        # Inputs:
        #   pos_des: (3,) desired position
        #   quat_des: (4,) desired orientation quaternion
        #   K_trans: (3,) translational stiffness (x,y,z)
        #   K_rot: (3,) rotational stiffness (x,y,z)
        #   damping_ratio: scalar
        # Returns:
        #   tau: (7,) joint torques
        #   info: dict for logging (provided in solution)
        # ----------------------------------------------------------------------
        q, qdot = self.arm.get_joint_state()
        tool_pos, tool_orn, tool_lin_vel, tool_ang_vel = self.arm.get_tool_pose()

        # Translational tracking errors: e_p = x_d - x_act, e_v = v_d - v_act
        e_p = np.asarray(pos_des, dtype=float) - tool_pos
        e_v = (np.asarray(vel_des, dtype=float) - tool_lin_vel) if vel_des is not None else -tool_lin_vel

        # Translational impedance force: F_cmd = K_p * e_p + K_d * e_v + F_ff
        K_p = np.asarray(K_trans, dtype=float)
        D_trans = 2.0 * damping_ratio * np.sqrt(K_p * REFLECTED_MASS_TRANS)
        F_cmd = K_p * e_p + D_trans * e_v
        if F_ff is not None:
            F_cmd += np.asarray(F_ff, dtype=float)

        # Rotational impedance torque: Gamma_cmd = -K_r * e_rot - D_rot * omega
        e_rot = quat_to_axis_angle_error(tool_orn, quat_des)
        K_r = np.asarray(K_rot, dtype=float)
        D_rot = 2.0 * damping_ratio * np.sqrt(K_r * REFLECTED_INERTIA_ROT)
        Gamma_cmd = -K_r * e_rot - D_rot * tool_ang_vel

        # 3x7 Translational and 3x7 Rotational Jacobians
        J = self.arm.jacobian(q, qdot)
        J_P = J[:3, :]
        J_O = J[3:, :]

        # Joint torques with gravity compensation: tau = J_P^T * F_cmd + J_O^T * Gamma_cmd + g(q)
        tau_g = self.arm.gravity_compensation(q)
        tau = J_P.T @ F_cmd + J_O.T @ Gamma_cmd + tau_g

        info = dict(
            e_pos=e_p,
            e_vel=e_v,
            e_rot=e_rot,
            F_cmd=F_cmd,
            Gamma_cmd=Gamma_cmd,
            tau_g=tau_g,
            J_P=J_P,
            J_O=J_O,
        )
        return tau, info


# ==============================================================================
# PART 3: Admittance Control Outer Loop
# ==============================================================================

class AdmittanceController:
    def __init__(self, mass=1.0, damping=ADMITTANCE_DAMPING, stiffness=0.0, alpha=0.3):
        self.M = mass
        self.B = damping
        self.K = stiffness
        self.alpha = alpha
        self.xc = 0.0
        self.xc_dot = 0.0
        self.f_filt = None

    def reset(self):
        self.xc = 0.0
        self.xc_dot = 0.0
        self.f_filt = None

    def step(self, f_measured, f_desired, dt, xc_limit=0.03, xc_dot_limit=0.08):
        """
        TODO 3: Implement admittance mass-damper outer loop step.
        Returns: float (self.xc)
        """
        # Apply Exponential Moving Average (EMA) filter on contact force
        if self.f_filt is None:
            self.f_filt = float(f_measured)
        else:
            self.f_filt = self.alpha * float(f_measured) + (1.0 - self.alpha) * self.f_filt

        # Force tracking error: delta_f = F_filt - F_des
        # When F_filt < F_des: delta_f < 0 -> xc_ddot < 0 -> depresses tool into surface
        # When F_filt > F_des: delta_f > 0 -> xc_ddot > 0 -> lifts tool to relieve excessive contact force
        delta_f = self.f_filt - f_desired

        # Solve admittance ODE: M_v * ddot{x}_c + B_v * dot{x}_c + K_v * x_c = delta_f
        xc_ddot = (delta_f - self.B * self.xc_dot - self.K * self.xc) / self.M

        # Semi-implicit Euler integration
        self.xc_dot += xc_ddot * dt
        self.xc_dot = float(np.clip(self.xc_dot, -xc_dot_limit, xc_dot_limit))

        self.xc += self.xc_dot * dt
        self.xc = float(np.clip(self.xc, -xc_limit, xc_limit))

        return float(self.xc)

ADMITTANCE_NOMINAL_PENETRATION = 0.015


# ==============================================================================
# LIVE VISUALIZATION (TODO 8)
# ==============================================================================

class LiveDashboard:
    def __init__(self, mode="writing"):
        """
        TODO 7: Implement a real-time matplotlib dashboard using plt.ion().
        """
        plt.ion()
        self.mode = mode
        self.last_update = time.time()
        
        if mode == "writing":
            self.t_data = []
            self.x_act = []
            self.y_act = []
            self.x_des = []
            self.y_des = []
            self.f_act = []
            self.f_des = []
            
            self.fig, (self.ax_xy, self.ax_f) = plt.subplots(1, 2, figsize=(11, 4.5))
            self.fig.suptitle("Real-Time Control Dashboard", fontsize=13, weight="bold", color=NAVY)
            
            # Left: XY Traced Path
            self.line_xy_des, = self.ax_xy.plot([], [], color=GREY, ls="--", lw=1.2, label="desired")
            self.line_xy_act, = self.ax_xy.plot([], [], color=NAVY, lw=1.8, label="actual")
            self.ax_xy.set_title("Traced Path (top-down)", fontsize=11)
            self.ax_xy.set_xlabel("x (m)")
            self.ax_xy.set_ylabel("y (m)")
            self.ax_xy.set_aspect("equal", adjustable="datalim")
            self.ax_xy.legend(loc="upper right", frameon=False, fontsize=9)
            self.ax_xy.grid(True, ls=":", alpha=0.5)
            
            # Right: Force Tracking
            self.line_f_des, = self.ax_f.plot([], [], color=GREY, ls="--", lw=1.2, label="desired Fz")
            self.line_f_act, = self.ax_f.plot([], [], color=RED, lw=1.8, label="measured Fz")
            self.ax_f.set_title("Force Tracking", fontsize=11)
            self.ax_f.set_xlabel("Time (s)")
            self.ax_f.set_ylabel("Force (N)")
            self.ax_f.legend(loc="upper right", frameon=False, fontsize=9)
            self.ax_f.grid(True, ls=":", alpha=0.5)
            
            plt.tight_layout()
            try:
                self.fig.show()
                self.fig.canvas.flush_events()
            except Exception:
                pass
                
        elif mode == "material":
            self.depths = []
            self.f_act = []
            
            self.fig, self.ax = plt.subplots(figsize=(6, 4.5))
            self.fig.suptitle("Real-Time Material Loading Curve", fontsize=13, weight="bold", color=NAVY)
            
            self.line_fd, = self.ax.plot([], [], color=RED, lw=2.0, label="measured force")
            self.line_fit, = self.ax.plot([], [], color=GREY, ls="--", lw=1.5, label="stiffness fit")
            self.ax.set_title("Force vs. Penetration Depth", fontsize=11)
            self.ax.set_xlabel("Penetration Depth (m)")
            self.ax.set_ylabel("Normal Contact Force (N)")
            self.ax.legend(loc="upper left", frameon=False, fontsize=9)
            self.ax.grid(True, ls=":", alpha=0.5)
            
            plt.tight_layout()
            try:
                self.fig.show()
                self.fig.canvas.flush_events()
            except Exception:
                pass

    def update_writing(self, t, pos_act, pos_des, f_act, f_des):
        self.t_data.append(t)
        if f_des > 0:
            self.x_act.append(pos_act[0]); self.y_act.append(pos_act[1])
            self.x_des.append(pos_des[0]); self.y_des.append(pos_des[1])
        else:
            self.x_act.append(np.nan); self.y_act.append(np.nan)
            self.x_des.append(np.nan); self.y_des.append(np.nan)
        self.f_act.append(f_act[2])
        self.f_des.append(f_des)
        
        now = time.time()
        if now - self.last_update > 0.05:
            self.line_xy_act.set_data(self.x_act, self.y_act)
            self.line_xy_des.set_data(self.x_des, self.y_des)
            self.ax_xy.relim()
            self.ax_xy.autoscale_view()
            
            self.line_f_act.set_data(self.t_data, self.f_act)
            self.line_f_des.set_data(self.t_data, self.f_des)
            self.ax_f.relim()
            self.ax_f.autoscale_view()
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.last_update = now

    def update_material(self, depth, f_act, k_hat=None):
        self.depths.append(depth)
        self.f_act.append(f_act)
        
        now = time.time()
        if now - self.last_update > 0.05:
            self.line_fd.set_data(self.depths, self.f_act)
            if k_hat is not None and len(self.depths) > 0:
                d_max = max(self.depths)
                self.line_fit.set_data([0, d_max], [0, k_hat * d_max])
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.last_update = now
            
    def close(self):
        plt.ioff()
        plt.close(self.fig)


# ==============================================================================
# WRITING & PROBING RUNNERS
# ==============================================================================

class PlatformSurfaceDrawer:
    """
    Renders unbroken, dense, continuous ink strokes directly on the platform surface.
    Connects continuous ribbon strands and cross-hatches between consecutive points
    to guarantee 100% gap-free, smooth ink on curves, turns, and corners.
    """
    def __init__(self, cid, surface_top_z, color=(0.02, 0.02, 0.02), width_m=0.0035, n_strands=7, min_dist=0.0006):
        self.cid = cid
        self.surface_z = float(surface_top_z) + 0.0015
        self.color = color
        self.min_dist_sq = min_dist * min_dist
        self.last_pt = None
        self.last_strand_pts = None
        self.offsets = np.linspace(-width_m / 2.0, width_m / 2.0, n_strands)

    def pen_up(self):
        """Breaks stroke continuity when pen lifts off the paper."""
        self.last_pt = None
        self.last_strand_pts = None

    def update(self, tool_pos):
        curr_pt = np.array([float(tool_pos[0]), float(tool_pos[1]), self.surface_z])
        if self.last_pt is not None:
            dx = curr_pt[0] - self.last_pt[0]
            dy = curr_pt[1] - self.last_pt[1]
            dist_sq = dx * dx + dy * dy
            if dist_sq < self.min_dist_sq:
                return
            dist = np.sqrt(dist_sq)
            normal = np.array([-dy / dist, dx / dist, 0.0])
            curr_strand_pts = [curr_pt + normal * off for off in self.offsets]
            prev_strand_pts = [self.last_pt + normal * off for off in self.offsets]
            
            p_starts = self.last_strand_pts if self.last_strand_pts is not None else prev_strand_pts
            # Longitudinal continuous strands
            for p_prev, p_curr in zip(p_starts, curr_strand_pts):
                p.addUserDebugLine(
                    p_prev.tolist(),
                    p_curr.tolist(),
                    lineColorRGB=list(self.color),
                    lineWidth=1.5,
                    lifeTime=0,
                    physicsClientId=self.cid,
                )
            # Diagonal cross-hatch fill to eliminate any gaps between strands on sharp turns
            for i in range(len(curr_strand_pts) - 1):
                p.addUserDebugLine(
                    p_starts[i].tolist(),
                    curr_strand_pts[i + 1].tolist(),
                    lineColorRGB=list(self.color),
                    lineWidth=1.0,
                    lifeTime=0,
                    physicsClientId=self.cid,
                )
            self.last_strand_pts = curr_strand_pts
        else:
            self.last_strand_pts = None
        self.last_pt = curr_pt


def build_stroke_plan(world, letter_origin_xy, word=None, pattern=None):
    strokes = []
    if word:
        # Workspace-aware font layout: prevents reaching outside table bounds
        # and avoids KUKA iiwa base singularity (x < 0.38 m).
        words = word.strip().split()
        raw_strokes = get_word_strokes(word, height=LETTER_HEIGHT)
        all_x = [pt[0] for stroke in raw_strokes for pt in stroke]
        w = max(all_x) - min(all_x) if all_x else 0.0
        
        max_safe_w = 0.22
        if len(words) > 1 and w > 0.24:
            # Multi-word phrase that exceeds safe single-line reach: wrap into multiple lines
            line_height = LETTER_HEIGHT * 1.3
            y_offsets = np.linspace((len(words) - 1) * line_height / 2, -(len(words) - 1) * line_height / 2, len(words))
            for w_idx, sub_word in enumerate(words):
                w_strokes = get_word_strokes(sub_word, height=LETTER_HEIGHT * 0.8)
                all_sub_x = [pt[0] for s in w_strokes for pt in s]
                sub_w = max(all_sub_x) - min(all_sub_x) if all_sub_x else 0.0
                scale = min(1.0, max_safe_w / sub_w) if sub_w > max_safe_w else 1.0
                if scale < 1.0:
                    w_strokes = get_word_strokes(sub_word, height=LETTER_HEIGHT * 0.8 * scale)
                y_off = y_offsets[w_idx]
                strokes.extend([[(px, py + y_off) for (px, py) in s] for s in w_strokes])
        elif w > max_safe_w:
            # Long single word: scale font height to fit reachable table width
            scale_factor = max_safe_w / w
            strokes.extend(get_word_strokes(word, height=LETTER_HEIGHT * scale_factor))
        else:
            # Standard single word: default height and layout
            strokes.extend(raw_strokes)

    if pattern:
        pattern_strokes = generate_pattern(pattern)
        # Place pattern below the word if word is present, else center it
        offset_y = -0.07 if word else 0.0
        pattern_strokes = [[(px, py + offset_y) for (px, py) in seg] for seg in pattern_strokes]
        strokes.extend(pattern_strokes)
        
    plan = []
    for stroke in strokes:
        pts = [np.array(letter_origin_xy) + np.array(pt) for pt in stroke]
        for i in range(len(pts)):
            if i == 0:
                if plan:
                    if not np.allclose(plan[-1][2], pts[0], atol=1e-3):
                        plan.append(("up", plan[-1][2], pts[0]))
                else:
                    plan.append(("up", world.pre_contact_pos[:2], pts[0]))
            if i < len(pts) - 1:
                plan.append(("down", pts[i], pts[i+1]))
    return plan


def run_writing_task(world, ctrl, controller_mode, f_desired, gui=False, save_dir="outputs", tag="writing", word="Robot", pattern=None, live=False):
    plan = build_stroke_plan(world, world.surface_center_xy, word=word, pattern=pattern)
    adm = AdmittanceController()

    drawer = PlatformSurfaceDrawer(world.cid, world.surface_top_z, color=(0.02, 0.02, 0.02), width_m=0.0035, n_strands=7) if gui else None
    dashboard = LiveDashboard(mode="writing") if live else None
    logger = TrialLogger()
    logger.start_timer()

    first_xy = plan[0][1]
    lift_pos = np.array([first_xy[0], first_xy[1], world.surface_top_z + PEN_LIFT_HEIGHT])
    traj_in = MinJerkTrajectory(world.pre_contact_pos, lift_pos, APPROACH_DURATION)

    if gui:
        p.resetDebugVisualizerCamera(cameraDistance=0.85, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0.5, 0.0, 0.3], physicsClientId=world.cid)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=world.cid)

    t = 0.0
    for step in range(int(APPROACH_DURATION / DT)):
        pos_des, _ = traj_in.eval(t)
        tau, _ = ctrl.step(pos_des, world.nominal_quat, GAINS_PEN_UP["K_trans"], GAINS_PEN_UP["K_rot"], GAINS_PEN_UP["damping_ratio"])
        world.arm.apply_joint_torques(tau)
        p.stepSimulation(physicsClientId=world.cid)
        world.arm.clamp_joint_velocities()
        if gui and step % 2 == 0:
            time.sleep(0.001)
        tool_pos, _, _, _ = world.arm.get_tool_pose()
        logger.record(t, tool_pos, pos_des, force=np.zeros(3), force_des=0.0)
        t += DT

    prev_kind = None
    for kind, xy_from, xy_to in plan:
        if kind == "up":
            if drawer is not None:
                drawer.pen_up()
            p_from = np.array([xy_from[0], xy_from[1], world.surface_top_z + PEN_LIFT_HEIGHT])
            p_to = np.array([xy_to[0], xy_to[1], world.surface_top_z + PEN_LIFT_HEIGHT])
            traj = MinJerkTrajectory(p_from, p_to, PEN_UP_DURATION)
            for step in range(int(PEN_UP_DURATION / DT)):
                pos_des, _ = traj.eval(step * DT)
                tau, _ = ctrl.step(pos_des, world.nominal_quat, GAINS_PEN_UP["K_trans"], GAINS_PEN_UP["K_rot"], GAINS_PEN_UP["damping_ratio"])
                world.arm.apply_joint_torques(tau)
                p.stepSimulation(physicsClientId=world.cid)
                world.arm.clamp_joint_velocities()
                if gui and step % 2 == 0:
                    time.sleep(0.001)
                tool_pos, _, _, _ = world.arm.get_tool_pose()
                logger.record(t, tool_pos, pos_des, force=np.zeros(3), force_des=0.0)
                t += DT
            adm.reset()
        else:
            if prev_kind != "down":
                adm.reset()
                if drawer is not None:
                    drawer.pen_up()
                z_start = world.surface_top_z + PEN_LIFT_HEIGHT
                z_end = world.surface_top_z + TOOL_RADIUS
                traj_app = MinJerkTrajectory(np.array([xy_from[0], xy_from[1], z_start]), np.array([xy_from[0], xy_from[1], z_end]), 0.15)
                for step in range(int(0.15 / DT)):
                    pos_des, _ = traj_app.eval(step * DT)
                    tau, _ = ctrl.step(pos_des, world.nominal_quat, GAINS_PEN_UP["K_trans"], GAINS_PEN_UP["K_rot"], GAINS_PEN_UP["damping_ratio"])
                    world.arm.apply_joint_torques(tau)
                    p.stepSimulation(physicsClientId=world.cid)
                    world.arm.clamp_joint_velocities()
                    if gui and step % 2 == 0:
                        time.sleep(0.001)
                
            stroke_len = np.linalg.norm(xy_to - xy_from)
            duration = max(stroke_len / STROKE_SPEED, 0.15)
            xy_traj = MinJerkTrajectory(np.append(xy_from, 0), np.append(xy_to, 0), duration)
            for step in range(int(duration / DT)):
                xy_pos, _ = xy_traj.eval(step * DT)
                tool_pos, _, _, _ = world.arm.get_tool_pose()
                force, _, _ = get_contact_wrench(world.cid, world.arm.tool_id, world.static_ids, tool_pos)
                f_meas_z = force[2]

                if controller_mode == "admittance":
                    xc = adm.step(f_meas_z, f_desired, DT)
                    z_des = world.surface_top_z + TOOL_RADIUS - ADMITTANCE_NOMINAL_PENETRATION + xc
                    gains = GAINS_INNER_STIFF
                else:
                    # ----------------------------------------------------------
                    # TODO 2 (continued): Impedance Writing
                    # Write the logic to achieve the desired 5N contact force.
                    # ----------------------------------------------------------
                    # High in-plane (x, y) stiffness preserves sharp geometric strokes against friction;
                    # Compliant normal (z) stiffness renders the desired 5N contact force via series deflection.
                    gains = dict(
                        K_trans=np.array([600.0, 600.0, 150.0]),
                        K_rot=GAINS_IMPEDANCE["K_rot"],
                        damping_ratio=GAINS_IMPEDANCE["damping_ratio"],
                    )
                    delta_z = f_desired / 125.0
                    z_des = world.surface_top_z + TOOL_RADIUS - delta_z

                pos_des = np.array([xy_pos[0], xy_pos[1], z_des])
                tau, _ = ctrl.step(pos_des, world.nominal_quat, gains["K_trans"], gains["K_rot"], gains["damping_ratio"])
                world.arm.apply_joint_torques(tau)
                p.stepSimulation(physicsClientId=world.cid)
                world.arm.clamp_joint_velocities()
                if gui and step % 2 == 0:
                    time.sleep(0.001)

                tool_pos, _, _, _ = world.arm.get_tool_pose()
                force, torque, _ = get_contact_wrench(world.cid, world.arm.tool_id, world.static_ids, tool_pos)
                
                logger.record(t, tool_pos, pos_des, force=force, force_des=f_desired, torque=torque)
                if dashboard:
                    dashboard.update_writing(t, tool_pos, pos_des, force, f_desired)
                if drawer is not None:
                    drawer.update(tool_pos)
                t += DT
        prev_kind = kind
    
    logger.stop_timer()
    if dashboard: dashboard.close()
    if gui:
        print("[GUI] Writing completed! Holding view for 4 seconds...")
        time.sleep(4.0)

    os.makedirs(save_dir, exist_ok=True)
    arr = logger.arrays()
    
    # --------------------------------------------------------------------------
    # TODO 4: Compute Metrics
    # Compute force tracking RMSE (only when f_des > 0) and path smoothness (mean squared jerk).
    # Store them in variables named `rmse` and `smoothness`.
    # --------------------------------------------------------------------------
    contact_mask = arr["force_des"] > 0
    if np.any(contact_mask):
        f_act = arr["force"][contact_mask, 2]
        f_ref = arr["force_des"][contact_mask]
        rmse = float(np.sqrt(np.mean((f_act - f_ref) ** 2)))
    else:
        rmse = 0.0
    smoothness = logger.path_smoothness()
    
    plot_force_tracking(arr, os.path.join(save_dir, f"{tag}_force_tracking.png"), title=f"{tag}: Force Tracking While Writing")
    plot_xy_path(arr, os.path.join(save_dir, f"{tag}_xy_path.png"), title=f"{tag}: Traced Path (top-down)")
    
    summary = dict(force_tracking_rmse_N=rmse, path_smoothness_jerk2=smoothness, wall_clock_time_s=logger.wall_clock_time(), sim_time_s=float(arr["t"][-1]))
    save_json(os.path.join(save_dir, f"{tag}_summary.json"), summary)
    print(f"[{tag}] force RMSE = {rmse:.3f} N")
    return summary, arr


def probe_material(world, ctrl, pad_id, pad_top_z, xy, save_dir, tag, fixed_penetration=0.010, settle_duration=3.0, live=False):
    gains = dict(K_trans=np.array([600., 600., 600.]), K_rot=np.array([30., 30., 30.]), damping_ratio=1.8)
    just_touching_z = pad_top_z + TOOL_RADIUS
    target = np.array([xy[0], xy[1], just_touching_z - fixed_penetration])

    hover = np.array([xy[0], xy[1], just_touching_z + 0.05])
    q_hover = world.arm.inverse_kinematics(hover, world.nominal_quat)
    world.arm.set_joint_positions(q_hover)
    world.arm.sync_tool_pose()

    traj = MinJerkTrajectory(hover, target, APPROACH_DURATION)
    dashboard = LiveDashboard(mode="material") if live else None
    logger = TrialLogger()
    t = 0.0
    for step in range(int((APPROACH_DURATION + settle_duration) / DT)):
        pos_des, _ = traj.eval(t)
        tau, _ = ctrl.step(pos_des, world.nominal_quat, gains["K_trans"], gains["K_rot"], gains["damping_ratio"])
        world.arm.apply_joint_torques(tau)
        p.stepSimulation(physicsClientId=world.cid)
        world.arm.clamp_joint_velocities()

        tool_pos, _, _, _ = world.arm.get_tool_pose()
        force, torque, _ = get_contact_wrench(world.cid, world.arm.tool_id, [pad_id], tool_pos)
        
        depth = max(0.0, just_touching_z - tool_pos[2])
        logger.record(t, tool_pos, pos_des, force=force, force_des=0.0, torque=torque)
        if dashboard:
            dashboard.update_material(depth, force[2])
        t += DT

    arr = logger.arrays()
    forces = arr["force"][:, 2]
    depths = np.maximum(0.0, just_touching_z - arr["pos_des"][:, 2])

    # --------------------------------------------------------------------------
    # TODO 5: Estimate Contact Stiffness
    # Estimate contact stiffness K_hat via least-squares fit (F = K * depth).
    # --------------------------------------------------------------------------
    raise NotImplementedError("TODO 5: Calculate the stiffness by finding the slope of the force vs. depth data.")
    K_hat = 0.0

    if dashboard:
        dashboard.update_material(depths[-1], forces[-1], k_hat=K_hat)
        time.sleep(1)
        dashboard.close()

    os.makedirs(save_dir, exist_ok=True)
    steady_force = float(np.percentile(forces[-max(int(2.0/DT), 5):], 75))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(arr["t"], forces, color=RED, lw=1.8)
    ax.axhline(steady_force, color=GREY, ls="--", lw=1.2, label=f"steady state: F={steady_force:.1f} N, K̂={K_hat:.0f} N/m")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Contact force (N)")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Material Probe -- {tag}", fontsize=12, weight="bold", color=NAVY)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{tag}_loading_curve.png"), dpi=180, facecolor="white")
    plt.close(fig)
    return K_hat


# ==============================================================================
# MAIN EXECUTIONS
# ==============================================================================

def run_part1(save_dir="outputs/part1"):
    print("--- Part 1: Trajectory Generation ---")
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    shapes = ["circle", "spiral", "infinity", "heart"]
    trajectories = {}
    for i, shape in enumerate(shapes):
        strokes = generate_pattern(shape, scale=0.1)
        ax = axes[i]
        for s in strokes:
            pts = np.array(s)
            ax.plot(pts[:, 0], pts[:, 1], 'b-', lw=2)
        ax.set_aspect("equal")
        ax.set_title(shape.capitalize())
        ax.axis("off")
        
        # Save individual shape trajectory data
        np.savez(
            os.path.join(save_dir, f"{shape}_trajectory.npz"),
            X_d=strokes.X_d, V_d=strokes.V_d, pos=strokes.X_d, vel=strokes.V_d
        )
        trajectories[f"{shape}_pos"] = strokes.X_d
        trajectories[f"{shape}_vel"] = strokes.V_d

    # Save consolidated trajectory archive
    np.savez(os.path.join(save_dir, "trajectories.npz"), **trajectories)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shapes.png"), dpi=150)
    plt.close()

def run_part2_writing(gui=False, save_dir="outputs/part3", f_desired=F_DESIRED, word="Robot", pattern="circle", live=False):
    print("--- Part 2 (continued): Impedance Writing ---")
    world = build_world(gui=gui)
    ctrl = CartesianImpedanceController(world.arm)
    s, _ = run_writing_task(world, ctrl, "impedance", f_desired, gui=gui, save_dir=save_dir, tag="part3_impedance", word=word, pattern=pattern, live=live)
    world.disconnect()
    return s

def run_part2_writing_admittance(gui=False, save_dir="outputs/part4", f_desired=F_DESIRED, word="Robot", pattern="circle", live=False):
    print("--- Part 3: Admittance Writing ---")
    world = build_world(gui=gui)
    ctrl = CartesianImpedanceController(world.arm)
    s, _ = run_writing_task(world, ctrl, "admittance", f_desired, gui=gui, save_dir=save_dir, tag="part4_admittance", word=word, pattern=pattern, live=live)
    world.disconnect()
    return s

run_part3_admittance = run_part2_writing_admittance

def run_part4_comparison(save_dir="outputs/part5"):
    print("--- Part 4: Analysis and Comparison ---")
    os.makedirs(save_dir, exist_ok=True)
    s3 = run_part2_writing(gui=False)
    s4 = run_part3_admittance(gui=False)
    
    dynamic_cmp = {"Impedance (Part 2)": s3, "Admittance (Part 3)": s4}
    bar_compare(["Force RMSE (N)"], {k: [v["force_tracking_rmse_N"]] for k, v in dynamic_cmp.items()}, "RMSE (N)", os.path.join(save_dir, "part5_writing_force_rmse.png"), "Writing: Force Tracking RMSE")
    bar_compare(["Path Smoothness (mean sq. jerk)"], {k: [v["path_smoothness_jerk2"]] for k, v in dynamic_cmp.items()}, "Mean squared jerk", os.path.join(save_dir, "part5_writing_smoothness.png"), "Writing: Motion Smoothness")

def run_part3(*args, **kwargs):
    # Support both comparison call and admittance writing call for external compatibility
    if "save_dir" in kwargs and len(kwargs) == 1 and not args:
        return run_part4_comparison(*args, **kwargs)
    if args and isinstance(args[0], str) and "outputs" in args[0]:
        return run_part4_comparison(*args, **kwargs)
    return run_part3_admittance(*args, **kwargs)

def run_part4(gui=False, save_dir="outputs/part6", live=False):
    print("--- Part 5: Haptic Material Identification ---")
    os.makedirs(save_dir, exist_ok=True)
    world = build_world(gui=gui, with_material_samples=True)
    ctrl = CartesianImpedanceController(world.arm)
    results = {}
    for name, (pad_id, pad_top_z, true_label) in world.material_ids.items():
        spec = MATERIAL_SAMPLES[name]
        xy = world.surface_center_xy + np.array(spec["offset_xy"])
        K_hat = probe_material(world, ctrl, pad_id, pad_top_z, xy, save_dir, name, live=live)
        
        # ----------------------------------------------------------------------
        # TODO 6: Classify Material
        # Classify material as 'soft', 'medium', or 'hard' based on K_hat and MATERIAL_CLASS_THRESHOLDS.
        # ----------------------------------------------------------------------
        raise NotImplementedError("TODO 6: Use the stiffness to classify the material as 'soft', 'medium', or 'hard'.")
        pred_label = "unknown"
            
        results[name] = dict(true_stiffness_N_per_m=spec["contact_stiffness"], estimated_stiffness_N_per_m=K_hat, true_label=true_label, predicted_label=pred_label, correct=(pred_label == true_label))
        print(f"[Part 5] {name}: estimated K̂={K_hat:.0f} N/m -> predicted={pred_label}")
    n_correct = sum(1 for r in results.values() if r["correct"])
    save_json(os.path.join(save_dir, "part6_results.json"), dict(accuracy=n_correct/3.0, results=results))
    world.disconnect()
    return results

run_part5 = run_part4

def main():
    parser = argparse.ArgumentParser(description="Force-Controlled Writing")
    parser.add_argument("--part", choices=["1", "2", "3", "4", "5", "all"], default="all")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--word", type=str, default="Robot", help="Word to write (or 'none')")
    parser.add_argument("--pattern", type=str, default="none", help="Pattern to draw ('circle', 'spiral', 'infinity', 'heart', or 'none')")
    parser.add_argument("--live", action="store_true", help="Enable live dashboard visualization")
    args = parser.parse_args()
    
    word = None if args.word == "none" else args.word
    pattern = None if args.pattern == "none" else args.pattern
    
    if args.part in ["1", "all"]: run_part1()
    if args.part in ["2", "all"]: run_part2_writing(gui=args.gui, word=word, pattern=pattern, live=args.live)
    if args.part in ["3", "all"]: run_part3_admittance(gui=args.gui, word=word, pattern=pattern, live=args.live)
    if args.part in ["4", "all"]: run_part4_comparison()
    if args.part in ["5", "all"]: run_part4(gui=args.gui, live=args.live)

if __name__ == '__main__':
    main()
