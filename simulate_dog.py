import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("dog.xml")
data = mujoco.MjData(model)

# PD controller
Kp = 8.0   # Proportional gain
Kd = 1.5   # Derivative gain

q_des = np.zeros(8)
qd_des = np.zeros(8)

qpos_offset = model.nq - model.nu
qvel_offset = model.nv - model.nu

print("=" * 70)
print("Robot Dog Simulation - MuJoCo (2-Joint Legs)")
print("=" * 70)
print(f"Model loaded: dog.xml")
print(f"Number of position coordinates (nq): {model.nq}")
print(f"Number of velocity coordinates (nv): {model.nv}")
print(f"Number of actuators (nu): {model.nu}")
print(f"Joint structure: 4 legs × 2 joints (hip + knee) = 8 actuators")
print(f"Maximum torque per joint: 20 Nm")
print(f"PD Controller gains: Kp={Kp}, Kd={Kd}")
print(f"Position offset: {qpos_offset}, Velocity offset: {qvel_offset}")
print("=" * 70)
print("Implementing trotting gait with stance/swing phases")
print("=" * 70)
print()

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    last_print = 0
    initial_x = data.qpos[0]
    
    while viewer.is_running():
        t = time.time() - start_time
        
        gait_freq = 1.2      
        stance_ratio = 0.6   
        
        phase = (t * gait_freq) % 1.0
        
        pair1_in_stance = phase < stance_ratio
        pair1_phase_norm = (phase / stance_ratio) if pair1_in_stance else ((phase - stance_ratio) / (1 - stance_ratio))
        
        pair2_phase = (phase + 0.5) % 1.0
        pair2_in_stance = pair2_phase < stance_ratio
        pair2_phase_norm = (pair2_phase / stance_ratio) if pair2_in_stance else ((pair2_phase - stance_ratio) / (1 - stance_ratio))
        
        def get_leg_angles(phase_norm, in_stance):
            """Returns (hip_angle, knee_angle) for a leg"""
            if in_stance:
                hip = 0.25 - 0.5 * phase_norm     
                knee = -0.2                        
            else:
                hip = -0.25 + 0.5 * phase_norm    
                knee = -0.5 - 0.4 * np.sin(np.pi * phase_norm)
            return hip, knee
        
        fl_hip, fl_knee = get_leg_angles(pair1_phase_norm, pair1_in_stance)
        q_des[0] = fl_hip
        q_des[1] = fl_knee
        fr_hip, fr_knee = get_leg_angles(pair2_phase_norm, pair2_in_stance)
        q_des[2] = fr_hip
        q_des[3] = fr_knee
        
        bl_hip, bl_knee = get_leg_angles(pair2_phase_norm, pair2_in_stance)
        q_des[4] = bl_hip
        q_des[5] = bl_knee
        br_hip, br_knee = get_leg_angles(pair1_phase_norm, pair1_in_stance)
        q_des[6] = br_hip
        q_des[7] = br_knee
        
        q_actual = data.qpos[qpos_offset:]
        qd_actual = data.qvel[qvel_offset:]
        position_error = q_des - q_actual
        velocity_error = qd_des - qd_actual
        tau = Kp * position_error + Kd * velocity_error

        tau = np.clip(tau, -1.0, 1.0)

        data.ctrl[:] = tau

        mujoco.mj_step(model, data)

        viewer.sync()