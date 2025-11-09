import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("dog.xml")
data = mujoco.MjData(model)

Kp = 8.0 
Kd = 1.5 

q_des = np.zeros(4)  # 4 joints
qd_des = np.zeros(4)

qpos_offset = model.nq - model.nu
qvel_offset = model.nv - model.nu

print("=" * 60)
print("Robot Dog Simulation - MuJoCo")
print("=" * 60)
print(f"Model loaded: dog.xml")
print(f"Number of position coordinates (nq): {model.nq}")
print(f"Number of velocity coordinates (nv): {model.nv}")
print(f"Number of actuators (nu): {model.nu}")
print(f"Maximum torque per joint: 20 Nm")
print(f"PD Controller gains: Kp={Kp}, Kd={Kd}")
print(f"Position offset: {qpos_offset}, Velocity offset: {qvel_offset}")
print("=" * 60)
print("Starting simulation...")
print("Controls:")
print("  - Left click + drag: Rotate view")
print("  - Right click + drag: Pan view")
print("  - Scroll: Zoom")
print("  - Space: Pause/Resume")
print("  - Close window to exit")
print("=" * 60)
print()

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    last_print = 0
    
    while viewer.is_running():
        t = time.time() - start_time
        
        amplitude = 0.3      
        frequency = 1.2      
        offset = 0.15      
        
        phase = 2 * np.pi * frequency * t
        
        q_des[0] = offset + amplitude * np.sin(phase)      # Front Left
        q_des[3] = -offset + amplitude * np.sin(phase)     # Back Right
        
        q_des[1] = offset + amplitude * np.sin(phase + np.pi)  # Front Right
        q_des[2] = -offset + amplitude * np.sin(phase + np.pi) # Back Left
        
        q_actual = data.qpos[qpos_offset:]   # Current joint positions
        qd_actual = data.qvel[qvel_offset:]  # Current joint velocities
        
        position_error = q_des - q_actual
        velocity_error = qd_des - qd_actual
        tau = Kp * position_error + Kd * velocity_error
        
        tau = np.clip(tau, -1.0, 1.0)
        
        data.ctrl[:] = tau
        
        mujoco.mj_step(model, data)
        
        viewer.sync()