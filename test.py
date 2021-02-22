from mujoco_py import load_model_from_path, MjSim, MjViewer 
 
import numpy as np 
import time

model = load_model_from_path('make_urdf/ur5_robotiq.xml') 
sim = MjSim(model) 
viewer = MjViewer(sim)  
viewer._hide_overlay = True

init_pos = np.array([np.pi, -np.pi/2, 0.0, 0.0, 0.0, 0.0]) 
print(len(sim.data.qpos))

for j in range(1000): 
    sim.data.qpos[:6] = init_pos 
    #random_action = np.random.rand(num_action) * 2.0 - 1.0 
    #sim.data.ctrl[:num_action] = np.zeros((num_action)) 
    #sim.data.ctrl[1] = -1.0 
    #sim.data.ctrl[6] = 0.1 
    sim.data.qpos[6] = 0.0
    sim.step() 
    viewer.render() 
    time.sleep(0.01)
