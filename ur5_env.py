from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py import MjRenderContextOffscreen
import mujoco_py

#from IPython.display import HTML, display
#import base64
#import glob
#import io
import cv2
from matplotlib import pyplot as plt
from copy import deepcopy
import numpy as np
import imageio
import types
import time


def save_video(frames, filename='video/mujoco.mp4', fps=60):
    writer = imageio.get_writer(filename, fps=fps)
    for f in frames:
        writer.append_data(f)
    writer.close()
    
"""
def show_video(filname='video/mujoco.mp4'):
    mp4list = glob.glob(filname)
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay 
            loop controls style="height: 400px;">
            <source src="data:video/mp4;base64,{0}" type="video/mp4" />
            </video>'''.format(encoded.decode('ascii'))))
    else: 
        print("Could not find video")
"""
        
def show_image(img):
    #cv2.imshow("test", img)
    plt.figure(figsize = (16,9))
    plt.axis('off')
    plt.imshow(img)


class UR5Env():
    def __init__(
            self, 
            render=True,
            image_state=True,
            camera_height=128,
            camera_width=128,
            control_freq=5,
            ):

        self.render = render
        self.image_state = image_state
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.control_freq = control_freq

        self._init_robot()

    def _init_robot(self):
        self.model = load_model_from_path('make_urdf/ur5_robotiq.xml')
        self.n_substeps = 1 #20
        self.sim = MjSim(self.model, nsubsteps=self.n_substeps)
        self.viewer = MjViewer(self.sim)
        self.viewer._hide_overlay = True

        self.arm_joint_list = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.gripper_joint_list = ['left_finger_joint', 'right_finger_joint']

        # Camera pose
        lookat_refer = self.sim.data.get_body_xpos('target_body_1')
        self.viewer.cam.lookat[0] = lookat_refer[0]
        self.viewer.cam.lookat[1] = lookat_refer[1]
        self.viewer.cam.lookat[2] = lookat_refer[2]
        self.viewer.cam.azimuth = -90 #-75
        self.viewer.cam.elevation = -60 #-15
        self.viewer.cam.distance = 1.5

        str = ''
        for joint_name in self.gripper_joint_list:
            str += "{} : {:.3f}, ".format(joint_name, self.sim.data.get_joint_qpos(joint_name))
        print(str)


        self.init_arm_pos = np.array([-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, 0.0])
        for joint_idx, joint_name in enumerate(self.arm_joint_list):
            self.sim.data.set_joint_qpos(joint_name, self.init_arm_pos[joint_idx])
            self.sim.data.set_joint_qvel(joint_name, 0.0)
        self.reset_mocap_welds()

        # Move end effector into position.
        gripper_target = np.array([0.0, 0.0, -0.1]) + self.sim.data.get_body_xpos('wrist_3_link')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()
            self.sim.render(mode='window')

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation. """
        if self.sim.model.nmocap > 0 and self.sim.model.eq_data is not None:
            for i in range(self.sim.model.eq_data.shape[0]):
                if self.sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    self.sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        self.sim.forward()
    
    def move_to_pos(self, pos=[0.0, 0.0, 0.63], quat=[0, 1, 0, 0], grasp=0.0):
        self.sim.data.mocap_pos[0] = self.sim.data.get_body_xpos('box_link') + np.array(pos)
        self.sim.data.mocap_quat[0] = np.array(quat)
        self.sim.data.ctrl[0] = grasp
        self.sim.data.ctrl[1] = grasp
        
        control_timestep = 1. / self.control_freq
        cur_time = time.time()
        end_time = cur_time + control_timestep

        #for i in range(10):
        while cur_time < end_time:
            self.sim.step()
            cur_time += self.sim.model.opt.timestep
            self.sim.render(mode='window')
        diff_pos = np.linalg.norm(self.sim.data.mocap_pos[0] - self.sim.data.get_body_xpos('robot0:mocap'))
        diff_quat = np.linalg.norm(self.sim.data.mocap_quat[0] - self.sim.data.get_body_xquat('robot0:mocap'))
        if diff_pos + diff_quat > 1e-3:
            print('Failed to move to target position.')

        #self.viewer._set_mujoco_buffers()
        im_state = self.viewer._read_pixels_as_in_window()
        #im_state = self.sim.render(camera_name="rlview", width=self.camera_width, height=self.camera_height, mode='offscreen')
        #im_state = self.sim.render(camera_name="rlview", width=self.camera_width, height=self.camera_height, mode='offscreen')
        #self.viewer._set_mujoco_buffers()
        return im_state

    def move_pos_diff(self, posdiff, quat=[0, 1, 0, 0], grasp=0.0):
        cur_pos = deepcopy(self.sim.data.mocap_pos[0]) - self.sim.data.get_body_xpos('box_link')
        target_pos = cur_pos + np.array(posdiff)
        return self.move_to_pos(target_pos, quat, grasp)


def discrete_env(object):
    def __init__(self, ur5_env, mov_dist=0.03):
        self.env = ur5_env 
        self.mov_dist = mov_dist

    def step(self, action, grasp):
        if action==0:
            env.move_pos_diff([0.0, dist, 0.0], grasp=grasp)
        elif action==1:
            env.move_pos_diff([dist, 0.0, 0.0], grasp=grasp)
        elif action==2:
            env.move_pos_diff([0.0, -dist, 0.0], grasp=grasp)
        elif action==3:
            env.move_pos_diff([-dist, 0.0, 0.0], grasp=grasp)
        elif action==4:
            env.move_pos_diff([0.0, 0.0, dist], grasp=grasp)
        elif action==5:
            env.move_pos_diff([0.0, 0.0, -dist], grasp=grasp)
        else:
            print("Error!! Wrong action!")

        state = self.get_state()
        reward, done = self.get_reward()

    def get_state(self):
        return

    def get_reward(self):
        return


if __name__=='__main__':
    env = UR5Env()
    env.move_to_pos()

    grasp = 0.0
    for i in range(100):
        x = input('Ctrl+c to exit. next?')
        if x==' ':
            x = x[0]
            grasp = 1.0 - grasp
            env.move_pos_diff([0.0, 0.0, 0.0], grasp=grasp)
            continue

        dist = 0.03
        if x=='w':
            frame = env.move_pos_diff([0.0, 0.0, dist], grasp=grasp)
        elif x=='s':
            frame = env.move_pos_diff([0.0, 0.0, -dist], grasp=grasp)
        elif x=='6':
            frame = env.move_pos_diff([dist, 0.0, 0.0], grasp=grasp)
        elif x=='4':
            frame = env.move_pos_diff([-dist, 0.0, 0.0], grasp=grasp)
        elif x=='8':
            frame = env.move_pos_diff([0.0, dist, 0.0], grasp=grasp)
        elif x=='2':
            frame = env.move_pos_diff([0.0, -dist, 0.0], grasp=grasp)

        plt.imshow(frame)
        plt.show()

    '''
    env.move_to_pos([0, 0.2, 0.85], grasp=0.0)
    env.move_to_pos([0, 0.2, 0.61], grasp=0.0)
    env.move_to_pos([0, 0.2, 0.61], grasp=1.0)
    env.move_to_pos([0, 0.2, 0.85], grasp=1.0)

    env.move_pos_diff([0.0, -0.05, 0], grasp=1.0)
    env.move_pos_diff([0.0, -0.05, 0], grasp=1.0)
    env.move_pos_diff([0.05, -0.05, 0], grasp=1.0)
    env.move_pos_diff([0.05, -0.05, 0], grasp=1.0)
    env.move_pos_diff([0.0, -0.0, 0.05], grasp=1.0)
    env.move_pos_diff([0.0, -0.0, 0.05], grasp=1.0)
    '''


