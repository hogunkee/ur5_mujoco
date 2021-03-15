from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py import MjRenderContextOffscreen
import mujoco_py

#from IPython.display import HTML, display
#import base64
#import glob
#import io
import cv2
import glfw
from matplotlib import pyplot as plt
from copy import deepcopy
import numpy as np
import imageio
import types
import time

import os
file_path = os.path.dirname(os.path.abspath(__file__))


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
    plt.show()


class UR5Env():
    def __init__(
            self, 
            render=True,
            image_state=True,
            camera_height=64,
            camera_width=64,
            control_freq=8,
            data_format='NHWC',
            ):

        self.model_xml = 'make_urdf/ur5_robotiq_cube.xml'

        self.render = render
        self.image_state = image_state
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.control_freq = control_freq
        self.data_format = data_format

        self._init_robot()

    def _init_robot(self):
        self.model = load_model_from_path(os.path.join(file_path, self.model_xml))
        #self.model = load_model_from_path(os.path.join(file_path, 'make_urdf/ur5_robotiq.xml'))
        self.n_substeps = 1 #20
        self.sim = MjSim(self.model, nsubsteps=self.n_substeps)
        self.viewer = MjViewer(self.sim)
        self.viewer._hide_overlay = True

        self.arm_joint_list = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.gripper_joint_list = ['left_finger_joint', 'right_finger_joint']

        # Camera pose
        lookat_refer = [0., 0., 0.9] #self.sim.data.get_body_xpos('target_body_1')
        self.viewer.cam.lookat[0] = lookat_refer[0]
        self.viewer.cam.lookat[1] = lookat_refer[1]
        self.viewer.cam.lookat[2] = lookat_refer[2]
        self.viewer.cam.azimuth = 35 #-75 #-90 #-75
        self.viewer.cam.elevation = -30 #-60 #-15
        self.viewer.cam.distance = 1.5

        str = ''
        for joint_name in self.gripper_joint_list:
            str += "{} : {:.3f}, ".format(joint_name, self.sim.data.get_joint_qpos(joint_name))
        # print(str)


        self.init_arm_pos = np.array([-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, 0.0])
        for joint_idx, joint_name in enumerate(self.arm_joint_list):
            self.sim.data.set_joint_qpos(joint_name, self.init_arm_pos[joint_idx])
            self.sim.data.set_joint_qvel(joint_name, 0.0)
        self.reset_mocap_welds()

        '''
        # Move end effector into position.
        gripper_target = np.array([0.0, 0.0, -0.1]) + self.sim.data.get_body_xpos('wrist_3_link')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        '''
        im_state = self.move_to_pos()
        return im_state
        '''
        for _ in range(10):
            self.sim.step()
            if self.render: self.sim.render(mode='window')
            else: self.sim.render(camera_name="rlview", width=self.camera_width, height=self.camera_height, mode='offscreen')
        '''

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation. """
        if self.sim.model.nmocap > 0 and self.sim.model.eq_data is not None:
            for i in range(self.sim.model.eq_data.shape[0]):
                if self.sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    self.sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        self.sim.forward()
    
    def move_to_pos(self, pos=[0.0, 0.0, 1.20], quat=[0, 1, 0, 0], grasp=0.0):
        self.sim.data.mocap_pos[0] = np.array(pos)
        #self.sim.data.mocap_pos[0] = self.sim.data.get_body_xpos('box_link') + np.array(pos)
        self.sim.data.mocap_quat[0] = np.array(quat)
        
        control_timestep = 1. / self.control_freq
        cur_time = time.time()
        end_time = cur_time + control_timestep

        while cur_time < end_time:
            self.sim.step()
            cur_time += self.sim.model.opt.timestep
            if self.render: self.sim.render(mode='window')
            else: self.sim.render(camera_name="rlview", width=self.camera_width, height=self.camera_height, mode='offscreen')

        pre_grasp = float(bool(sum(self.sim.data.ctrl)))
        self.sim.data.ctrl[0] = grasp
        self.sim.data.ctrl[1] = grasp
        if grasp != pre_grasp:
            cur_time = time.time()
            end_time = cur_time + 2.0*control_timestep
            #for i in range(20):
            while cur_time < end_time:
                self.sim.step()
                cur_time += self.sim.model.opt.timestep
                if self.render: self.sim.render(mode='window')
                else: self.sim.render(camera_name="rlview", width=self.camera_width, height=self.camera_height, mode='offscreen')

        diff_pos = np.linalg.norm(self.sim.data.mocap_pos[0] - self.sim.data.get_body_xpos('robot0:mocap'))
        diff_quat = np.linalg.norm(self.sim.data.mocap_quat[0] - self.sim.data.get_body_xquat('robot0:mocap'))
        if diff_pos + diff_quat > 1e-3:
            print('Failed to move to target position.')

        if self.render:
            self.viewer._set_mujoco_buffers()
            '''
            im_state = self.viewer._read_pixels_as_in_window()
            im_state = self.viewer._read_pixels_as_in_window()
            '''
            im_state = self.sim.render(camera_name="rlview", width=self.camera_width, height=self.camera_height, mode='offscreen')
            im_state = self.sim.render(camera_name="rlview", width=self.camera_width, height=self.camera_height, mode='offscreen')
            im_state = np.flip(im_state, axis=1)
            self.viewer._set_mujoco_buffers()
        else:
            im_state = self.sim.render(camera_name="rlview", width=self.camera_width, height=self.camera_height, mode='offscreen')
            im_state = np.flip(im_state, axis=1)
        if self.data_format=='NCHW':
            im_state = np.transpose(im_state, [2, 0, 1])
        return im_state

    def move_pos_diff(self, posdiff, quat=[0, 1, 0, 0], grasp=0.0):
        cur_pos = deepcopy(self.sim.data.mocap_pos[0])# - self.sim.data.get_body_xpos('box_link')
        target_pos = cur_pos + np.array(posdiff)
        return self.move_to_pos(target_pos, quat, grasp)


class discrete_env(object):
    def __init__(self, ur5_env, task=1, mov_dist=0.03, max_steps=50):
        self.action_type="disc_10d"
        self.env = ur5_env 
        self.init_pos = [0.0, 0.0, 1.10]
        self.task = task
        if self.task==0:
            self.init_pos = [0.0, 0.0, 1.20]
        elif self.task==1:
            self.threshold = 0.4
        elif self.task==2:
            self.threshold = 0.1
        elif self.task==3:
            self.threshold = np.pi/18
        else:
            pass
        self.mov_dist = mov_dist
        self.z_min = 1.05
        self.time_penalty = 1e-3
        self.max_steps = max_steps
        self.step_count = 0

        self.env.move_to_pos(self.init_pos, grasp=1.0)

    def reset(self):
        glfw.destroy_window(self.env.viewer.window)
        self.env.viewer = None
        im_state = self.env._init_robot()
        gripper_height, curr_grasp = self.get_gripper_state()
        self.env.move_to_pos(self.init_pos)
        self.step_count = 0
        return im_state, np.array([gripper_height, curr_grasp])

    def step(self, action, grasp):
        self.pre_mocap_pos = deepcopy(self.env.sim.data.mocap_pos[0])
        self.pre_target_pos = deepcopy(self.env.sim.data.get_body_xpos('target_body_1'))

        dist = self.mov_dist
        if self.action_type=="disc_6d":
            if action==0:
                im_state = self.env.move_pos_diff([0.0, dist, 0.0], grasp=grasp)
            elif action==1:
                im_state = self.env.move_pos_diff([dist, 0.0, 0.0], grasp=grasp)
            elif action==2:
                im_state = self.env.move_pos_diff([0.0, -dist, 0.0], grasp=grasp)
            elif action==3:
                im_state = self.env.move_pos_diff([-dist, 0.0, 0.0], grasp=grasp)
            elif action==4:
                im_state = self.env.move_pos_diff([0.0, 0.0, dist], grasp=grasp)
            elif action==5:
                if self.pre_mocap_pos[2]-dist < self.z_min:
                    im_state = self.env.move_pos_diff([0.0, 0.0, -self.pre_mocap_pos[2]+self.z_min], grasp=grasp)
                else:
                    im_state = self.env.move_pos_diff([0.0, 0.0, -dist], grasp=grasp)
            else:
                print("Error!! Wrong action!")
        elif self.action_type=="disc_10d":
            dist2 = dist/np.sqrt(2)
            if action==0:
                if self.pre_mocap_pos[2]-dist < self.z_min:
                    im_state = self.env.move_pos_diff([0.0, 0.0, -self.pre_mocap_pos[2]+self.z_min], grasp=grasp)
                else:
                    im_state = self.env.move_pos_diff([0.0, 0.0, -dist], grasp=grasp)
            elif action==5:
                im_state = self.env.move_pos_diff([0.0, 0.0, dist], grasp=grasp)
            elif action==8:
                im_state = self.env.move_pos_diff([0.0, dist, 0.0], grasp=grasp)
            elif action==9:
                im_state = self.env.move_pos_diff([dist2, dist2, 0.0], grasp=grasp)
            elif action==6:
                im_state = self.env.move_pos_diff([dist, 0.0, 0.0], grasp=grasp)
            elif action==3:
                im_state = self.env.move_pos_diff([dist2, -dist2, 0.0], grasp=grasp)
            elif action==2:
                im_state = self.env.move_pos_diff([0.0, -dist, 0.0], grasp=grasp)
            elif action==1:
                im_state = self.env.move_pos_diff([-dist2, -dist2, 0.0], grasp=grasp)
            elif action==4:
                im_state = self.env.move_pos_diff([-dist, 0.0, 0.0], grasp=grasp)
            elif action==7:
                im_state = self.env.move_pos_diff([-dist2, dist2, 0.0], grasp=grasp)
            else:
                print("Error!! Wrong action!")

        gripper_height, curr_grasp = self.get_gripper_state()
        reward, done = self.get_reward()

        self.step_count += 1
        if self.step_count==self.max_steps:
            done = True
        return [im_state, np.array([gripper_height, curr_grasp])], reward, done, None

    def get_gripper_state(self):
        # return grasp_height, gripper_close
        return self.env.sim.data.mocap_pos[0][2], int(bool(sum(self.env.sim.data.ctrl)))

    def get_reward(self):
        # 0: Reach   #
        # 1: Spread  #
        # 2: Gather  #
        # 3: Line up #
        ## Reach ##
        if self.task == 0:
            target_pos = self.env.sim.data.get_body_xpos('target_body_1')
            if np.linalg.norm(target_pos - self.pre_target_pos) > 1e-3:
                reward = 1.0
                done = True
            else:
                reward = -self.time_penalty
                done = False
        ## Spread ##
        elif self.task == 1:
            pos_1 = self.env.sim.data.get_body_xpos('target_body_1')
            pos_2 = self.env.sim.data.get_body_xpos('target_body_2')
            pos_3 = self.env.sim.data.get_body_xpos('target_body_3')
            dist_12 = np.linalg.norm(pos_1 - pos_2)
            dist_23 = np.linalg.norm(pos_2 - pos_3)
            dist_31 = np.linalg.norm(pos_3 - pos_1)
            min_dist = np.min([dist_12, dist_23, dist_31])
            if min_dist > self.threshold:
                reward = 1.0
                done = True
            else:
                reward = 0.0
                done = False
        ## Gather ##
        elif self.task == 2:
            pos_1 = self.env.sim.data.get_body_xpos('target_body_1')
            pos_2 = self.env.sim.data.get_body_xpos('target_body_2')
            pos_3 = self.env.sim.data.get_body_xpos('target_body_3')
            dist_12 = np.linalg.norm(pos_1 - pos_2)
            dist_23 = np.linalg.norm(pos_2 - pos_3)
            dist_31 = np.linalg.norm(pos_3 - pos_1)
            max_dist = np.max([dist_12, dist_23, dist_31])
            if max_dist < self.threshold:
                reward = 1.0
                done = True
            else:
                reward = 0.0
                done = False
        ## Line Up ##
        elif self.task==3:
            pos_1 = self.env.sim.data.get_body_xpos('target_body_1')
            pos_2 = self.env.sim.data.get_body_xpos('target_body_2')
            pos_3 = self.env.sim.data.get_body_xpos('target_body_3')
            theta_12 = np.arctan((pos_1[1] - pos_2[1])/(pos_1[0] - pos_2[0] + 1e-10))
            theta_23 = np.arctan((pos_2[1] - pos_3[1])/(pos_2[0] - pos_3[0] + 1e-10))
            theta_31 = np.arctan((pos_3[1] - pos_1[1])/(pos_3[0] - pos_1[0] + 1e-10))
            dtheta_1 = np.min([(theta_12-theta_31)%np.pi, np.abs((theta_12-theta_31)%np.pi-np.pi)])
            dtheta_2 = np.min([(theta_12-theta_23)%np.pi, np.abs((theta_12-theta_23)%np.pi-np.pi)])
            dtheta_3 = np.min([(theta_31-theta_23)%np.pi, np.abs((theta_31-theta_23)%np.pi-np.pi)])
            max_dtheta = np.max([dtheta_1, dtheta_2, dtheta_3])
            if max_dtheta < self.threshold:
                reward = 1.0
                done = True
            else:
                reward = 0.0
                done = False
        else:
            reward = 0.0
            done = False
        return reward, done


if __name__=='__main__':
    env = UR5Env(render=True, camera_height=96, camera_width=96)
    env = discrete_env(env, task=3, mov_dist=0.03, max_steps=100)

    for i in range(100):
        #action = [np.random.randint(6), np.random.randint(2)]
        try:
            action = [int(input("action? ")), 1]
        except KeyboardInterrupt:
            exit()
        except:
            continue
        if action[0] > 9:
            continue
        print('{} steps. action: {}'.format(env.step_count, action))
        states, reward, done, info = env.step(*action)
        print('Reward: {}. Done: {}'.format(reward, done))
        #show_image(states[0])
        if done:
            print('Done. New episode starts.')
            env.reset()

    '''
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
            im_state = env.move_pos_diff([0.0, 0.0, dist], grasp=grasp)
        elif x=='s':
            im_state = env.move_pos_diff([0.0, 0.0, -dist], grasp=grasp)
        elif x=='6':
            im_state = env.move_pos_diff([dist, 0.0, 0.0], grasp=grasp)
        elif x=='4':
            im_state = env.move_pos_diff([-dist, 0.0, 0.0], grasp=grasp)
        elif x=='8':
            im_state = env.move_pos_diff([0.0, dist, 0.0], grasp=grasp)
        elif x=='2':
            im_state = env.move_pos_diff([0.0, -dist, 0.0], grasp=grasp)

        plt.imshow(im_state)
        plt.show()
    '''
