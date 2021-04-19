from ur5_env import *
from reward_functions import *
import cv2
from transform_utils import euler2quat

class pushpixel_env(object):
    def __init__(self, ur5_env, num_blocks=1, mov_dist=0.05, max_steps=50, task=0):
        self.env = ur5_env 
        self.num_blocks = num_blocks
        self.num_bins = 8

        self.task = task # 0: Reach / 1: Push
        self.mov_dist = mov_dist
        self.block_range_x = [-0.25, 0.25]
        self.block_range_y = [-0.15, 0.35]
        self.eef_range_x = [-0.3, 0.3]
        self.eef_range_y = [-0.2, 0.4]
        self.z_push = 1.05
        self.z_prepush = self.z_push + self.mov_dist
        self.z_collision_check = self.z_push + 0.025
        self.time_penalty = 0.02 #0.1
        self.max_steps = max_steps
        self.step_count = 0
        self.threshold = 0.05

        self.init_pos = [0.0, -0.23, 1.4]

        self.cam_id = 1
        self.cam_theta = 30 * np.pi / 180
        # cam_mat = self.env.sim.data.get_camera_xmat("rlview")
        # cam_pos = self.env.sim.data.get_camera_xpos("rlview")

        self.colors = np.array([ 
            [0.6784, 1.0, 0.1843], 
            [0.93, 0.545, 0.93], 
            [0.9686, 0.902, 0] 
            ])

        self.init_env()

    def get_reward(self):
        if self.task == 0:
            return reward_reach(self)
        elif self.task == 1:
            return reward_push_reverse(self)
            #return reward_push_dense(self)

    def init_env(self):
        self.env._init_robot()
        range_x = self.block_range_x
        range_y = self.block_range_y
        self.env.sim.data.qpos[12:15] = [0, 0, 0]
        self.env.sim.data.qpos[19:22] = [0, 0, 0]
        self.env.sim.data.qpos[26:29] = [0, 0, 0]
        self.goal1 = [0., 0.]
        self.goal2 = [0., 0.]
        self.goal3 = [0., 0.]
        self.success1 = False
        self.success2 = False
        self.success3 = False
        self.goal_image = np.zeros([self.env.camera_height, self.env.camera_width, 3])
        if self.num_blocks >= 1:
            tx1 = np.random.uniform(*range_x)
            ty1 = np.random.uniform(*range_y)
            tz1 = 0.9
            self.env.sim.data.qpos[12:15] = [tx1, ty1, tz1]
            x, y, z, w = euler2quat([0, 0, np.random.uniform(2*np.pi)])
            self.env.sim.data.qpos[15:19] = [w, x, y, z]
            gx1 = np.random.uniform(*range_x)
            gy1 = np.random.uniform(*range_y)
            self.goal1 = [gx1, gy1]
            # self.goal_image[self.pos2pixel(*self.goal1)] = self.colors[0]
            cv2.circle(self.goal_image, self.pos2pixel(*self.goal1), 1, self.colors[0], -1)
        if self.num_blocks >= 2:
            tx2 = np.random.uniform(*range_x)
            ty2 = np.random.uniform(*range_y)
            tz2 = 0.9
            self.env.sim.data.qpos[19:22] = [tx2, ty2, tz2]
            x, y, z, w = euler2quat([0, 0, np.random.uniform(2 * np.pi)])
            self.env.sim.data.qpos[22:26] = [w, x, y, z]
            gx2 = np.random.uniform(*range_x)
            gy2 = np.random.uniform(*range_y)
            self.goal2 = [gx2, gy2]
            # self.goal_image[self.pos2pixel(*self.goal2)] = self.colors[1]
            cv2.circle(self.goal_image, self.pos2pixel(*self.goal2), 1, self.colors[1], -1)
        if self.num_blocks >= 3:
            tx3 = np.random.uniform(*range_x)
            ty3 = np.random.uniform(*range_y)
            tz3 = 0.9
            self.env.sim.data.qpos[26:29] = [tx3, ty3, tz3]
            x, y, z, w = euler2quat([0, 0, np.random.uniform(2 * np.pi)])
            self.env.sim.data.qpos[29:33] = [w, x, y, z]
            gx3 = np.random.uniform(*range_x)
            gy3 = np.random.uniform(*range_y)
            self.goal3 = [gx3, gy3]
            # self.goal_image[self.pos2pixel(*self.goal3)] = self.colors[2]
            cv2.circle(self.goal_image, self.pos2pixel(*self.goal3), 1, self.colors[2], -1)

        im_state = self.env.move_to_pos(self.init_pos, grasp=1.0)
        if self.env.data_format=='NCHW':
            self.goal_image = np.transpose(self.goal_image, [2, 0, 1])
        self.step_count = 0

        return im_state

    def reset(self):
        glfw.destroy_window(self.env.viewer.window)
        self.env.viewer = None
        im_state = self.init_env()
        if self.task==0:
            return [im_state]
        else:
            return [im_state, self.goal_image]

    def step(self, action, grasp=1.0):
        self.pre_gripper_pos = deepcopy(self.env.sim.data.mocap_pos[0])
        self.pre_target_pos = deepcopy(self.env.sim.data.get_body_xpos('target_body_1'))
        self.pre_pos1 = deepcopy(self.env.sim.data.get_body_xpos('target_body_1')[:2])
        self.pre_pos2 = deepcopy(self.env.sim.data.get_body_xpos('target_body_2')[:2])
        self.pre_pos3 = deepcopy(self.env.sim.data.get_body_xpos('target_body_3')[:2])

        px, py, theta_idx = action
        if theta_idx >= self.num_bins:
            print("Error! theta_idx cannot be bigger than number of angle bins.")
            exit()
        theta = theta_idx * (2*np.pi / self.num_bins)
        im_state, collision = self.push_from_pixel(px, py, theta)

        reward, success = self.get_reward()
        if collision:
            reward = -0.1

        self.step_count += 1
        done = success
        if self.step_count==self.max_steps:
            done = True
        if not self.check_blocks_in_range():
            reward = -1.
            done = True

        info = {'collision': collision, 'success': success}
        pre_poses = np.array([self.pre_pos1, self.pre_pos2, self.pre_pos3])[:self.num_blocks]
        pos1 = deepcopy(self.env.sim.data.get_body_xpos('target_body_1')[:2])
        pos2 = deepcopy(self.env.sim.data.get_body_xpos('target_body_2')[:2])
        pos3 = deepcopy(self.env.sim.data.get_body_xpos('target_body_3')[:2])
        poses = np.array([pos1, pos2, pos3])[:self.num_blocks]
        info['pre_pose'] = pre_poses
        info['pose'] = poses

        if self.task == 0:
            return [im_state], reward, done, info
        else:
            return [im_state, self.goal_image], reward, done, info

    def clip_pos(self, pose):
        x, y = pose
        range_x = self.eef_range_x
        range_y = self.eef_range_y
        x = np.max((x, range_x[0]))
        x = np.min((x, range_x[1]))
        y = np.max((y, range_y[0]))
        y = np.min((y, range_y[1]))
        return x, y

    def check_blocks_in_range(self):
        pos1 = self.env.sim.data.get_body_xpos('target_body_1')[:2]
        pos2 = self.env.sim.data.get_body_xpos('target_body_2')[:2]
        pos3 = self.env.sim.data.get_body_xpos('target_body_3')[:2]
        poses = [pos1, pos2, pos3]
        self.block_range_x
        x_max, y_max = np.concatenate(poses[:self.num_blocks]).reshape(-1, 2).max(0)
        x_min, y_min = np.concatenate(poses[:self.num_blocks]).reshape(-1, 2).min(0)
        if x_max > self.block_range_x[1] or x_min < self.block_range_x[0]:
            return False
        if y_max > self.block_range_y[1] or y_min < self.block_range_y[0]:
            return False
        return True

    def push_from_pixel(self, px, py, theta):
        pos_before = np.array(self.pixel2pos(px, py))
        pos_before[:2] = self.clip_pos(pos_before[:2])
        pos_after = pos_before + self.mov_dist * np.array([np.sin(theta), np.cos(theta), 0.])
        pos_after[:2] = self.clip_pos(pos_after[:2])

        self.env.move_to_pos([pos_before[0], pos_before[1], self.z_prepush], grasp=1.0)
        self.env.move_to_pos([pos_before[0], pos_before[1], self.z_collision_check], grasp=1.0)
        force = self.env.sim.data.sensordata
        if np.abs(force[2]) > 1.0 or np.abs(force[5]) > 1.0:
            print("Collision!")
            self.env.move_to_pos([pos_before[0], pos_before[1], self.z_prepush], grasp=1.0)
            im_state = self.env.move_to_pos(self.init_pos, grasp=1.0)
            return im_state, True
        self.env.move_to_pos([pos_before[0], pos_before[1], self.z_push], grasp=1.0)
        self.env.move_to_pos([pos_after[0], pos_after[1], self.z_push], grasp=1.0)
        self.env.move_to_pos([pos_after[0], pos_after[1], self.z_prepush], grasp=1.0)
        im_state = self.env.move_to_pos(self.init_pos, grasp=1.0)
        return im_state, False

    def pixel2pos(self, u, v):
        theta = self.cam_theta
        cx, cy, cz = self.env.sim.model.cam_pos[self.cam_id]
        fovy = self.env.sim.model.cam_fovy[self.cam_id]
        f = 0.5 * self.env.camera_height / np.tan(fovy * np.pi / 360)
        u0 = 0.5 * self.env.camera_width
        v0 = 0.5 * self.env.camera_height
        z0 = 0.9  # table height
        y_cam = (cz - z0) / (np.sin(theta) + np.cos(theta) * f / (v - v0 + 1e-10))
        x_cam = (u - u0) / (v - v0 + 1e-10) * y_cam
        x = - x_cam
        y = np.tan(theta) * (z0 - cz) + cy + 1 / np.cos(theta) * y_cam
        z = z0
        # print("cam pos:", [x_cam, y_cam])
        # print("world pos:", [x, y])
        # print()
        return x, y, z

    def pos2pixel(self, x, y):
        theta = self.cam_theta
        cx, cy, cz = self.env.sim.model.cam_pos[self.cam_id]
        fovy = self.env.sim.model.cam_fovy[self.cam_id]
        f = 0.5 * self.env.camera_height / np.tan(fovy * np.pi / 360)
        u0 = 0.5 * self.env.camera_width
        v0 = 0.5 * self.env.camera_height
        z0 = 0.9  # table height
        y_cam = np.cos(theta) * (y - cy - np.tan(theta) * (z0 - cz))
        dv = f * np.cos(theta) / ((cz - z0) / y_cam - np.sin(theta))
        v = dv + v0
        u = - dv * x / y_cam + u0
        return int(u), int(v)

    def move2pixel(self, u, v):
        target_pos = np.array(self.pixel2pos(u, v))
        target_pos[2] = 1.05
        frame = self.env.move_to_pos(target_pos)
        plt.imshow(frame)
        plt.show()


if __name__=='__main__':
    visualize = True
    env = UR5Env(render=True, camera_height=64, camera_width=64, control_freq=5)
    env = pushpixel_env(env, num_blocks=2, mov_dist=0.05, max_steps=100, task=1)

    states = env.reset()
    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        s = deepcopy(states[0])
        s[states[1].max(2)!=0] = 0
        im = ax.imshow(s + states[1])
        plt.show(block=False)
        fig.canvas.draw()
        fig.canvas.draw()

    for i in range(100):
        #action = [np.random.randint(6), np.random.randint(2)]
        try:
            # action = input("Put action x, y, theta: ")
            # action = [int(a) for a in action.split()]
            action = [np.random.randint(10, 64), np.random.randint(10, 64), np.random.randint(8)]
        except KeyboardInterrupt:
            exit()
        except:
            continue
        if action[2] > 9:
            continue
        print('{} steps. action: {}'.format(env.step_count, action))
        states, reward, done, info = env.step(action)
        if visualize:
            s = deepcopy(states[0])
            s[states[1].max(2)!=0] = 0
            im = ax.imshow(s + states[1])
            fig.canvas.draw()

        print('Reward: {}. Done: {}'.format(reward, done))
        if done:
            print('Done. New episode starts.')
            states = env.reset()
            if visualize:
                s = deepcopy(states[0])
                s[states[1].max(2)!=0] = 0
                im = ax.imshow(s + states[1])
                fig.canvas.draw()
