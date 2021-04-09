from ur5_env import *
import cv2
from transform_utils import euler2quat

class pushpixel_env(object):
    def __init__(self, ur5_env, num_blocks=1, mov_dist=0.05, max_steps=50, task=0):
        self.env = ur5_env 
        self.num_blocks = num_blocks
        self.num_bins = 8

        self.task = task # 0: Reach / 1: Push
        if self.task==0:
            self.get_reward = self.reward_reach
        elif self.task==1:
            self.get_reward = self.reward_push_dense
        self.mov_dist = mov_dist
        self.range_x = [-0.3, 0.3]
        self.range_y = [-0.2, 0.4]
        self.z_push = 1.05
        self.z_prepush = self.z_push + self.mov_dist
        self.z_collision_check = self.z_push + 0.025
        self.time_penalty = 0.02 #0.1
        self.max_steps = max_steps
        self.step_count = 0
        self.threshold = 0.1

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

    def init_env(self):
        self.env._init_robot()
        range_x = self.range_x
        range_y = self.range_y
        self.env.sim.data.qpos[12:15] = [0, 0, 0]
        self.env.sim.data.qpos[19:22] = [0, 0, 0]
        self.env.sim.data.qpos[26:29] = [0, 0, 0]
        self.goal1 = [0., 0.]
        self.goal2 = [0., 0.]
        self.goal3 = [0., 0.]
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
            cv2.circle(self.goal_image, self.pos2pixel(*self.goal1), 2, self.colors[0], -1)
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
            cv2.circle(self.goal_image, self.pos2pixel(*self.goal2), 2, self.colors[1], -1)
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
            cv2.circle(self.goal_image, self.pos2pixel(*self.goal3), 2, self.colors[2], -1)

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
        im_state = self.push_from_pixel(px, py, theta)

        reward, done = self.get_reward()

        self.step_count += 1
        if self.step_count==self.max_steps:
            done = True

        if self.task == 0:
            return [im_state], reward, done, None
        else:
            return [im_state, self.goal_image], reward, done, None

    def clip_pos(self, pose):
        x, y = pose
        x = np.max((x, self.range_x[0]))
        x = np.min((x, self.range_x[1]))
        y = np.max((y, self.range_y[0]))
        y = np.min((y, self.range_y[1]))
        return x, y

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
            return im_state
        self.env.move_to_pos([pos_before[0], pos_before[1], self.z_push], grasp=1.0)
        self.env.move_to_pos([pos_after[0], pos_after[1], self.z_push], grasp=1.0)
        self.env.move_to_pos([pos_after[0], pos_after[1], self.z_prepush], grasp=1.0)
        im_state = self.env.move_to_pos(self.init_pos, grasp=1.0)
        return im_state

    def reward_reach(self):
        target_pos = self.env.sim.data.get_body_xpos('target_body_1')
        if np.linalg.norm(target_pos - self.pre_target_pos) > 1e-3:
            reward = 1.0
            done = True
        else:
            reward = -self.time_penalty
            done = False
        return reward, done

    def reward_push_sparse(self):
        done = False
        reward = 0.0
        if self.num_blocks >= 1:
            pos1 = self.env.sim.data.get_body_xpos('target_body_1')[:2]
            if np.linalg.norm(pos1 - self.goal1) < self.threshold:
                reward += 1.0
        if self.num_blocks >= 2:
            pos2 = self.env.sim.data.get_body_xpos('target_body_2')[:2]
            if np.linalg.norm(pos2 - self.goal2) < self.threshold:
                reward += 1.0
        if self.num_blocks >= 3:
            pos3 = self.env.sim.data.get_body_xpos('target_body_3')[:2]
            if np.linalg.norm(pos3 - self.goal3) < self.threshold:
                reward += 1.0

        if reward >= self.num_blocks:
            done = True
        reward += -self.time_penalty
        return reward, done

    def reward_push_dense(self):
        done = False
        reward = 0.0
        if self.num_blocks >= 1:
            pos1 = self.env.sim.data.get_body_xpos('target_body_1')[:2]
            dist1 = np.linalg.norm(pos1 - self.goal1)
            if dist1 < self.threshold:
                reward += 1.0
            else:
                pre_dist1 = np.linalg.norm(self.pre_pos1 - self.goal1)
                reward += pre_dist1 - dist1
        if self.num_blocks >= 2:
            pos2 = self.env.sim.data.get_body_xpos('target_body_2')[:2]
            dist2 = np.linalg.norm(pos2 - self.goal2)
            if dist2 < self.threshold:
                reward += 1.0
            else:
                pre_dist2 = np.linalg.norm(self.pre_pos2 - self.goal2)
                reward += pre_dist2 - dist2
        if self.num_blocks >= 3:
            pos3 = self.env.sim.data.get_body_xpos('target_body_3')[:2]
            dist3 = np.linalg.norm(pos3 - self.goal3)
            if dist3 < self.threshold:
                reward += 1.0
            else:
                pre_dist3 = np.linalg.norm(self.pre_pos3 - self.goal3)
                reward += pre_dist3 - dist3

        if reward >= self.num_blocks:
            done = True
        reward += -self.time_penalty
        return reward, done

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
    env = UR5Env(render=True, camera_height=64, camera_width=64, control_freq=5)
    env = pushpixel_env(env, num_blocks=3, mov_dist=0.05, max_steps=100, task=1)
    frame = env.reset()
    frames = []

    for i in range(100):
        #action = [np.random.randint(6), np.random.randint(2)]
        try:
            action = [np.random.randint(10, 64), np.random.randint(10, 64), np.random.randint(8)]
        except KeyboardInterrupt:
            exit()
        except:
            continue
        if action[2] > 9:
            continue
        print('{} steps. action: {}'.format(env.step_count, action))
        states, reward, done, info = env.step(action)
        #plt.imshow(states[0])
        #plt.show()
        print('Reward: {}. Done: {}'.format(reward, done))
        #show_image(states[0])
        if done:
            print('Done. New episode starts.')
            env.reset()
