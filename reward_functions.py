import numpy as np

def reward_reach(self):
    target_pos = self.env.sim.data.get_body_xpos('target_body_1')
    if np.linalg.norm(target_pos - self.pre_target_pos) > 1e-3:
        reward = 1.0
        done = False #True
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
    reward_scale = 10
    min_reward = -2
    done = False
    reward = 0.0
    if self.num_blocks >= 1:
        pos1 = self.env.sim.data.get_body_xpos('target_body_1')[:2]
        dist1 = np.linalg.norm(pos1 - self.goal1)
        if not self.success1:
            if dist1 < self.threshold:
                reward += 1.0
                self.success1 = True
            else:
                pre_dist1 = np.linalg.norm(self.pre_pos1 - self.goal1)
                reward += reward_scale * (pre_dist1 - dist1)
    if self.num_blocks >= 2:
        pos2 = self.env.sim.data.get_body_xpos('target_body_2')[:2]
        dist2 = np.linalg.norm(pos2 - self.goal2)
        if not self.success2:
            if dist2 < self.threshold:
                reward += 1.0
                self.success2 = True
            else:
                pre_dist2 = np.linalg.norm(self.pre_pos2 - self.goal2)
                reward += reward_scale * (pre_dist2 - dist2)
    if self.num_blocks >= 3:
        pos3 = self.env.sim.data.get_body_xpos('target_body_3')[:2]
        dist3 = np.linalg.norm(pos3 - self.goal3)
        if not self.success3:
            if dist3 < self.threshold:
                reward += 1.0
                self.success3 = True
            else:
                pre_dist3 = np.linalg.norm(self.pre_pos3 - self.goal3)
                reward += reward_scale * (pre_dist3 - dist3)

    if np.sum([self.success1, self.success2, self.success3]) >= self.num_blocks:
        done = True
    reward += -self.time_penalty
    reward = max(reward, min_reward)
    return reward, done

def reward_push_reverse(self):
    reward_scale = 0.5
    min_reward = -2
    done = False
    reward = 0.0
    if self.num_blocks >= 1:
        pos1 = self.env.sim.data.get_body_xpos('object_1')[:2]
        dist1 = np.linalg.norm(pos1 - self.goal1)
        if not self.success1:
            if dist1 < self.threshold:
                self.success1 = True
            pre_dist1 = np.linalg.norm(self.pre_pos1 - self.goal1)
            if dist1 < pre_dist1 - 0.001:
                reward += 1
            reward += reward_scale * min(10, (1/dist1 - 1/pre_dist1))
    if self.num_blocks >= 2:
        pos2 = self.env.sim.data.get_body_xpos('object_2')[:2]
        dist2 = np.linalg.norm(pos2 - self.goal2)
        if not self.success2:
            if dist2 < self.threshold:
                self.success2 = True
            pre_dist2 = np.linalg.norm(self.pre_pos2 - self.goal2)
            if dist2 < pre_dist2 - 0.001:
                reward += 1
            reward += reward_scale * min(10, (1/dist2 - 1/pre_dist2))
    if self.num_blocks >= 3:
        pos3 = self.env.sim.data.get_body_xpos('object_3')[:2]
        dist3 = np.linalg.norm(pos3 - self.goal3)
        if not self.success3:
            if dist3 < self.threshold:
                self.success3 = True
            pre_dist3 = np.linalg.norm(self.pre_pos3 - self.goal3)
            if dist3 < pre_dist3 - 0.001:
                reward += 1
            reward += reward_scale * min(10, (1/dist3 - 1/pre_dist3))

    if np.sum([self.success1, self.success2, self.success3]) >= self.num_blocks:
        done = True
    reward += -self.time_penalty
    reward = max(reward, min_reward)
    return reward, done
