# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.

from abc import ABC
from abc import abstractmethod
from typing import Optional
import gym
from typing import Union
from gym import utils, spaces
from gym import error
import numpy as np
import torch
from matplotlib import collections as mc


class GoalEnv(gym.Env):
    """The GoalEnv class that was migrated from gym (v0.22) to gym-robotics."""

    def reset(
        self, seed: Optional[int] = None, return_info: bool = False, options=None
    ):
        super().reset(seed=seed)
        # Enforce that each GoalEnv uses a Goal-compatible observation space.
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise error.Error(
                "GoalEnv requires an observation space of type gym.spaces.Dict"
            )
        for key in ["observation", "achieved_goal", "desired_goal"]:
            if key not in self.observation_space.spaces:
                raise error.Error('GoalEnv requires the "{}" key.'.format(key))

    @abstractmethod
    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the step reward.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to
            the desired goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'],
                                                    ob['desired_goal'], info)
        """
        raise NotImplementedError


@torch.no_grad()
def intersect(a, b, c, d):
    x1, x2, x3, x4 = a[:, 0], b[:, 0], c[0], d[0]
    y1, y2, y3, y4 = a[:, 1], b[:, 1], c[1], d[1]
    denom = (x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3)

    criterion1 = denom != 0
    t = ((y3 - y4) * (x1 - x3) + (x4 - x3) * (y1 - y3)) / denom
    criterion2 = torch.logical_and(t > 0, t < 1)
    t = ((y1 - y2) * (x1 - x3) + (x2 - x1) * (y1 - y3)) / denom
    criterion3 = torch.logical_and(t > 0, t < 1)

    return torch.logical_and(torch.logical_and(criterion1, criterion2), criterion3)


class GMazeCommon:
    def __init__(self, device: str, num_envs: int):
        self.num_envs = num_envs
        self.device = device
        utils.EzPickle.__init__(**locals())
        self.compute_reward = None
        self.frame_skip = 2
        # initial position + orientation
        self.init_qpos = torch.tensor(
            np.tile(np.array([-1.0, 0.0, 0.0]), (self.num_envs, 1))
        ).to(self.device)
        self.reset_states = None
        self.reset_steps = None
        self.steps = None
        self.done = torch.zeros((num_envs, 1), dtype=torch.int).to(self.device)
        self.init_qvel = None  # velocities are not used
        self.state = self.init_qpos
        self.walls = []
        self._obs_dim = 3
        self._action_dim = 1
        high = np.ones(self._action_dim)
        low = -high
        self.single_action_space = spaces.Box(low=low, high=high, dtype=np.float64)
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, self.num_envs
        )
        self.max_episode_steps = 70

    def set_init_qpos(self, qpos):
        self.init_qpos = torch.tensor(np.tile(np.array(qpos), (self.num_envs, 1))).to(
            self.device
        )

    def set_reset_states(
        self, reset_states: list, reset_steps: Union[list, None] = None
    ):
        self.reset_states = np.array(reset_states)
        if reset_steps is None:
            self.reset_steps = np.zeros((len(reset_states),), dtype=int)
        else:
            self.reset_steps = np.array(reset_steps)

    @abstractmethod
    def reset_done(self, options=None, seed: Optional[int] = None, infos=None):
        pass

    @torch.no_grad()
    def reset_model(self):
        # reset state to initial value
        if self.reset_states is None:
            self.state = self.init_qpos
            self.steps = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
            return {}
        else:
            indices = np.random.choice(len(self.reset_states), self.num_envs)
            self.state = torch.tensor(self.reset_states[indices]).to(self.device)
            self.steps = torch.tensor(self.reset_steps[indices], dtype=torch.int).to(
                self.device
            )
            return {"reset_states": indices}

    @torch.no_grad()
    def common_reset(self):
        return self.reset_model()  # reset state to initial value

    @torch.no_grad()
    def common_reset_done(self):
        if self.reset_states is None:
            self.state = torch.where(self.done == 1, self.init_qpos, self.state)
            zeros = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
            self.steps = torch.where(self.done.flatten() == 1, zeros, self.steps)
            return {}
        else:
            indices = np.random.choice(len(self.reset_states), self.num_envs)
            r_state = torch.tensor(self.reset_states[indices]).to(self.device)
            r_steps = torch.tensor(self.reset_steps[indices], dtype=torch.int).to(
                self.device
            )
            self.state = torch.where(self.done == 1, r_state, self.state)
            self.steps = torch.where(self.done.flatten() == 1, r_steps, self.steps)
            return {"reset_states": indices}

    @torch.no_grad()
    def common_step(self, action: Union[np.ndarray, torch.Tensor]):
        # add action to the state frame_skip times,
        # checking -1 & +1 boundaries and intersections with walls
        if torch.is_tensor(action):
            action = action.to(self.device)
        else:
            action = torch.tensor(action).to(self.device)
        for k in range(self.frame_skip):
            cosval = torch.cos(torch.pi * self.state[:, 2])
            sinval = torch.sin(torch.pi * self.state[:, 2])
            ns_01 = self.state[:, :2] + 1.0 / 20.0 * torch.stack(
                (cosval, sinval), dim=1
            ).to(self.device)
            ns_01 = ns_01.clip(-1.0, 1.0)
            ns_2 = self.state[:, 2] + action[:, 0] / 10.0
            ns_2 = (ns_2 + 1.0) % 2.0 - 1.0
            new_state = torch.hstack((ns_01, ns_2.unsqueeze(dim=1)))

            intersection = torch.full((self.num_envs,), False).to(self.device)
            for (w1, w2) in self.walls:
                intersection = torch.logical_or(
                    intersection, intersect(self.state, new_state, w1, w2)
                )
            intersection = torch.unsqueeze(intersection, dim=-1)
            self.state = self.state * intersection + new_state * torch.logical_not(
                intersection
            )
        self.steps += 1
        truncation = (
            (self.steps == self.max_episode_steps).double().reshape((self.num_envs, 1))
        )
        return action, truncation

    def set_reward_function(self, reward_function):
        self.compute_reward = (
            reward_function  # the reward function is not defined by the environment
        )

    def set_frame_skip(self, frame_skip: int = 2):
        self.frame_skip = (
            frame_skip  # a call to step() repeats the action frame_skip times
        )

    def set_walls(self, walls=None):
        if walls is None:
            self.walls = [
                ([0.5, -0.5], [0.5, 1.01]),
                ([-0.5, -0.5], [-0.5, 1.01]),
                ([0.0, -1.01], [0.0, 0.5]),
            ]
        else:
            self.walls = walls

    @torch.no_grad()
    def set_state(self, qpos: torch.Tensor, qvel: torch.Tensor = None):
        self.state = qpos

    def plot(self, ax):
        lines = []
        rgbs = []
        for w in self.walls:
            lines.append(w)
            rgbs.append((0, 0, 0, 1))
        ax.add_collection(mc.LineCollection(lines, colors=rgbs, linewidths=2))
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])


@torch.no_grad()
def default_reward_fun(action, new_obs):
    reward = 1.0 * (torch.logical_and(new_obs[:, 0] > 0.5, new_obs[:, 1] > 0.5))
    return torch.unsqueeze(reward, dim=-1)


class GMazeDubins(GMazeCommon, gym.Env, utils.EzPickle, ABC):
    def __init__(self, device: str = "cpu", num_envs: int = 1):
        super().__init__(device, num_envs)

        self.set_reward_function(default_reward_fun)

        high = np.ones(self._obs_dim)
        low = -high
        self.single_observation_space = spaces.Box(low, high, dtype=np.float64)
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )

    @torch.no_grad()
    def step(self, action: np.ndarray):
        action, truncation = self.common_step(action)

        observation = self.state
        reward = self.compute_reward(action, observation).reshape((self.num_envs, 1))
        self.done = truncation
        info = {"truncation": truncation.detach().cpu().numpy()}
        return (
            observation.detach().cpu().numpy(),
            reward.detach().cpu().numpy(),
            self.done.detach().cpu().numpy(),
            info,
        )

    @torch.no_grad()
    def reset(
        self, seed: Optional[int] = None, return_info: bool = False, options=None
    ):
        info = self.common_reset()
        if return_info:
            return self.state.detach().cpu().numpy(), info
        else:
            return self.state.detach().cpu().numpy()

    @torch.no_grad()
    def reset_done(
        self, seed: Optional[int] = None, return_info: bool = False, options=None
    ):
        info = self.common_reset_done()
        if return_info:
            return self.state.detach().cpu().numpy(), info
        else:
            return self.state.detach().cpu().numpy()


@torch.no_grad()
def achieved_g(state):
    return state[:, :2]


@torch.no_grad()
def goal_distance(goal_a, goal_b):
    if torch.is_tensor(goal_a):
        return torch.linalg.norm(goal_a[:, :2] - goal_b[:, :2], axis=-1)
    else:
        return np.linalg.norm(goal_a[:, :2] - goal_b[:, :2], axis=-1)


@torch.no_grad()
def default_compute_reward(
    achieved_goal: Union[np.ndarray, torch.Tensor],
    desired_goal: Union[np.ndarray, torch.Tensor],
    info: dict,
):
    distance_threshold = 0.1
    # reward_type = "sparse"
    reward_type = "dense"
    d = goal_distance(achieved_goal, desired_goal)
    if reward_type == "sparse":
        return -1.0 * (d > distance_threshold)
    else:
        return -d


@torch.no_grad()
def default_success_function(achieved_goal: torch.Tensor, desired_goal: torch.Tensor):
    distance_threshold = 0.1
    d = goal_distance(achieved_goal, desired_goal)
    return 1.0 * (d < distance_threshold)


class GMazeGoalDubins(GMazeCommon, GoalEnv, utils.EzPickle, ABC):
    def __init__(self, device: str = "cpu", num_envs: int = 1):
        super().__init__(device, num_envs)

        high = np.ones(self._obs_dim)
        low = -high
        self._achieved_goal_dim = 2
        self._desired_goal_dim = 2
        high_achieved_goal = np.ones(self._achieved_goal_dim)
        low_achieved_goal = -high_achieved_goal
        high_desired_goal = np.ones(self._desired_goal_dim)
        low_desired_goal = -high_desired_goal
        self.single_observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(low, high, dtype=np.float64),
                achieved_goal=spaces.Box(
                    low_achieved_goal, high_achieved_goal, dtype=np.float64
                ),
                desired_goal=spaces.Box(
                    low_desired_goal, high_desired_goal, dtype=np.float64
                ),
            )
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )
        self.goal = None

        self.compute_reward = None
        self.set_reward_function(default_compute_reward)

        self._is_success = None
        self.set_success_function(default_success_function)

    @torch.no_grad()
    def set_success_function(self, success_function):
        self._is_success = success_function

    @torch.no_grad()
    def _sample_goal(self):
        return achieved_g(torch.rand(self.num_envs, 2) * 2.0 - 1).to(self.device)

    @torch.no_grad()
    def set_goal(self, goal: Union[np.ndarray, torch.Tensor]):
        if torch.is_tensor(goal):
            self.goal = goal.to(self.device)
        else:
            self.goal = torch.tensor(goal).to(self.device)

    @torch.no_grad()
    def reset(
        self, seed: Optional[int] = None, return_info: bool = False, options=None
    ):
        info = self.common_reset()
        self.set_goal(self._sample_goal())  # sample goal
        res = {
            "observation": self.state.detach().cpu().numpy(),
            "achieved_goal": achieved_g(self.state).detach().cpu().numpy(),
            "desired_goal": self.goal.detach().cpu().numpy(),
        }
        if return_info:
            return res, info
        else:
            return res

    @torch.no_grad()
    def reset_done(
        self, seed: Optional[int] = None, return_info: bool = False, options=None
    ):
        info = self.common_reset_done()
        newgoal = self._sample_goal()
        self.set_goal(torch.where(self.done == 1, newgoal, self.goal))
        res = {
            "observation": self.state.detach().cpu().numpy(),
            "achieved_goal": achieved_g(self.state).detach().cpu().numpy(),
            "desired_goal": self.goal.detach().cpu().numpy(),
        }
        if return_info:
            return res, info
        else:
            return res

    @torch.no_grad()
    def step(self, action: np.ndarray):
        _, truncation = self.common_step(action)

        reward = self.compute_reward(achieved_g(self.state), self.goal, {}).reshape(
            (self.num_envs, 1)
        )
        is_success = self._is_success(achieved_g(self.state), self.goal).reshape(
            (self.num_envs, 1)
        )
        truncation = truncation * (1 - is_success)
        info = {
            "is_success": is_success.detach().cpu().numpy(),
            "truncation": truncation.detach().cpu().numpy(),
        }
        self.done = torch.maximum(truncation, is_success)

        return (
            {
                "observation": self.state.detach().cpu().numpy(),
                "achieved_goal": achieved_g(self.state).detach().cpu().numpy(),
                "desired_goal": self.goal.detach().cpu().numpy(),
            },
            reward.detach().cpu().numpy(),
            self.done.detach().cpu().numpy(),
            info,
        )
