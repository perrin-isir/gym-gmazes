# Copyright 2022 Nicolas Perrin-Gilbert.
#
# Licensed under the BSD 3-Clause License.
from typing import Optional
import gym
from typing import Union
from gym import spaces
from gym import error
import numpy as np
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


def intersect(a, b, c, d):
    x1, x2, x3, x4 = a[:, 0], b[:, 0], c[0], d[0]
    y1, y2, y3, y4 = a[:, 1], b[:, 1], c[1], d[1]
    denom = (x4 - x3) * (y1 - y2) - (x1 - x2) * (y4 - y3)

    t = np.divide(
        ((y3 - y4) * (x1 - x3) + (x4 - x3) * (y1 - y3)),
        denom,
        out=np.zeros_like(denom),
        where=denom != 0,
    )
    criterion1 = np.logical_and(t > 0, t < 1)
    t = np.divide(
        ((y1 - y2) * (x1 - x3) + (x2 - x1) * (y1 - y3)),
        denom,
        out=np.zeros_like(denom),
        where=denom != 0,
    )
    criterion2 = np.logical_and(t > 0, t < 1)

    return np.logical_and(criterion1, criterion2)


class GMazeCommon:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.compute_reward = None
        self.frame_skip = 2
        # initial position + orientation
        self.init_qpos = np.tile(np.array([-1.0, 0.0, 0.0]), (self.num_envs, 1))
        self.reset_states = None
        self.reset_steps = None
        self.steps = None
        self.done = np.zeros((num_envs, 1), dtype="int")
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
        self.init_qpos = np.tile(np.array(qpos), (self.num_envs, 1))

    def set_reset_states(
        self, reset_states: list, reset_steps: Union[list, None] = None
    ):
        self.reset_states = np.array(reset_states)
        if reset_steps is None:
            self.reset_steps = np.zeros((len(reset_states),), dtype=int)
        else:
            self.reset_steps = np.array(reset_steps, dtype=int)

    def reset_done(self, done, *, options=None, seed: Optional[int] = None, infos=None):
        pass

    def reset_model(self):
        # reset state to initial value
        if self.reset_states is None:
            self.state = self.init_qpos
            self.steps = np.zeros(self.num_envs, dtype=int)
            return {}
        else:
            indices = np.random.choice(len(self.reset_states), self.num_envs)
            self.state = self.reset_states[indices]
            self.steps = self.reset_steps[indices]
            return {"reset_states": indices}

    def common_reset(self):
        return self.reset_model()  # reset state to initial value

    def common_reset_done(self, done):
        # done = self.done
        if not isinstance(done, np.ndarray):
            done = np.asarray(done)
        if self.reset_states is None:
            self.state = np.where(done == 1, self.init_qpos, self.state)
            zeros = np.zeros(self.num_envs, dtype=int)
            self.steps = np.where(done.flatten() == 1, zeros, self.steps)
            return {}
        else:
            indices = np.random.choice(len(self.reset_states), self.num_envs)
            r_state = self.reset_states[indices]
            r_steps = self.reset_steps[indices]
            self.state = np.where(done == 1, r_state, self.state)
            self.steps = np.where(done.flatten() == 1, r_steps, self.steps)
            return {"reset_states": indices}

    def common_step(self, action):
        # add action to the state frame_skip times,
        # checking -1 & +1 boundaries and intersections with walls
        if not isinstance(action, np.ndarray):
            action = np.asarray(action)
        for k in range(self.frame_skip):
            cosval = np.cos(np.pi * self.state[:, 2])
            sinval = np.sin(np.pi * self.state[:, 2])
            ns_01 = self.state[:, :2] + 1.0 / 20.0 * np.stack((cosval, sinval), axis=1)
            ns_01 = ns_01.clip(-1.0, 1.0)
            ns_2 = self.state[:, 2] + action[:, 0] / 10.0
            ns_2 = (ns_2 + 1.0) % 2.0 - 1.0
            new_state = np.hstack((ns_01, np.expand_dims(ns_2, axis=1)))

            intersection = np.full((self.num_envs,), False)
            for (w1, w2) in self.walls:
                intersection = np.logical_or(
                    intersection, intersect(self.state, new_state, w1, w2)
                )
            intersection = np.expand_dims(intersection, axis=-1)
            self.state = self.state * intersection + new_state * np.logical_not(
                intersection
            )
        self.steps += 1
        truncation = np.asarray(
            (self.steps == self.max_episode_steps), dtype=float
        ).reshape((self.num_envs, 1))
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

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray):
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


def default_reward_fun(action, new_obs):
    reward = 1.0 * (np.logical_and(new_obs[:, 0] > 0.5, new_obs[:, 1] > 0.5))
    return np.expand_dims(reward, axis=-1)


class GMazeDubins(GMazeCommon, gym.Env):
    def __init__(self, num_envs: int = 1):
        super().__init__(num_envs)

        self.set_reward_function(default_reward_fun)

        high = np.ones(self._obs_dim)
        low = -high
        self.single_observation_space = spaces.Box(low, high, dtype=np.float64)
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )

    def step(self, action: np.ndarray):
        action, truncation = self.common_step(action)

        observation = self.state
        reward = self.compute_reward(action, observation).reshape((self.num_envs, 1))
        self.done = truncation
        info = {"truncation": truncation}
        return (
            observation,
            reward,
            self.done,
            info,
        )

    def reset(
        self, seed: Optional[int] = None, return_info: bool = False, options=None
    ):
        info = self.common_reset()
        if return_info:
            return self.state, info
        else:
            return self.state

    def reset_done(
        self,
        done,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options=None
    ):
        info = self.common_reset_done(done)
        if return_info:
            return self.state, info
        else:
            return self.state


def achieved_g(state):
    return state[:, :2]


def goal_distance(goal_a, goal_b):
    return np.linalg.norm(goal_a[:, :2] - goal_b[:, :2], axis=-1)


def default_compute_reward(
    achieved_goal: np.ndarray,
    desired_goal: np.ndarray,
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


def default_success_function(achieved_goal: np.ndarray, desired_goal: np.ndarray):
    distance_threshold = 0.1
    d = goal_distance(achieved_goal, desired_goal)
    return 1.0 * (d < distance_threshold)


class GMazeGoalDubins(GMazeCommon, GoalEnv):
    def __init__(self, num_envs: int = 1):
        super().__init__(num_envs)

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

    def set_success_function(self, success_function):
        self._is_success = success_function

    def _sample_goal(self):
        return achieved_g(np.random.rand(self.num_envs, 2) * 2.0 - 1)

    def set_goal(self, goal):
        if not isinstance(goal, np.ndarray):
            goal = np.asarray(goal)
        self.goal = goal

    def reset(
        self, seed: Optional[int] = None, return_info: bool = False, options=None
    ):
        info = self.common_reset()
        self.set_goal(self._sample_goal())  # sample goal
        res = {
            "observation": self.state,
            "achieved_goal": achieved_g(self.state),
            "desired_goal": self.goal,
        }
        if return_info:
            return res, info
        else:
            return res

    def reset_done(
        self,
        done,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options=None
    ):
        if not isinstance(done, np.ndarray):
            done = np.asarray(done)
        info = self.common_reset_done(done)
        newgoal = self._sample_goal()
        self.set_goal(np.where(done == 1, newgoal, self.goal))
        res = {
            "observation": self.state,
            "achieved_goal": achieved_g(self.state),
            "desired_goal": self.goal,
        }
        if return_info:
            return res, info
        else:
            return res

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
            "is_success": is_success,
            "truncation": truncation,
        }
        self.done = np.maximum(truncation, is_success)

        return (
            {
                "observation": self.state,
                "achieved_goal": achieved_g(self.state),
                "desired_goal": self.goal,
            },
            reward,
            self.done,
            info,
        )
