import time

import gym
import mujoco
import mujoco.viewer
import numpy as np

import franka_sim

# import joystick wrapper
from franka_env.envs.wrappers import JoystickIntervention

env = gym.make("PandaPickCubeVision-v0", render_mode="human", image_obs=True)
env = JoystickIntervention(env)

env.reset()

# intervene on position control
for i in range(100000):
    env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
