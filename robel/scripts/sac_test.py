# Copyright 2019 The ROBEL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to perform rollouts on an environment.

Example usage:
# Visualize an environment:
python -m robel.scripts.rollout -e DClawTurnFixed-v0 --render

# Benchmark offscreen rendering:
python -m robel.scripts.rollout -e DClawTurnFixed-v0 --render rgb_array
"""

import argparse
import collections
import os
import pickle
import time
from typing import Callable, Optional

import gym
import numpy as np

import robel
from robel.scripts.utils import EpisodeLogger, parse_env_args
import torch
from rl_modules.robel_sac import SAC


# The default environment to load when no environment name is given.
DEFAULT_ENV_NAME = 'DClawTurnFixed-v0'

# The default number of episodes to run.
DEFAULT_EPISODE_COUNT = 20

#################################
parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="DClawTurnFixed-v0",
                    help='Mujoco Gym environment')
#是否连接实际夹爪
parser.add_argument('-real',type=bool,default=False, help='run on hardware or not.')

parser.add_argument('-o', '--output', help='The directory to save rollout data to.')
#加载已经训练好的策略
parser.add_argument('-pa', '--policy_actor',
    default='/home/wgk/dataset1/RL/Adversarial_Skill_Learning_for_Robust_Manipulation/train/models/po_sac_actor_DClawTurnFixed-v0_1211023autotune',
    help='The path to the policy actor file to load.')
parser.add_argument('-pc', '--policy_critic',
    default='/home/wgk/dataset1/RL/Adversarial_Skill_Learning_for_Robust_Manipulation/train/models/po_sac_critic_DClawTurnFixed-v0_1211023autotune',
    help='The path to the policy critic file to load.')

#测试的回合数量
parser.add_argument('-n','--num_episodes', type=int,default=DEFAULT_EPISODE_COUNT,
    help='The number of episodes to run.')

parser.add_argument('--seed', type=int, default=121, help='The seed for the environment.')
#是否渲染
parser.add_argument('-r','--render',nargs='?',const='human',default='human',
    help=('The rendering mode. If provided, renders to a window. A render mode string can be passed here.'))


#############SAC相关##############
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')

args = parser.parse_args()

# Named tuple for information stored over a trajectory.
Trajectory = collections.namedtuple('Trajectory', [
    'actions',
    'observations',
    'rewards',
    'total_reward',
    'infos',
    'renders',
    'durations',
])


def do_rollouts(env,
                num_episodes: int, #int类型，且不可缺省
                max_episode_length: Optional[int] = None,   #
                action_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                render_mode: Optional[str] = None):
    """Performs rollouts with the given environment.

    Args:
        num_episodes: 交互的回合数
        max_episode_length: 一条轨迹的最大长度
        action_fn: 用于采样交互动作的函数，如果为None，就在动作空间中随机采样
        render_mode: 是否进行渲染显示，如果是None就不渲染

    Yields:
        Trajectory containing:
            observations: 每个回个的观测值
            rewards: The rewards for the episode. 每个step的reward？
            total_reward: 整个回合的总的reward
            infos: The auxiliary information during the episode. 
            renders: Rendered frames during the episode.
            durations: The running execution durations.
    """
    # If no action function is given, use random actions from the action space.
    if action_fn is None:
        action_fn = lambda _: env.action_space.sample()  

    # Maintain a dictionary of execution durations.
    durations = collections.defaultdict(float)

    # Define a function to maintain a running average of durations.
    def record_duration(key: str, iteration: int, value: float):
        durations[key] = (durations[key] * iteration + value) / (iteration + 1)

    total_steps = 0

    for episode in range(num_episodes):
        episode_start = time.time()
        #初始化环境
        obs = env.reset()
        record_duration('reset', episode, time.time() - episode_start)

        done = False
        #一个回合的动作序列？
        episode_actions = []
        episode_obs = [obs]

        episode_rewards = []
        episode_total_reward = 0

        episode_info = collections.defaultdict(list)
        episode_renders = []

        #开始与环境交互直到回合结束
        while not done:
            step_start = time.time()

            # Get the action for the current observation.
            action = action_fn.select_action(obs, evaluate=True)
            action_time = time.time()
            record_duration('action', total_steps, action_time - step_start)

            # Advance the environment with the action.
            obs, reward, done, info = env.step(action)
            #time.sleep(0.05)
            step_time = time.time()
            record_duration('step', total_steps, step_time - action_time)

            # Render the environment if needed.
            if render_mode is not None:
                render_result = env.render(render_mode)
                record_duration('render', total_steps, time.time() - step_time)
                if render_result is not None:
                    episode_renders.append(render_result)

            # Record episode information.
            #记录当前回合的数据
            episode_actions.append(action) #
            episode_obs.append(obs)
            episode_rewards.append(reward)
            episode_total_reward += reward
            for key, value in info.items():
                episode_info[key].append(value)

            total_steps += 1
            #如果轨迹长度过长
            if (max_episode_length is not None
                    and len(episode_obs) >= max_episode_length):
                done = True

        # Combine the information into a trajectory.
        trajectory = Trajectory(
            actions=np.array(episode_actions),
            observations=np.array(episode_obs),
            rewards=np.array(episode_rewards),
            total_reward=episode_total_reward,
            infos={key: np.array(value) for key, value in episode_info.items()},
            renders=np.array(episode_renders) if episode_renders else None,
            durations=dict(durations),
        )
        yield trajectory


def rollout_script():
    """Performs a rollout script.
    """
    
    #env_id, params, args = parse_env_args(parser, default_env_name=args.env_name)
    #为目标类设置一下目标环境的参数，比如跑的最大回合数
    #robel.set_env_params(env_id, params)

    #构建环境
    if args.real:
        env = gym.make(args.env_name,device_path='/dev/ttyUSB0')
    else:
        env = gym.make(args.env_name)
        
    
    #env = gym.make('DClawTurnFixed-v0', device_path='/dev/ttyUSB0')

    obs_space = env.observation_space
    action_space = env.action_space


    if args.seed is not None:
        env.seed(args.seed)
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)



    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    agent_actor_path = args.policy_actor
    agent_critic_path = args.policy_critic
    agent.load_model(agent_actor_path, agent_critic_path)


    paths = []
    try:
        episode_num = 0
        #在do_rollouts中进行轨迹采样
        for traj in do_rollouts(
                env,#环境
                num_episodes=args.num_episodes,#回合数
                action_fn=agent,# 代理策略
                render_mode=args.render,# 是否渲染结果显示出来
        ):
            print('Episode {}'.format(episode_num))
            print('> Total reward: {}'.format(traj.total_reward))
            if traj.durations:
                print('> Execution times:')
                for key in sorted(traj.durations):
                    print('{}{}: {:.2f}ms'.format(' ' * 4, key,
                                                  traj.durations[key] * 1000))
            episode_num += 1

            if args.output:
                paths.append(
                    dict(
                        actions=traj.actions,
                        observations=traj.observations,
                        rewards=traj.rewards,
                        total_reward=traj.total_reward,
                        infos=traj.infos,
                    ))
    finally:
        env.close()

        if paths and args.output:
            os.makedirs(args.output, exist_ok=True)
            # Serialize the paths.
            save_path = os.path.join(args.output, 'paths.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(paths, f)

            # Log the paths to a CSV file.
            csv_path = os.path.join(args.output,
                                    '{}-results.csv'.format(args.env_name))
            with EpisodeLogger(csv_path) as logger:
                for path in paths:
                    logger.log_path(path)



if __name__ == '__main__':
    #
    rollout_script()
