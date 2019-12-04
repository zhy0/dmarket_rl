import pytest
import numpy as np
from dmarket.environments import SingleAgentTrainingEnv, MultiAgentTrainingEnv
from dmarket.info_settings import BlackBoxSetting, TimeInformationWrapper
from dmarket.agents import ConstantAgent, GymRLAgent, TimeLinearAgent


def test_single_env():
    rl_agent = GymRLAgent('buyer', 100)
    fixed_agents = [
        ConstantAgent('seller', 80),
        ConstantAgent('seller', 90),
        ConstantAgent('seller', 95),
    ]
    env = SingleAgentTrainingEnv(rl_agent, fixed_agents, BlackBoxSetting())

    # It should have well defined observation space
    obs = env.reset()
    assert env.observation_space.contains(obs)

    # The step function and action_space sampler should work
    obs, reward, done, _ = env.step(env.action_space.sample())

    # It should give positive reward and be done if agent matches
    env.reset()
    obs, reward, done, _ = env.step(4) # buy at 90
    assert obs == np.array([0.1]) # should be normalized
    assert reward == 15
    assert done


def test_multi_env():
    rl_agents = [
        GymRLAgent('buyer',  110,  'A'),
        GymRLAgent('seller', 90,   'B'),
    ]
    fixed_agents = [
        ConstantAgent('buyer',   1, 'B1'),
        ConstantAgent('buyer', 105, 'B105'),
        ConstantAgent('seller', 95, 'S95'),
    ]
    env = MultiAgentTrainingEnv(rl_agents, fixed_agents, BlackBoxSetting())

    # It should have well defined observation space
    obs = env.reset()
    assert env.observation_space.contains(obs['A'])
    assert env.observation_space.contains(obs['B'])

    # The step function and action_space sampler should work
    obs, rew, done, _ = env.step({'A': 20, 'B': 1})
    assert obs == {'A': np.array([0.5]), 'B': np.array([0.025])}
    assert rew == {'A': 0, 'B': 8.625}
    assert done == {'A': False, 'B': True, '__all__': False}

    obs, rew, done, _ = env.step({'A': 0})
    assert obs == {'A': np.array([0.])}
    assert rew == {'A': 7.5}
    assert done == {'A': True, '__all__': True}


def test_multi_time_env():
    rl_agents = [
        GymRLAgent('buyer', 130, 'A'),
        GymRLAgent('seller', 70, 'B'),
    ]
    fixed_agents = [
        TimeLinearAgent('buyer',  100, max_steps=2, noise=0),
        TimeLinearAgent('seller', 100, max_steps=2, noise=0),
    ]
    env = MultiAgentTrainingEnv(rl_agents, fixed_agents,
                                TimeInformationWrapper(BlackBoxSetting()))

    # It should have well-defined observation space
    obs = env.reset()
    assert env.observation_space.contains(obs['A'])
    assert env.observation_space.contains(obs['B'])

    # Seller offers 150, buyer offers 50
    obs,  rew, done, _ = env.step({'A': 20, 'B': 20})
    assert obs == {'A': np.array([0.5]), 'B': np.array([0.5])}
    assert rew == {'A': 0, 'B': 0}
    assert done == {'A': False, 'B': False, '__all__': False}

    # Seller offers 125, buyer offers 75
    obs,  rew, done, _ = env.step({'A': 20, 'B': 0})
    assert obs == {'A': np.array([0.5]), 'B': np.array([0])}
    assert rew == {'A': 0, 'B': 2.5}
    assert done == {'A': False, 'B': True, '__all__': False}

    # Seller offers 100
    obs,  rew, done, _ = env.step({'A': 0})
    print(env.market.offer_history[-1])
    assert obs == {'A': np.array([0.])}
    assert rew == {'A': 15}
    assert done == {'A': True, '__all__': True}
