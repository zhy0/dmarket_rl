import pytest
from dmarket.environments import SingleAgentTrainingEnv
from dmarket.info_settings import BlackBoxSetting
from dmarket.agents import ConstantAgent, GymRLAgent

@pytest.fixture()
def single_agent_env():
    def create_env(rl_role, rl_res, buyer_res_prices, seller_res_prices):
        """Create a SingleAgentTrainingEnv with ConstantAgents"""
        rl_agent = GymRLAgent(rl_role, rl_res)
        buyers = [
            ConstantAgent('buyer', price)
            for price in buyer_res_prices
        ]
        sellers = [
            ConstantAgent('seller', price)
            for price in seller_res_prices
        ]
        setting = BlackBoxSetting()
        return SingleAgentTrainingEnv(rl_agent, buyers + sellers, setting)
    return create_env


def test_single_env(single_agent_env):
    env = single_agent_env('buyer', 100, [], [80, 90, 95])
    # It should have well defined observation space
    obs = env.reset()
    assert env.observation_space.contains(obs)

    # The step function and action_space sampler should work
    obs, reward, done, _ = env.step(env.action_space.sample())

    # It should give positive reward and be done if agent matches
    env = single_agent_env('buyer', 100, [], [80]) # Will always sell at 80
    obs, reward, done, _ = env.step(0)
    assert reward > 0
    assert done

    # Same if agent is a seller
    env = single_agent_env('seller', 100, [110], []) # Will always buy at 110
    obs, reward, done, _ = env.step(0)
    assert reward > 0
    assert done
