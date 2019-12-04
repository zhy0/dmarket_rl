import numpy as np
import gym
from gym.spaces import Discrete, Box
from dmarket.engine import MarketEngine
from dmarket.info_settings import TimeInformationWrapper


class MultiAgentTrainingEnv(gym.Env):
    """
    OpenAI Gym environment for training multiple agents simultaneously.

    This environment is designed for training multiple agents at the same time.
    The environment does the conversion from discrete action space to prices
    based on reservation prices. Therefore, it might not be suitable when other
    non-RL agents need to be in the environment (unless these also have
    discrete action spaces). This environment is compatible with RLlib's
    ``MultiAgentEnv``.

    Parameters
    ----------
    rl_agents: list
        A list of RL agent objects, must be instances of ``GymRLAgent``.
    fixed_agents: list
        A list of fixed agents. All instances must implement the ``get_offer``
        function.
    setting: InformationSetting object
        The information setting of the market environment.
    discretization: int, optional (default=20)
        The number of different offers the agents can make. This determines the
        action space of the agents.
    max_factor: float, optional (default=0.5)
        A factor of the reservation price that determines the range of prices
        the agents can offer.
    max_steps: int, optional (default=30)
        Maximum number of rounds before a single market game terminates. This
        is passed on to the market engine.

    Attributes
    ----------
    observation_space: Box object
        Derived from ``setting`` parameter.
    market: MarketEngine object
        The underlying market engine object.
    """
    def __init__(self, rl_agents, fixed_agents, setting, max_factor=0.5,
                 max_steps=30):

        self.rl_agents = {
            rl_agent.name: rl_agent for rl_agent in rl_agents
        }
        self.fixed_agents = {
            fixed_agent.name: fixed_agent for fixed_agent in fixed_agents
        }
        self.all_agents = {}
        self.all_agents.update(self.rl_agents)
        self.all_agents.update(self.fixed_agents)

        if isinstance(setting, TimeInformationWrapper):
            self.observation_space = setting.base_setting.observation_space
            self.rl_setting = setting.base_setting
        else:
            self.observation_space = setting.observation_space
            self.rl_setting = setting

        self.setting = setting
        self.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=self.rl_setting.observation_space.shape
        )

        buyer_ids =  [
            agent.name
            for agent in self.all_agents.values()
            if agent.role == 'buyer'
        ]
        seller_ids =  [
            agent.name
            for agent in self.all_agents.values()
            if agent.role == 'seller'
        ]
        self.market = MarketEngine(buyer_ids, seller_ids, max_steps)


    def _get_rewards(self, agent_ids, deals):
        """
        Compute the reward of each agent given the deals in the last round.
        """
        result = {}
        for agent_id in agent_ids:
            if agent_id in deals:
                agent = self.all_agents[agent_id]
                sign = (-1 if agent.role == 'buyer' else +1)
                res_price = agent.reservation_price
                result[agent_id] = (deals[agent_id] - res_price)*sign
            else:
                result[agent_id] = 0
        return result


    def reset(self):
        """
        Reset the training market environment.

        Returns
        -------
        observations: dict
            Initial observations for all agents.
        """
        self.market.reset()
        return self.rl_setting.get_states(self.rl_agents.keys(), self.market)


    def step(self, actions):
        """
        Parameters
        ----------
        actions: dict
            A dictionary of actions per agent, indexed by agent id.

        Returns
        -------
        observations: dict
            Observations per agent. Will only contain those agents found in the
            passed actions dict.
        rewards: dict
            Reward per agent. Will only contain those agents found in the
            passed actions dict.
        done: dict
            Dict of True/False to indicate whether each agent is done or not.
            Will only contain those agents found in the passed actions dict.
            Contains additionally a string key ``__all__`` to indicate
            whether every agent is done.
        """
        # Get dict of fixed agents that aren't yet done
        other_agents = {
            agent.name: agent for agent in self.fixed_agents.values()
            if agent.name not in self.market.done
        }

        # First get offers of other fixed agents
        obs = self.setting.get_states(other_agents.keys(), self.market)
        offers = {
            agent.name: agent.get_offer(obs[agent.name])
            for agent in other_agents.values()
        }

        # Update offers with RL offers from the actions dict
        rl_agent_ids = set(actions.keys())
        for rl_agent_id in rl_agent_ids:
            offers[rl_agent_id] = self.rl_agents[rl_agent_id].action_to_price(
                actions[rl_agent_id]
            )

        # Step the market
        deals = self.market.step(offers)

        # Obs, done, rewards for RL agents
        obs = self.rl_setting.get_states(rl_agent_ids, self.market)
        for rl_agent_id in obs.keys(): # Normalize each observation
            obs[rl_agent_id] = self.rl_agents[rl_agent_id].normalize(
                obs[rl_agent_id]
            )
        done = {
            rl_agent_id: (rl_agent_id in self.market.done)
            for rl_agent_id in rl_agent_ids
        }
        done["__all__"] = (rl_agent_ids.issubset(self.market.done))
        rew = self._get_rewards(rl_agent_ids, deals)
        return obs, rew, done, {}


class SingleAgentTrainingEnv(MultiAgentTrainingEnv):
    def __init__(self, rl_agent, fixed_agents, setting, max_steps=30):
        self.rl_agent = rl_agent
        self.action_space = Discrete(rl_agent.discretization)
        super().__init__([rl_agent], fixed_agents, setting, max_steps)

    def reset(self):
        return super().reset()[self.rl_agent.name]

    def step(self, action):
        obs, rew, done, _ = super().step({
            self.rl_agent.name: action
        })
        return obs[self.rl_agent.name], \
               rew[self.rl_agent.name], \
              done[self.rl_agent.name], _
