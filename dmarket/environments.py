import numpy as np
import gym
from gym.spaces import Discrete, Box
from dmarket.engine import MarketEngine


class SingleAgentTrainingEnv(gym.Env):
    """
    OpenAI Gym environment for single agent training.

    Parameters
    ----------
    rl_agent: GymRLAgent object
        The settings of the RL agent to train for. This determines the action
        space.

    fixed_agents: list
        A list of MarketAgent objects that have implemented the ``get_offer``
        method. These will serve as other agents in the market.

    setting: InformationSetting object
        This determines the observation space of the agent.

    max_steps: int
        Maximum number of rounds before a single market game terminates. This
        is passed on to the market engine.

    Attributes
    ----------
    action_space: Discrete object
        The actions the RL agent can take.

    observation_space: Box object
        The space in which obervations are elements of.

    market: MarketEngine object
        The market object used

    """
    def __init__(self, rl_agent, fixed_agents, setting, max_steps=30):

        self.rl_agent = rl_agent
        self.fixed_agents = fixed_agents
        self.fixed_agent_ids = [id(agent) for agent in fixed_agents]
        self.setting = setting
        self.action_space = Discrete(rl_agent.discretization)
        self.observation_space = setting.observation_space

        buyer_ids =  [
            id(agent)
            for agent in fixed_agents + [rl_agent]
            if agent.role == 'buyer'
        ]
        seller_ids =  [
            id(agent)
            for agent in fixed_agents + [rl_agent]
            if agent.role == 'seller'
        ]

        self.market = MarketEngine(buyer_ids, seller_ids, max_steps)

    def reset(self):
        """Resets the current environment."""
        self.market.reset()
        return self.setting.get_state(id(self.rl_agent), self.market)

    def get_offers(self, agents, observations):
        """Compute offers for a list of agents given their observations."""
        offers = {
            id(agent): agent.get_offer(observations[id(agent)])
            for agent in agents
        }
        return offers


    def get_reward(self, agent, deals):
        """
        Compute the reward for the RL agent given the set of deals.

        The reward is simply the difference between the deal price and the
        agents reservation price.
        """
        if not id(agent) in deals:
            return 0

        deal_price = deals[id(agent)]
        if agent.role == 'buyer':
            return agent.reservation_price - deal_price
        else:
            return deal_price - agent.reservation_price


    def step(self, action):
        """Compute the next state of the market."""
        observations = self.setting.get_states(self.fixed_agent_ids, self.market)
        # Get all fixed agents that haven't been matched yet
        agents = [
            agent for agent in self.fixed_agents
            if id(agent) not in self.market.done
        ]
        offers = self.get_offers(agents, observations)
        offers[id(self.rl_agent)] = self.rl_agent.action_to_price(action)

        deals = self.market.step(offers)
        observation = self.setting.get_state(id(self.rl_agent), self.market)
        reward = self.get_reward(self.rl_agent, deals)
        done = (id(self.rl_agent) in self.market.done)

        return observation, reward, done, {}


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
    agents: list
        A list of tuples with agent id, role, and reservation price.
        The role must either be the string 'buyer' or 'seller'.
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
    action_space: Discrete object
        Discrete action space derived from ``discretization``.
    market: MarketEngine object
        The underlying market engine object.
    agents: dict
        Internal dictionary of agent data indexed by agent id. It contains
        a tuple consisting of role (-1 for buyer, +1 for seller), reservation
        price, minimum offer price and maximum offer price.
    """
    def __init__(self, agents, setting, discretization=20,
                 max_factor=0.5, max_steps=30):

        self.observation_space = Box(low=-np.inf, high=np.inf,
                                     shape=setting.observation_space.shape)
        self.action_space = Discrete(discretization)
        self.setting = setting
        self.discretization = discretization

        self.agents = self._init_agent_data(agents, discretization, max_factor)
        buyer_ids  = [a_id for a_id, data in self.agents.items() if data[0] == -1]
        seller_ids = [a_id for a_id, data in self.agents.items() if data[0] == +1]

        self.market = MarketEngine(buyer_ids, seller_ids, max_steps)

    @staticmethod
    def _init_agent_data(agents, discretization, max_factor):
        """Initialize the agent data. """
        result = {}
        for agent_id, role, res_price in agents:
            sign = (-1 if role == 'buyer' else 1)
            c = 1 + sign*max_factor
            a = min(res_price, c*res_price)
            b = max(res_price, c*res_price)
            result[agent_id] = (sign, res_price, a, b)
        return result



    def _scale_observations(self, observations):
        """
        Scale the observations of each agent according to their reservation
        price.
        """
        result = {}
        for agent_id, obs in observations.items():
            sign, res_price, _, _ = self.agents[agent_id]
            result[agent_id] = np.heaviside(obs, 0) * sign *\
                (obs  - res_price)/res_price
        return result


    def _actions_to_prices(self, actions):
        """
        Convert action of each agent to a price according on their reservation
        price.
        """
        result = {}
        for agent_id, action in actions.items():
            N = self.discretization
            m = N/2
            l = action - N/2
            s, _, a, b =  self.agents[agent_id]
            result[agent_id] = ((m - l*s)*a + (m + l*s)*b)/N
        return result


    def _get_rewards(self, agents, deals):
        """
        Compute the reward of each agent given the deals in the last round.
        """
        result = {}
        for agent_id in agents:
            if agent_id in deals:
                sign, res_price, _, _ = self.agents[agent_id]
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
        return self.setting.get_states(self.agents.keys(), self.market)


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
        agent_ids = actions.keys()
        deals = self.market.step(
            self._actions_to_prices(actions)
        )
        obs = self._scale_observations(
            self.setting.get_states(agent_ids, self.market)
        )
        done = {
            agent_id: (agent_id in self.market.done)
            for agent_id in agent_ids
        }
        done["__all__"] = (self.market.done == self.agents.keys())
        rew = self._get_rewards(agent_ids, deals)
        return obs, rew, done, {}

