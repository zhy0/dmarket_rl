import gym
from gym.spaces import Discrete

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
