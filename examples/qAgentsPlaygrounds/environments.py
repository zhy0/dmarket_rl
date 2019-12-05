__author__ = "Thomas Asikis"
__credits__ = ["Copyright (c) 2019 Thomas Asikis"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Thomas Asikis"

from gym import Env
from abc import abstractmethod
import pandas as pd


class MarketEnvironment(Env):
    def __init__(self, sellers: list, buyers: list, max_steps: int, matcher, setting):
        """
        An abstract market environment extending the typical gym environment
        :param sellers: A list containing all the agents that are extending the Seller agent
        :param buyers: A list containing all the agents that are extending the Buyer agent
        :param max_steps: the maximum number of steps that runs for this round.
        """
        self.sellers = [dict(id=x.agent_id, res_price=x.reservation_price, role="Seller") for x in
                        sellers]
        self.buyers = [dict(id=x.agent_id, res_price=x.reservation_price, role="Buyer")
                       for x in buyers]
        self.agents = pd.DataFrame(self.sellers + self.buyers)
        self.agent_ids = set(self.agents['id'].unique())
        self.agent_roles = self.agents[['id', 'role']].set_index('id').to_dict()['role']
        self.max_steps = max_steps

        # assign matcher and assign info setting
        self.matcher = matcher
        self.setting = setting(self.agents)

        self.n_sellers = len(self.sellers)
        self.n_buyers = len(self.buyers)
        self.matched: set = None
        self.deal_history: list = None
        self.offers = None
        self.current_actions = dict()
        self.realized_deals = None
        self.time = None
        self.done = None
        self.reset()

    def step(self, actions):
        """
        The step function takes the agents actions and returns the new state, reward,
        id the state is terminal and any other info.
        :param actions: a dictionary containing the action per agent
        :return: a tuple of 4 objects: the object describing the next state, a data structure
        containing the reward per agent, a data structure containing boolean values expressing
        whether an agent reached a terminal state, and finally a dictionary object containing any extra info.
        """
        rewards = self.matcher.match(
            current_actions=actions,
            offers=self.offers,
            env_time=self.time,
            agents=self.agents,
            matched=self.matched,
            done=self.done,
            deal_history=self.deal_history
        )
        new_state = dict((agent_id, self.setting.get_state(agent_id, self.deal_history, self.agents,
                                                           self.offers))
                         for agent_id in self.agents['id'])
        self.time += 1
        return new_state, rewards, self.done, None

    def reset(self):
        """
        Resets the environment to an initial state, so that the game can be repeated.
        :return: the initial state so that a new round begins.
        """
        self.matched = set()
        self.deal_history = list()
        self.time = 0
        self.done = dict((x, False) for x in self.agents['id'].tolist())
        self._init_offers()
        self.realized_deals = []
        self.current_actions = dict()
        new_state = dict((agent_id, self.setting.get_state(agent_id, self.deal_history, self.agents,
                                                           self.offers))
                         for agent_id in self.agents['id'])

        return new_state

    def _init_offers(self):
        zero_actions = dict((agent_id, 0) for agent_id in self.agents['id'].unique())
        self.offers = pd.merge(self.agents, pd.Series(zero_actions,
                                                      name='offer').reset_index()
                               .rename(columns={"index": "id"}), on='id')
        self.offers['time'] = self.time
        # display(self.offers)

    @abstractmethod
    def render(self, mode='human'):
        """
        This method renders the environment in a specific visualiation. e.g. human is to render
        for a human observer.
        :param mode: Please check the gym env docstring
        :return: A render object
        """
        pass
