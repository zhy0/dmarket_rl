__author__ = "Thomas Asikis"
__credits__ = ["Copyright (c) 2019 Thomas Asikis"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Thomas Asikis"


from gym.spaces import Box
import numpy as np
import pandas as pd
from abc import abstractmethod


class InformationSetting:
    def __init__(self, agents):
        """
        An abstract implementation of an information setting.
        Usually an observation space given the gym specification is provided, to enable
        describing of how a single agent observation looks like.
        Most of the time it will be a box environment, constraint between
        [low, high], with dimension defined by parameter shape.
        :param agents:
        """
        self.agents = agents
        pass

    @abstractmethod
    def get_state(self, agent_id: str, deal_history: pd.DataFrame, agents: pd.DataFrame,
                  offers: pd.DataFrame):
        """
        The method that generates the state for the agents, based on the information setting.
        :param agent_id: usually a string, a unique id for an agent
        :param deal_history: the dictionary containing all the successful deals till now
        :param agents: the dataframe containing the agent information
        :param offers: The dataframe containing the past offers from agents
        :return: In the abstract case, a zero value scalar is retuned.
        """
        return np.zeros(1)


class BlackBoxSetting(InformationSetting):
    def __init__(self, agents):
        """
        The agent is only aware about their past actions, which some positive real value
        representing an offer.
        :param agents: The dataframe of agents in the environment.
        """
        super().__init__(agents)
        self.observation_space = Box(low=0, high=np.infty, shape=[1], dtype=np.float32)

    def get_state(self, agent_id: str, deal_history: pd.DataFrame, agents: pd.DataFrame,
                  offers: pd.DataFrame):
        last_offer = offers[offers['id'] == agent_id]['offer']
        return np.array(last_offer)


class SameSideSetting(InformationSetting):
    def __init__(self, agents):
        """
        The agent is only aware about the last offers submitted by agents sharing the same role.
        The observation for each agent is a vector number of agents dimensions, which contains
        positive
        values for agents of the same role, and zero for agents of the other side.
        :param agents: The dataframe of agents in the environment.
        """
        super().__init__(agents)
        self.observation_space = Box(low=0, high=np.infty, shape=[agents.shape[0]],
                                     dtype=np.float32)

    def get_state(self, agent_id: str, deal_history: pd.DataFrame, agents: pd.DataFrame,
                  offers: pd.DataFrame):
        agent_role = agents.loc[agents['id'] == agent_id]['role'].iloc[0]
        obs = offers[['role', 'offer']]
        obs.loc[obs['role'] == agent_role, 'offer'] = 0
        return obs['offer'].to_numpy()


class OtherSideSetting(InformationSetting):
    def __init__(self, agents):
        """
        The agent is only aware about the last offers submitted by agents sharing the other role.
        The observation for each agent is a vector number of agents dimensions, which contains
        positive
        values for agents of the other role, and zero for agents of the same side.
        :param agents: The dataframe of agents in the environment.
        """
        super().__init__(agents)
        self.observation_space = Box(low=0, high=np.infty, shape=[agents.shape[0]],
                                     dtype=np.float32)

    def get_state(self, agent_id: str, deal_history: pd.DataFrame, agents: pd.DataFrame,
                  offers: pd.DataFrame):
        agent_role = agents.loc[agents['id'] == agent_id]['role'].iloc[0]
        obs = offers[['role', 'offer']]
        obs.loc[obs['role'] != agent_role, 'offer'] = 0
        return obs['offer'].to_numpy()


class FullInformationSetting(InformationSetting):
    def __init__(self, agents):
        """
        The agent is aware about the all offers submitted by agents.
        The observation for each agent is a vector number of agents dimensions, which contains
        positive
        values for offers of all agents.
        :param agents: The dataframe of agents in the environment.
        """
        super().__init__(agents)
        self.observation_space = Box(low=0, high=np.infty, shape=[agents.shape[0]],
                                     dtype=np.float32)

    def get_state(self, agent_id: str, deal_history: pd.DataFrame, agents: pd.DataFrame,
                  offers: pd.DataFrame):
        obs = offers[['role', 'offer']]
        return obs['offer'].to_numpy()


class DealInformationSetting(InformationSetting):
    def __init__(self, agents):
        """
        The agent is aware about the all deal values in the current round.
        The observation for each agent is a vector of dimensions equal to the minimum number
        of agents having the same role. The vector contains positive values for all successful
        offers and zeros otherwise.
        :param agents: The dataframe of agents in the environment.
        """
        super().__init__(agents)
        seller_n = sum(agents['role'] == 'Seller')
        buyer_n = sum(agents['role'] == 'Buyer')
        self.max_deal_n = min(seller_n, buyer_n)
        self.observation_space = Box(low=0, high=np.infty, shape=[self.max_deal_n],
                                     dtype=np.float32)

    def get_state(self, agent_id: str, deal_history: pd.DataFrame, agents: pd.DataFrame,
                  offers: pd.DataFrame):
        res = np.zeros(self.max_deal_n)
        if deal_history is not None:
            for i in range(len(deal_history)):
                res[i] = deal_history[i]['deal_price']
        return res


class DealFullInformationSetting(InformationSetting):
    def __init__(self, agents):
        """
        The agent is aware about the all deal values in the current round and all agent offers.
        The observation for each agent is a vector of dimensions equal to the minimum number
        of agents having the same role plus the number of agents.
        The vector contains positive values for all successful offers and zeros otherwise.
        :param agents: The dataframe of agents in the environment.
        """
        # the observation space for a single agent, is the size of all agents
        # since the other side is unknow part of the observation will be fixed to zero
        super().__init__(agents)
        self.deal_setting = DealInformationSetting(agents)
        self.full_info_setting = FullInformationSetting(agents)
        self.n_feats = agents.shape[0] + self.deal_setting.max_deal_n
        self.observation_space = Box(low=0, high=np.infty, shape=[self.n_feats], dtype=np.float32)
        # action space per agent should be the same

    def get_state(self, agent_id: str, deal_history: pd.DataFrame, agents: pd.DataFrame,
                  offers: pd.DataFrame):
        full_info_obs = self.full_info_setting.get_state(agent_id, deal_history, agents, offers)
        deal_info_obs = self.deal_setting.get_state(agent_id, deal_history, agents, offers)
        res = np.concatenate([full_info_obs, deal_info_obs])
        return res
