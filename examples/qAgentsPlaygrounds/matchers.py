__author__ = "Thomas Asikis"
__credits__ = ["Copyright (c) 2019 Thomas Asikis"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Thomas Asikis"

import random
import pandas as pd
from abc import abstractmethod


class Matcher:
    def __init__(self):
        """
        Abstract matcher object. This object is used by the Market environment to match agent offers
        and also decide the deal price.
        """
        pass

    @abstractmethod
    def match(self,
              current_actions: dict,
              offers: pd.DataFrame,
              env_time: int,
              agents: pd.DataFrame,
              matched: set,
              done: dict,
              deal_history: pd.DataFrame):
        """
        The matching method, which relies on several data structures passed from the market object.
        :param current_actions: A dictionary of agent id and offer value
        :param offers: The dataframe containing the past offers from agents
        :param env_time: the current time step in the market
        :param agents: the dataframe containing the agent information
        :param matched: the set containing all the ids of matched agents in this round
        :param done: the dictionary with agent id as key and a boolean value to determine if an
        agent has terminated the episode
        :param deal_history: the dictionary containing all the successful deals till now
        :return: the dictionary containing the the agent id as keys and the rewards as values
        """
        rewards: dict = None
        return rewards


class RandomMatcher(Matcher):
    def __init__(self, reward_on_reference=False):
        """
        A random matcher, which decides the deal price of a matched pair by sampling a uniform
        distribution bounded in [seller_ask, buyer_bid] range.
        The reward is calculated as the difference from cost or the difference to budget for
        sellers and buyers.
        :param reward_on_reference: The parameter to use a different reward calculation.
        If set to true the reward now becomes: offer - reservation price for, sellers
        and: reservation price - offer, for buyers.
        You may chose to use this reward scheme, but you have to justify why it is better than
        the old!
        """
        super().__init__()
        self.reward_on_reference = reward_on_reference

    def match(self,
              current_actions: dict,
              offers: pd.DataFrame,
              env_time: int,
              agents: pd.DataFrame,
              matched: set,
              done: dict,
              deal_history: pd.DataFrame):
        """
        The matching method, which relies on several data structures passed from the market object.
        :param current_actions: A dictionary of agent id and offer value
        :param offers: The dataframe containing the past offers from agents
        :param env_time: the current time step in the market
        :param agents: the dataframe containing the agent information
        :param matched: the set containing all the ids of matched agents in this round
        :param done: the dictionary with agent id as key and a boolean value to determine if an
        agent has terminated the episode
        :param deal_history: the dictionary containing all the successful deals till now
        :return: the dictionary containing the the agent id as keys and the rewards as values
        """
        # update offers
        for agent_id, offer in current_actions.items():
            if agent_id not in matched:
                offers.loc[offers['id'] == agent_id, ['offer', 'time']] = (offer, env_time)
        # keep buyer and seller offers with non-matched ids sorted:
        # descending by offer value for buyers
        # ascending by offer value for sellers
        # and do a second sorting on ascending time to break ties for both
        buyer_offers = offers[(offers['role'] == 'Buyer') &
                              (~offers['id'].isin(matched))] \
            .sort_values(['offer', 'time'], ascending=[False, True])

        seller_offers = offers[(offers['role'] == 'Seller') &
                               (~offers['id'].isin(matched))] \
            .sort_values(['offer', 'time'], ascending=[True, True])

        min_len = min(seller_offers.shape[0], buyer_offers.shape[0])
        rewards = dict((aid, 0) for aid in agents['id'].tolist())
        for i in range(min_len):
            considered_seller = seller_offers.iloc[i, :]
            considered_buyer = buyer_offers.iloc[i, :]
            if considered_buyer['offer'] >= considered_seller['offer']:
                # if seller price is lower or equal to buyer price
                # matching is performed
                matched.add(considered_buyer['id'])
                matched.add(considered_seller['id'])

                # keeping both done and matched is redundant
                done[considered_buyer['id']] = True
                done[considered_seller['id']] = True

                deal_price = random.uniform(considered_seller['offer'], considered_buyer[
                    'offer'])
                if self.reward_on_reference:
                    rewards[considered_buyer['id']] = considered_buyer['res_price'] - considered_buyer['offer']
                    rewards[considered_seller['id']] = considered_seller['offer'] - considered_seller['res_price']
                else:
                    rewards[considered_buyer['id']] = considered_buyer['offer'] - deal_price
                    rewards[considered_seller['id']] = deal_price - considered_seller['offer']
                matching = dict(Seller=considered_seller['id'], Buyer=considered_buyer['id'],
                                time=env_time, deal_price=deal_price)
                deal_history.append(matching)
            else:
                # not possible that new matches can occur after this failure due to sorting.
                break

        return rewards
