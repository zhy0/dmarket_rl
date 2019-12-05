from agents import Buyer, Seller, MarketAgent
#from environments import MarketEnvironment
import random
import math
import numpy as np


class RandOfferBuyer(Buyer):
    def __init__(self, agent_id: str,  reservation_price: float, default_price: float, n_states=11):
        """
        A reinforced buyer agent that extends the market agent
        The agent has a discrete number of states, which correspond to its offer at the timestep
        :param agent_id: a unique id that differentiates this agent to other agents
        :param reservation_price: the reservation price, or maximum price that this agent is
        willing to buy
        :param default_price: the default price, or starting price that this agent is 
        willing to buy at the first timestep. Smaller than the reservation price
        """
        assert reservation_price > default_price,"Buyer Default Price must be smaller than the Reservation Price!"
        super().__init__(agent_id, reservation_price)
        self.n_states = n_states
        self.default_price = default_price
        self.offers = np.linspace(default_price, reservation_price, n_states)
        #List for storing rewards in each episode, not used as info for the learning process
        self.rewards = []
        
    def get_offer(self, previous: float, offers: dict, verbose=False):
        """
        With Information provided by the Market Setting, decide on a new offer that the agent 
        believes will succeed
        """
        new = self.offers[random.randint(0,self.n_states-1)]
        new_offer = {self.agent_id: new}
        offers.update(new_offer) 


class RandOfferSeller(Seller):
    def __init__(self, agent_id: str,  reservation_price: float, default_price: float, n_states=11):
        """
        A reinforced seller agent that extends the market agent
        The agent has a discrete number of states, which correspond to its offer at the timestep
        :param agent_id: a unique id that differentiates this agent to other agents
        :param reservation_price: the reservation price, or minimum price that this agent is
        willing to sell
        :param default_price: the default price, or starting price that this agent is 
        willing to sell at the first timestep. Greater than the reservation price
        """
        assert reservation_price < default_price,"Seller Default Price must be greater than the Reservation Price!"
        super().__init__(agent_id, reservation_price)
        self.n_states = n_states
        self.default_price = default_price
        self.offers = np.linspace(reservation_price, default_price, n_states)
        #List for storing rewards in each episode, not used as info for the learning process
        self.rewards = []
        
    def get_offer(self, previous: float, offers: dict, verbose=False):
        """
        With Information provided by the Market Setting, decide on a new offer that the agent 
        believes will succeed
        """
        new = self.offers[random.randint(0,self.n_states-1)]
        new_offer = {self.agent_id: new}
        offers.update(new_offer)