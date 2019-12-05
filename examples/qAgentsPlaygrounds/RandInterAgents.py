from agents import Buyer, Seller, MarketAgent
#from environments import MarketEnvironment
import random
import math


class RandInterBuyer(Buyer):
    def __init__(self, agent_id: str,  reservation_price: float, default_price: float):
        """
        A new buyer agent that extends the market agent
        :param agent_id: a unique id that differentiates this agent to other agents
        :param reservation_price: the reservation price, or maximum price that this agent is
        willing to buy
        """
        super().__init__(agent_id, reservation_price)
        self.default_price = default_price
        #List for storing rewards in each episode, not used as info for the learning process
        self.rewards = []

    def get_offer(self, previous: float, offers: dict, verbose=False):
        """
        With Information provided by the Market Setting, decide on a new offer that the agent 
        believes will succeed
        """
        if (previous==0): previous = self.default_price
        if(previous >= self.reservation_price - self.reservation_price*0.05): 
            if(verbose): print(f"{self.agent_id}: reached self reservation price: {self.reservation_price}")
            new = self.reservation_price
        else :
            new = math.floor(random.uniform(previous, self.reservation_price))
            if(verbose): print(f"{self.agent_id}: between {previous} and {self.reservation_price} is {new}.")
        new_offer = {self.agent_id: new}
        offers.update(new_offer) 


class RandInterSeller(Seller):
    def __init__(self, agent_id: str,  reservation_price: float, default_price = float):
        """
        A new seller agent that extends the market agent
        :param agent_id: a unique id that differentiates this agent to other agents
        :param reservation_price: the reservation price, or minimum price that this agent is
        willing to sell
        """
        super().__init__(agent_id, reservation_price)
        self.default_price = default_price
        #List for storing rewards in each episode, not used as info for the learning process
        self.rewards = []
        
    def get_offer(self, previous: float, offers: dict, verbose=False):
        """
        With Information provided by the Market Setting, decide on a new offer that the agent 
        believes will succeed
        """
        if (previous==0): previous = self.default_price
        if(previous <= self.reservation_price + self.reservation_price*0.05):
            if(verbose): print(f"{self.agent_id}: reached self reservation price: {self.reservation_price}")
            new = self.reservation_price  
        else :
            new = math.ceil(random.uniform(previous, self.reservation_price))
            if(verbose): print(f"{self.agent_id}: between {previous} and {self.reservation_price} is {new}.")
        new_offer = {self.agent_id: new}
        offers.update(new_offer)