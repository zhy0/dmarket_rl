from agents import Buyer, Seller, MarketAgent
#from environments import MarketEnvironment
import random
import math
import numpy as np


class QLearningBuyer(Buyer):
    def __init__(self, agent_id: str,  reservation_price: float, default_price: float, n_states=11, alpha=0.1, gamma=0.1, epsilon=0.1):
        """
        A q-learning buyer agent that extends the market agent
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
        self.q_table = np.zeros((n_states, n_states))
        #Starting state is the default price index
        self.state = 0
        self.next_state = 0
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        #Whether or not the agent has had a deal in this market episode.
        self.done = False
        #List for storing rewards in each episode, not used as info for the learning process
        self.rewards = []
        
    def get_offer(self, previous: float, offers: dict, verbose=False, greedy=False):
        """
        With Information provided by the Market Setting, decide on a new offer that the agent 
        believes will succeed
        """
        #If greedy: always exploit, never expore
        if(greedy):
            eps = 0
        else:
            eps = self.epsilon
        
        if(random.uniform(0,1) < eps):
            self.next_state = random.randint(0,self.n_states-1)
            if verbose: print(f'{self.agent_id} Exploring: next:{self.next_state}, previous:{self.state} ')
        else:
            #self.next_state = np.argmax(self.q_table[self.state])
            self.next_state = np.random.choice(np.where(self.q_table[self.state] == self.q_table[self.state].max())[0])
            if verbose: print(f'{self.agent_id} Exploiting: next:{self.next_state}, previous:{self.state} ')
                
        new = self.offers[self.next_state]
        new_offer = {self.agent_id: new}
        offers.update(new_offer)
        
    def update_table(self, reward, verbose=False):
        '''
        Update q table of agent according to the reward received from market step
        '''        
        if not self.done:
            old_value = self.q_table[self.state, self.next_state]
            next_max = np.max(self.q_table[self.next_state])

            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
            self.q_table[self.state,self.next_state] = new_value
        # If Agent has just had a deal, update self.done
        if ((not self.done) and (reward != 0)):
            #self.done = True
            if verbose: print(f'Agent done! Reward is {reward}')

        self.state = self.next_state
        
class QLearningSeller(Seller):
    def __init__(self, agent_id: str,  reservation_price: float, default_price: float, n_states=11, alpha=0.1, gamma=0.1, epsilon=0.1):
        """
        A q-learning seller agent that extends the market agent
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
        self.q_table = np.zeros((n_states, n_states))
        #Starting state is the default price index
        self.state = n_states - 1
        self.next_state = n_states - 1
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        #Whether or not the agent has had a deal in this market episode.
        self.done = False
        #List for storing rewards in each episode, not used as info for the learning process
        self.rewards = []
        
    def get_offer(self, previous: float, offers: dict, verbose=False, greedy=False):
        """
        With Information provided by the Market Setting, decide on a new offer that the agent 
        believes will succeed
        """
        #If greedy: always exploit, never expore
        if(greedy):
            eps = 0
        else:
            eps = self.epsilon
        
        if(random.uniform(0,1) < eps):
            self.next_state = random.randint(0,self.n_states-1)
            if verbose: print(f'{self.agent_id} Exploring: next:{self.next_state}, previous:{self.state} ')
        else:
            #self.next_state = np.argmax(self.q_table[self.state])
            self.next_state = np.random.choice(np.where(self.q_table[self.state] == self.q_table[self.state].max())[0])
            if verbose: print(f'{self.agent_id} Exploiting: next:{self.next_state}, previous:{self.state} ')
                
        new = self.offers[self.next_state]
        new_offer = {self.agent_id: new}
        offers.update(new_offer)    

    def update_table(self, reward, verbose=False):
        '''
        Update q table of agent according to the reward received from market step
        '''
        
        if not self.done:
            old_value = self.q_table[self.state, self.next_state]
            next_max = np.max(self.q_table[self.next_state])
            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
            if verbose: print(f'Updating: new value of q[{self.state}][{self.next_state}]={new_value}')
            self.q_table[self.state,self.next_state] = new_value
        # If Agent has just had a deal, update self.done
        if ((not self.done) and (reward != 0)):
            #self.done = True
            if verbose: print(f'Agent done!')
                
        self.state = self.next_state