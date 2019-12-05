import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

from QLearningAgents import QLearningBuyer, QLearningSeller
from RandOfferAgents import RandOfferBuyer, RandOfferSeller
from RandInterAgents import RandInterBuyer, RandInterSeller

def get_agents_equal(sellr_reserve = 20, buyer_reserve = 100,
                     sellr_default = 100, buyer_default = 20,
                     n_rnd_off_buyers = 0, n_rnd_off_sellrs = 0,
                     n_rnd_int_buyers = 0, n_rnd_int_sellrs = 0,
                     n_q_learn_buyers = 5, n_q_learn_sellrs = 5):
    '''
    Return requested amount of buyers, sellers with equal reservation, default prices.
    Also return list of Q Learning Agents, used for training them.
    '''
    rnd_off_buyers = []
    rnd_off_sellrs = []

    rnd_int_buyers = []
    rnd_int_sellrs = []

    q_learn_buyers = []
    q_learn_sellrs = []
    
    #Random Buyers
    for i in range(n_rnd_off_buyers):
        rnd_off_buyers.append(RandOfferBuyer(agent_id = f'B[r{i}]', reservation_price = buyer_reserve, default_price = buyer_default))
    #Random Sellers
    for i in range(n_rnd_off_sellrs):
        rnd_off_sellrs.append(RandOfferSeller(agent_id = f'S[r{i}]', reservation_price = sellr_reserve, default_price = sellr_default))    
    
    #RandomIntervalBuyers
    for i in range(n_rnd_int_buyers):
        rnd_int_buyers.append(RandInterBuyer(agent_id = f'B[n{i}]', reservation_price = buyer_reserve, default_price = buyer_default))
    #RandomIntervalSellers    
    for i in range(n_rnd_int_sellrs):
        rnd_int_sellrs.append(RandInterSeller(agent_id = f'S[n{i}]', reservation_price = sellr_reserve, default_price = sellr_default))
    
    #Q Learning Buyers
    for i in range(n_q_learn_buyers):
        q_learn_buyers.append(QLearningBuyer(agent_id = f'B[q{i}]', reservation_price = buyer_reserve, default_price = buyer_default))
    #Q Learning Sellers
    for i in range(n_q_learn_sellrs):
        q_learn_sellrs.append(QLearningSeller(agent_id = f'S[q{i}]', reservation_price = sellr_reserve, default_price = sellr_default))

    buyers = rnd_off_buyers + rnd_int_buyers + q_learn_buyers
    sellers = rnd_off_sellrs + rnd_int_sellrs + q_learn_sellrs
    
    inequality = abs(len(sellers)-len(buyers))
    
    q_learn_agents = q_learn_buyers + q_learn_sellrs
    
    return buyers, sellers, inequality, q_learn_agents
    
def update_q_tables(q_learn_agents, done_dict, rewards_dict, negative_reward=0):
    '''
    Updates the Q Table of each Q Learning Agent in environment.
    
    For each step in episode:
    Until the agent has a deal, get negative reward.
    When the agent has a deal, get the reward, as given by algorithm.
    After the agent has had a deal, do not change Q Table.
    '''
    for agent in q_learn_agents:
        if(not done_dict[agent.agent_id]):
            agent.update_table(negative_reward)
        elif(not agent.done):
            agent.update_table(rewards_dict[agent.agent_id])
            agent.done = True
            
def calculate_stats(agents, rewards_dict, agent_sum, n_stats, episode):
    '''
    Calculate and print statistics from the training/testing episodes
    
    After each episode, either:
    (Every once in n_stats episodes) append to agent's rewards list 
    the average reward over last n_states
    or:
    (Every other time) add the latest reward to cumulative reward
    since last averaging
    '''
    for index, agent in enumerate(agents):
        reward = rewards_dict[agent.agent_id]
        if episode % n_stats == 0:
            temp_reward = agent_sum[index] / n_stats
            agent.rewards.append(temp_reward)
            agent_sum[index] = reward
            print(f'{agent.agent_id}: Rewards={temp_reward}')
        else:
            agent_sum[index] += reward
            
def calculate_steps(n_stats, episode, steps, steps_sum, steps_list):
    '''
    Calculate and print statistics from the training/testing episodes
    
    After each episode, either:
    (Every once in n_stats episodes) append to agent's rewards list 
    the average reward over last n_states
    or:
    (Every other time) add the latest reward to cumulative reward
    since last averaging
    '''

    if episode % n_stats == 0:
        steps_avg = steps_sum / n_stats
        steps_list.append(steps_avg)
        steps_sum = steps
        print(f'Episode {episode}: Steps={steps_avg}')
        return steps_sum, steps_avg
    else:
        steps_sum += steps
        return steps_sum, 0
           
def plot_stats(agents, n_stats, PATH, steps_list=None):
    '''
    Plot statistics from the training/testing episodes
    '''
    # Plot Rewards evolution
    for agent in agents:
        if(agent.reservation_price < agent.default_price):
            plt.plot(np.arange(0, len(agent.rewards)*n_stats, n_stats), agent.rewards, 'b', label=f'{agent.agent_id}')
        else:
            plt.plot(np.arange(0, len(agent.rewards)*n_stats, n_stats), agent.rewards, 'r', label=f'{agent.agent_id}')
    plt.title('Averaged Rewards')
    plt.xlabel('Episodes')
    plt.ylabel(f'Rewards Average over {n_stats} Episodes')
    plt.legend()
    plt.savefig(f'{PATH}/Rewards.pdf')
    plt.show()
    
    if steps_list:
        # Plot Steps evolution
        plt.plot(np.arange(0, len(agents[0].rewards)*n_stats, n_stats), steps_list, label=f'{agents[0].agent_id}')
        plt.title('Steps per Episode')
        plt.xlabel('Episodes')
        plt.ylabel(f'Steps Average over {n_stats} Episodes')
        plt.legend()
        plt.savefig(f'{PATH}/Steps.pdf')
        plt.show()
        
def plot_q_tables(q_agents, PATH):
    for q_agent in q_agents:
        plt.pcolor(q_agent.q_table)
        axes = plt.axes()
        plt.colorbar()
        plt.title(f'{q_agent.agent_id} : Q Table')
        plt.xlabel('Next Offer')
        plt.ylabel('Current Offer')
        axes.set_xticks([x for x in range(0, q_agent.n_states)])
        axes.set_yticks([x for x in range(0, q_agent.n_states)])
        axes.set_xticklabels(q_agent.offers)
        axes.set_yticklabels(q_agent.offers)
        plt.savefig(f'{PATH}/q_table_{q_agent.agent_id}.svg')
        plt.show()
def save_stats(agents, n_stats, steps_list, PATH):
    '''
    Data is saved in format: Episode, Steps, Rewards
    '''
    stats = np.zeros((len(steps_list), (2 + len(agents))))
    stats[:,0] = np.arange(0, len(agents[0].rewards)*n_stats, n_stats)
    stats[:,1] = steps_list
    for j, agent in enumerate(agents):
        stats[:,j+2] = agent.rewards
    np.savetxt(f'{PATH}/stats.csv', stats, delimiter=',')
    
    
def learn(market_env, buyers, sellers, q_learn_agents, n_episodes, n_stats, negative_reward, inequality):
    """
    Train given Agents by running n_episodes episodes
    with specified market environment, agents, and other statistics
    """
    assert q_learn_agents, 'No agents that can learn has been given. Use method evaluate instead. '
    buyer_sum = np.zeros(len(buyers))
    seller_sum = np.zeros(len(sellers))

    steps_sum = 0
    steps_list = []

    for i in range(1, n_episodes + 1):
        state = market_env.reset()

        all_done = all(market_env.done.values())
        reward = {}
        rewards = {}

        # Variable to store whether agent is done for Episode
        for q_agent in q_learn_agents:
            q_agent.done = False

        # Do as many steps as necessary for all agents
        while not all_done:
            #dictionary to hold offers in current step of the episode
            step_offers = {}
            # Get offers from agents
            for agent in buyers + sellers:
                f = market_env.setting.get_state(agent_id=agent.agent_id,
                                         deal_history=market_env.deal_history,
                                         agents=market_env.agents,
                                         offers=market_env.offers)
                agent.get_offer(f[0], step_offers, verbose=False)

            # Market step
            observations, rewards, done, _ = market_env.step(step_offers)

            # Update q tables of q-learning agents
            update_q_tables(q_learn_agents, done, rewards, negative_reward)

            # Check whether all agents are done/or if more deals can be made
            all_done = all(done.values())
            if(inequality != 0):
                # If unequal number of sellers and buyers exist, one agent will not be "done" even though Episode should end
                done_count = 0
                for is_done in done.values():
                    if not is_done:
                        done_count += 1
                if(done_count == inequality):
                    all_done = True

            # Gather all non-zero rewards
            # (rewards from deals struck in this step of the episode)
            # In the last iteration of loop: All deal rewards are captured
            for item in rewards.items():
                if(item[1] != 0):
                    temp = {item[0] : item[1]}
                    reward.update(temp)

        # Save only the deal-striking rewards from each episode.
        rewards.update(reward)
        # Clear output for better visibility
        if i % n_stats == 0:
            clear_output(wait = True)    
        # Calculate Average number of steps per episode in the last n_stats episodes
        steps_sum, steps_avg = calculate_steps(n_stats=n_stats, episode=i, steps=market_env.time,
                                               steps_sum=steps_sum, steps_list=steps_list)
        # Calculate Average rewards for buyers per episode in the last n_stats episodes
        calculate_stats(agents=buyers, rewards_dict=rewards, agent_sum=buyer_sum,
                        n_stats=n_stats, episode=i)
        # Calculate Average rewards for sellers per episode in the last n_stats episodes
        calculate_stats(agents=sellers, rewards_dict=rewards, agent_sum=seller_sum,
                        n_stats=n_stats, episode=i)
    return steps_list
    