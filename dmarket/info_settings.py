import numpy as np
from gym.spaces import Discrete, Box

class InformationSetting:
    """
    Abstract information setting class.

    Attributes
    ----------
    observation_space: gym.spaces object
        The specification of the observation space under this setting.
    """

    def __init__(self):
        pass

    def get_states(self, agent_ids, market):
        """
        Compute the observations of agents given the market object.

        Parameters
        ----------
        agent_ids: list
            A list of agent ids to compute the observations for.

        market: MarketEngine object
            The current market object.

        Returns
        -------
        states: dict
            A dictionary of observations for each agent id. Each observation
            should be an element of the ``observation_space``.
        """
        pass

    def get_state(self, agent_id, market):
        return self.get_states([agent_id], market)[agent_id]


class BlackBoxSetting(InformationSetting):
    """
    The agent is aware of only its own last offer.

    Attributes
    ----------
    observation_space: Box object
        Represents the last offer of the agent. Each element is a numpy array
        with a single entry. If there was no offer, it will be ``[0]``.
    """
    def __init__(self):
        self.observation_space = Box(low=0, high=np.infty, shape=[1])

    def get_states(self, agent_ids, market):
        if not market.offer_history:
            return {agent_id: np.array([0]) for agent_id in agent_ids}

        bids, asks = market.offer_history[-1]
        # This might be a bit slow
        bids = {agent_id: val for val, agent_id in bids}
        asks = {agent_id: val for val, agent_id in asks}
        result = {}
        for agent_id in agent_ids:
            if agent_id in bids:
                result[agent_id] = np.array([bids[agent_id]])
            elif agent_id in asks:
                result[agent_id] = np.array([asks[agent_id]])
            else:
                result[agent_id] = np.array([0])
        return result


class OfferInformationSetting(InformationSetting):
    """
    The agent is aware of the best N offers of either side of the last round.

    Parameters
    ----------
    n_offers: int, optional (default=5)
        Number of offers to see. For instance, 5 would mean the agents see the
        best 5 bids and asks.

    Attributes
    ----------
    observation_space: Box object
        Each element is a numpy array of shape ``(2, n_offers)``. The first row
        contains the bids and second row the asks. No offers will be
        represented by 0.
    """
    def __init__(self, n_offers=5):
        self.n_offers = n_offers
        self.observation_space = Box(low=0, high=np.infty, shape=[2, n_offers])

    def get_states(self, agent_ids, market):
        n = self.n_offers
        offers = np.zeros(shape=(2, n))
        if not market.offer_history:
            return {agent_id: offers for agent_id in agent_ids}

        bids, asks = market.offer_history[-1]
        for i, (bid, agent_id) in enumerate(bids[0:n]): offers[0][i] = bid
        for i, (ask, agent_id) in enumerate(asks[0:n]): offers[1][i] = ask
        # The information each agent gets is the same
        return {agent_id: offers for agent_id in agent_ids}



class DealInformationSetting(InformationSetting):
    """
    The agent is aware of N deals of the last round.

    Note: the N deals need not be sorted on deal price. It depends the order
    the matcher matches deals, see ``MarketEngine.matcher``.

    Parameters
    ----------
    n_deals: int, optional (default=5)
        Number of deals to see.

    Attributes
    ----------
    observation_space: Box object
        Each element is a numpy array of shape ``(n_offers,)``. No deals will
        be represented by 0.
    """
    def __init__(self, n_deals=5):
        self.n_deals = n_deals
        self.observation_space = Box(low=0, high=np.infty, shape=[n_deals])

    def get_states(self, agent_ids, market):
        n = self.n_deals
        if market.deal_history:
            # Here we exploit that deal_history contains the same deal twice
            # in a row, once for the buyer and once for the seller. Since
            # Python >= 3.6 dicts preserve the order of insertion, we can
            # rely on this to obtain the distinct deals that happened.
            deals = list(market.deal_history[-1].values())[0:2*n:2]
            deals = np.pad(deals, (0, n-len(deals))) # Pad it with zeros
        else: deals = np.zeros(n)
        return {agent_id: deals for agent_id in agent_ids}
