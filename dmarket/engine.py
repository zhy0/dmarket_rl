import numpy as np

class MarketEngine:
    """
    Core double auction, single unit market matching enigne

    Parameters
    ----------
    buyers : array_like
        List of buyer ids, should be distinct.

    sellers : array_like
        List of seller ids, should be distinct.

    max_steps: int (optional, default=30)
        Number of maximum market rounds.


    Attributes
    -------
    time: int
        The current time step the market is in. Starts at 0.

    done: set
        The set of agent ids that are done for this game. If ``max_steps`` is
        reached, this will contain all agent ids.

    offer_history: list
        This is a list of all offers up until the current step. Each entry of
        this list is a tuple of the form ``(bids, asks)`` which represent the
        set of bids and asks respectively for that time step. Both ``bids`` and
        ``asks`` are in the form used by the matcher, i.e., they are lists of
        tuples of the form ``[(offer1, agent_id1), (offer2, agent_id2), ...]``.

    deal_history: list
        A list of all deals up until the current time step. Each entry contains
        a dict of the form ``{agent_id1: deal_price1, ...}`` for all agents
        that were matched in that round.
    """

    def __init__(self, buyers, sellers, max_steps=30):
        self.buyers = set(buyers)
        self.sellers = set(sellers)
        self.agents = self.buyers.union(self.sellers)
        self.max_steps = max_steps
        self.reset()


    def reset(self):
        """Reset the market to its initial unmatched state."""
        self.time = 0
        self.done = set()
        self.offer_history = list()
        self.deal_history = list()


    def step(self, offers):
        """
        Compute the next market state given a set of offers.

        This function will update each of the class attributes ``time``,
        ``done``, ``offer_history`` and ``deal_history``.

        Parameters
        ----------
        offers: dict
            A dictionary of offers indexed by agent_id.

        Returns
        -------
        deals: dict
            A dictionary indexed by agent_id containing the deal price of each
            succesfully matched market agent.
        """
        bids = []
        asks = []

        for agent_id, offer in offers.items():
            if agent_id in self.done:
                continue
            elif agent_id in self.buyers:
                bids.append((offer, agent_id))
            elif agent_id in self.sellers:
                asks.append((offer, agent_id))
            else:
                raise RuntimeError(f"Received offer from unkown agent {agent_id}")

        deals = self.match(bids, asks)
        self.deal_history.append(deals)
        self.offer_history.append((bids, asks))
        self.time += 1

        for agent_id in deals:
            self.done.add(agent_id)

        if self.time >= self.max_steps \
           or self.buyers.issubset(self.done) \
           or self.sellers.issubset(self.done):
            self.done = self.agents

        return deals


    @staticmethod
    def match(bids, asks):
        """
        Core matching algorithm for market engine.

        Matching is done as follows: If a bid is higher than an ask, then the
        buyer will be matched with the seller. If there are multiple bids and
        asks that could be matched, then matching is determined by the offer
        placed: the highest bidder will be matched with the lowest asking
        seller, the second highest bidder will be matched with the second
        lowest seller and so on.  No match will happen if none of the bids
        exceed the asks.

        The deal price is determined to be the mid-price of the matched bid and
        ask. Note: the deal price does not have to be monotone in the offers of
        either side, i.e., it could happen that the second highest bidder
        obtains a better deal price than the highest bidder.

        Parameters
        ----------
        bids: list of tuples
            A list of the form ``[(bid1, agent_id1), (bid2, agent_id2), ...]``.
        asks: list of tuples
            A list of the form ``[(ask1, agent_id1), (ask2, agent_id2), ...]``.

        Returns
        -------
        deals: dict
            A dictionary indexed by agent_id containing the deal price. Only
            agents that were matched will be in this dict. Note: the same deal
            price will appear twice under the buyer and seller id.
        """
        bids.sort(reverse=True)
        asks.sort(reverse=False)
        deals = dict()
        n = min(len(bids), len(asks))
        for (bid, buyer_id), (ask, seller_id) in zip(bids[0:n], asks[0:n]):
            if bid >= ask:
                price = (bid + ask)/2
                deals[buyer_id] = price
                deals[seller_id] = price
            else:
                break
        return deals

