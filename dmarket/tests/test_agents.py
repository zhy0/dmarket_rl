import pytest
from dmarket.agents import *

def test_random_agent():
    b = UniformRandomAgent('buyer', 100)
    s = UniformRandomAgent('seller', 100)

    # Buyers should offer less than their res price
    # and sellers more than their res price
    assert b.get_offer(None) < 100
    assert s.get_offer(None) > 100


def test_rl_agent_disc():
    b = GymRLAgent('buyer', 100)
    s = GymRLAgent('seller', 100)

    # action == 0 should mean worst offer, i.e., reservation price
    assert b.action_to_price(0) == 100
    assert s.action_to_price(0) == 100

    # Price should be decreasing for buyer and increasing for seller
    assert b.action_to_price(0) > b.action_to_price(1)
    assert s.action_to_price(0) < s.action_to_price(1)

    # Price higher than res_price should be positive for seller and vice versa
    assert b.normalize(120) < 0
    assert s.normalize(120) > 0

    # Zero prices (i.e., no offer) should normalize to zero
    assert b.normalize(0) == 0
    assert s.normalize(0) == 0

    # Negative prices should also normalize to zero
    assert b.normalize(-1) == 0
    assert s.normalize(-1) == 0


def test_linear_agent():
    b = TimeLinearAgent('buyer', 100, noise=0)
    s = TimeLinearAgent('seller', 100, noise=0)

    # Buyers should increase and sellers should decrease their price
    assert b.get_offer((None, 0)) < b.get_offer((None, 1))
    assert s.get_offer((None, 0)) > s.get_offer((None, 1))

