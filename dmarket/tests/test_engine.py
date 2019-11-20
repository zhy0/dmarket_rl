import pytest
import numpy as np

def test_market_step(market):
    m = market(1,1)
    m.step({})
    assert m.time == 1
    assert m.done == set()
    assert m.offer_history ==[([], [])]
    assert m.deal_history == [{}]


def test_market_bid_ask_sorted(market):
    m = market(3, 3)
    offers = {
        0: 100,
        1: 90,
        2: 95,

        3: 110,
        4: 100,
        5: 105
    }
    m.step(offers)
    assert m.offer_history[-1] == (
        [(100, 0), (95, 2), (90, 1)],
        [(100, 4), (105, 5), (110, 3)],
    )

def test_market_done(market):
    m = market(1, 3)
    # No one should be done in zeroth round
    assert m.done == set()

    # Everyone should be done once the only buyer matches
    m.step({0: 100, 1: 100})
    assert m.done == {0, 1, 2, 3}

    # Everyone should be done after max_steps
    m.reset()
    for i in range(m.max_steps): m.step({})
    assert m.done == {0, 1, 2, 3}

