import pytest

from dmarket.info_settings import *


def test_blackbox(market):
    m = market(10, 10)
    b = BlackBoxSetting()

    # Initial state should be zeros
    assert b.get_states([0,1], m) == {0: [0], 1: [0]}

    # Agents should see their respective offers
    m.step({0: 100, 1: 150})
    assert b.get_states([0,1, 2], m) == {0: [100], 1: [150], 2: [0]}


def test_offer_info(market):
    m = market(10,10)
    setting = OfferInformationSetting(5)

    # Initial state should be of shape (2, 5)
    assert setting.get_states([0], m)[0].shape == (2, 5)

    # First row should be bids, second should be asks
    m.step({0: 90, 10: 110})
    np.testing.assert_array_equal(
        setting.get_states([0, 10], m)[0],
        [
            [90, 0, 0, 0, 0],
            [110, 0, 0, 0, 0]
        ]
    )

    # It should contain the best deals of both sides, sorted
    m.reset()
    m.step({
        0: 90, 1: 91, 2: 92, 3: 93, 4: 94, 5: 105,
        10: 100, 11: 101, 12: 102, 13: 103, 14: 104, 15: 95,
    })
    np.testing.assert_array_equal(
        setting.get_states([0, 10], m)[0],
        [
            [105, 94, 93, 92, 91],
            [95, 100, 101, 102, 103],
        ]
    )


def test_deal_info(market):
    m = market(10,10)
    setting = DealInformationSetting(5)

    # Initial state should be of shape (2, 5)
    assert setting.get_states([0], m)[0].shape == (5,)

    # It should contain deals made, not necessarily sorted on deal price value
    m.step({0: 110, 1: 96, 10: 90, 11: 94})
    np.testing.assert_array_equal(
        setting.get_states([0, 10], m)[0],
        [100, 95, 0, 0, 0],
    )
