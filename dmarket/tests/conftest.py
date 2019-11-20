import pytest
from dmarket.engine import MarketEngine


@pytest.fixture()
def market():
    def create_market(n_buyers, n_sellers):
        buyer_ids  = list(range(n_buyers))
        seller_ids = list(range(n_buyers, n_buyers + n_sellers))
        return MarketEngine(buyer_ids, seller_ids)
    return create_market

