import pytest
from ..scenarios.scen_4 import Scenario4
from ..scenarios.scen_9 import Scenario9
from ..scenarios.scen_10 import Scenario10
from ..scenarios.scen_11 import Scenario11

def test_scenario4_generate_returns_list():
    scen4 = Scenario4(num_trades=5, counter_offset=100)
    data = scen4.generate()
    assert isinstance(data, list)
    assert all(isinstance(pair, tuple) and len(pair)==2 for pair in data)
    # must contain at least one counter-trade (negative start=end)
    assert any(s<0 and s==e for s,e in data)


def test_scenario9_and_balance():
    scen9 = Scenario9()
    data = scen9.generate()
    assert isinstance(data, list)
    # check zeros separators present
    assert any(pair==(0,0) for pair in data)


def test_scenario10():
    pass

def test_scenario11():
    pass



#WRITE UNITE TEST TO CHECK TRADED MATCHES