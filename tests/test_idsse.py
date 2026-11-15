import numpy as np

from possession.external import idsse

XML = """<?xml version="1.0"?>
<PutDataRequest><Positions>
<FrameSet GameSection="firstHalf" MatchId="M" TeamId="HOME" PersonId="P1">
<Frame N="10" T="t" X="-10.00" Y="1.00" D="0" S="0" A="0" M="1"/>
<Frame N="11" T="t" X="-9.00" Y="1.00" D="0" S="0" A="0" M="1"/>
<Frame N="12" T="t" X="-8.00" Y="1.00" D="0" S="0" A="0" M="1"/>
</FrameSet>
<FrameSet GameSection="firstHalf" MatchId="M" TeamId="AWAY" PersonId="P2">
<Frame N="11" T="t" X="5.00" Y="2.00" D="0" S="0" A="0" M="1"/>
<Frame N="12" T="t" X="6.00" Y="2.00" D="0" S="0" A="0" M="1"/>
</FrameSet>
<FrameSet GameSection="firstHalf" MatchId="M" TeamId="BALL" PersonId="B">
<Frame N="10" T="t" X="0.00" Y="0.00" Z="0.10" D="0" S="0" A="0" M="1" BallPossession="1" BallStatus="1"/>
<Frame N="11" T="t" X="1.00" Y="0.00" Z="0.10" D="0" S="0" A="0" M="1" BallPossession="2" BallStatus="1"/>
<Frame N="12" T="t" X="2.00" Y="0.00" Z="0.10" D="0" S="0" A="0" M="1" BallPossession="2" BallStatus="0"/>
</FrameSet>
</Positions></PutDataRequest>
"""


def test_read_positions_and_half_arrays(tmp_path):
    path = tmp_path / "positions.xml"
    path.write_text(XML)
    halves = idsse.read_positions(str(path))
    n, x, y, poss, status = halves["firstHalf"]["ball"]
    assert list(n) == [10, 11, 12] and list(poss) == [1, 2, 2] and list(status) == [1, 1, 0]
    assert set(halves["firstHalf"]["persons"]) == {"P1", "P2"}
    players, ball, teams, p, s = idsse._half_arrays(halves["firstHalf"], "HOME")
    assert list(teams) == ["H", "A"]
    assert players.shape == (3, 2, 2) and np.isnan(players[0, 1, 0]) and players[1, 1, 0] == 5.0   # the away player has no frame 10
    assert ball[2, 0] == 2.0
