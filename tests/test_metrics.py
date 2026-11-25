import numpy as np

from uplift_transform.metrics import qini_coefficient, qini_curve


def test_qini_curve_and_coefficient_match_hand_calculation():
    # Targeting order by score: (y=1,t=1), (y=0,t=1), (y=1,t=0), (y=0,t=0)
    y = [1, 0, 1, 0]
    t = [1, 1, 0, 0]
    score = [4, 3, 2, 1]

    x, gain = qini_curve(y, t, score)

    np.testing.assert_allclose(x, [0, 0.25, 0.5, 0.75, 1.0])
    # Gains: 0, then 1, 1, 1 - 1*(2/1) = -1, then 1 - 1*(2/2) = 0
    np.testing.assert_allclose(gain, [0, 1, 1, -1, 0])
    # Total gain is 0, so the random line is flat at 0; area = 0.125 + 0.25 + 0 - 0.125
    assert np.isclose(qini_coefficient(y, t, score), 0.25)
