import unittest

import numpy as np

from filters.baselines import moving_average
from sim.propagation import SCENARIO_A, rssi_to_distance


class FilterDomainLogicTests(unittest.TestCase):
    def test_rssi_domain_moving_average_uses_raw_rssi(self) -> None:
        rssi = np.array([-40.0, -60.0, -60.0])

        smoothed_in_rssi_domain = rssi_to_distance(
            moving_average(rssi, 2),
            SCENARIO_A,
        )
        incorrect_distance_domain_path = rssi_to_distance(
            moving_average(rssi_to_distance(rssi, SCENARIO_A), 2),
            SCENARIO_A,
        )

        np.testing.assert_allclose(
            smoothed_in_rssi_domain,
            np.array([1.0, np.sqrt(10.0), 10.0]),
        )
        self.assertFalse(
            np.allclose(
                smoothed_in_rssi_domain,
                incorrect_distance_domain_path,
                equal_nan=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
