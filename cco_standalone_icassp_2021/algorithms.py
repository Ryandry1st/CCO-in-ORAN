from typing import Tuple

import numpy as np
import torch
from problem_formulation import (
    CCORasterBlanketFormulation,
)
from simulated_rsrp import SimulatedRSRP


class CCOAlgorithm:
    def __init__(
        self,
        simulated_rsrp: SimulatedRSRP,
        problem_formulation: CCORasterBlanketFormulation,
        **kwargs,
    ):
        self.simulated_rsrp = simulated_rsrp
        self.problem_formulation = problem_formulation

        # Get configuration range for downtilts and powers
        (
            self.downtilt_range,
            self.power_range,
        ) = self.simulated_rsrp.get_configuration_range()

        # Get the number of total sectors
        _, self.num_sectors = self.simulated_rsrp.get_configuration_shape()

    def step(self) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        float,
        Tuple[float, float],
    ]:
        """Stub for one step of the algorithm.

            Return tuple::
            1. configuration : nested tuple of optimal tilts and optimal powers
            2. reward : weigted combination of metrics
            3. metrics : tuple of dual objectives : under-coverage and over-coverage
        """
        return [None, None], None, [0.0, 0.0]


class RandomSelection(CCOAlgorithm):
    def __init__(
        self,
        simulated_rsrp: SimulatedRSRP,
        problem_formulation: CCORasterBlanketFormulation,
        **kwargs,
    ):
        super().__init__(
            simulated_rsrp=simulated_rsrp, problem_formulation=problem_formulation
        )

    def step(self) -> Tuple[Tuple[np.ndarray, np.ndarray], float, Tuple[float, float]]:
        # Random select powers and downtilts

        downtilts_for_sectors = np.random.uniform(
            self.downtilt_range[0], self.downtilt_range[1], self.num_sectors
        )
        power_for_sectors = np.random.uniform(
            self.power_range[0], self.power_range[1], self.num_sectors
        )

        # power_for_sectors = [max_tx_power_dBm] * num_sectors
        configuration = (downtilts_for_sectors, power_for_sectors)
        # Get the rsrp and interferences powermap
        (
            rsrp_powermap,
            interference_powermap,
            _,
        ) = self.simulated_rsrp.get_RSRP_and_interference_powermap(configuration)

        # According to the problem formulation, calculate the reward
        reward = self.problem_formulation.get_objective_value(
            rsrp_powermap, interference_powermap
        )

        # Get the metrics
        metrics = self.problem_formulation.get_weak_over_coverage_area_percentages(
            rsrp_powermap, interference_powermap
        )
        return configuration, reward, metrics
