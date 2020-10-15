import importlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch
from algorithms import CCOAlgorithm
from problem_formulation import (
    CCORasterBlanketFormulation,
)
from simulated_rsrp import SimulatedRSRP


class CCOSimulation:
    @dataclass
    class Result:
        """
        Result from one iteration (1 epoch).
            metrics:                A tuple of (weak_coverage percentage and over_coverage percentage)
            objective_value:        Combining objective values
            downtilt_configuration: Downtilt configuration for all sectors
            power_configuration:    Power configuration for all sectors
        """

        metrics: Tuple[float, float]
        objective_value: float
        downtilt_configuration: np.ndarray
        power_configuration: np.ndarray

    def __init__(
        self,
        simulated_rsrp: SimulatedRSRP,
        problem_formulation: CCORasterBlanketFormulation,
        algorithm: CCOAlgorithm,
        epochs: int,
    ):
        self.simulated_rsrp = simulated_rsrp
        self.problem_formulation = problem_formulation
        self.epochs = epochs
        self.algorithm = algorithm

    def run(self) -> Iterator[Tuple[float, float]]:
        # set random seed
        torch.manual_seed(1)
        np.random.seed(1)

        for _ in range(self.epochs):
            # get the configuration, objective_value and metrics from current algorithm
            configuration, objective_value, metrics = self.algorithm.step()

            result = CCOSimulation.Result(
                metrics=metrics,
                objective_value=objective_value,
                downtilt_configuration=configuration[0],
                power_configuration=configuration[1],
            )
            yield result

    @staticmethod
    def run_from_json(json_data: Dict[Any, Any]) -> Iterator[Tuple[float, float]]:
        cco_simulation = CCOSimulation.construct_from_json(json_data)
        for result in cco_simulation.run():
            yield result

    @staticmethod
    def construct_from_json(json_data: Dict[Any, Any]) -> "CCOSimulation":
        # Create SimulatedRSRP object
        simulated_rsrp = SimulatedRSRP.construct_from_npz_files(
            json_data["simulated_rsrp"]["path"],
            tuple(json_data["simulated_rsrp"]["power_range"]),
        )

        # Create problem formulation object
        problem_formulation_module = importlib.import_module(
            json_data["problem_formulation"]["module"]
        )
        problem_formulation_class = getattr(
            problem_formulation_module, json_data["problem_formulation"]["classname"]
        )
        problem_formulation = problem_formulation_class(
            **json_data["problem_formulation"]["parameters"]
        )

        # Create ML algorithms module object
        algorithm_module = importlib.import_module(json_data["algorithm"]["module"])
        algorithm_class = getattr(algorithm_module, json_data["algorithm"]["classname"])

        # Algorithm object construct
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        algorithm = algorithm_class(
            simulated_rsrp=simulated_rsrp,
            problem_formulation=problem_formulation,
            device=device,
            **json_data["algorithm"]["parameters"],
        )

        # Create simulation object
        simulation_module = importlib.import_module(json_data["simulation"]["module"])
        simulation_class = getattr(
            simulation_module, json_data["simulation"]["classname"]
        )
        cco_simulation = simulation_class(
            simulated_rsrp=simulated_rsrp,
            problem_formulation=problem_formulation,
            algorithm=algorithm,
            epochs=json_data["simulation"]["epochs"],
        )
        return cco_simulation

    @staticmethod
    def print_result(result: Result, epoch: Optional[int] = None) -> None:
        os.system("clear")
        if epoch:
            print("epoch :: ", epoch)
        print("Weak Coverage Percentage :: ", result.metrics[0] * 100)
        print("Over Coverage Percentage :: ", result.metrics[1] * 100)
        print("Combining Objective Value :: ", result.objective_value)


def run(json_data: Dict[Any, Any]):
    for i, result in enumerate(CCOSimulation.run_from_json(json_data), 1):
        CCOSimulation.print_result(result, i)
