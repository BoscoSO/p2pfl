#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""Federated Averaging (FedAvg) Aggregator."""

import os
import pickle
import signal
import tempfile
from typing import List

import numpy as np
import psutil
import tensorflow as tf

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.exceptions import DecodingParamsError
from p2pfl.learning.p2pfl_model import P2PFLModel
from p2pfl.learning.tensorflowlite.keras_tflite_model import MLP, KerasTfliteModel
from p2pfl.learning.tensorflowlite.temp import ModelTFLite


class FedAvg(Aggregator):
    """
    Federated Averaging (FedAvg) [McMahan et al., 2016].

    Paper: https://arxiv.org/abs/1602.05629.
    """

    def aggregate(self, models: List[P2PFLModel]) -> P2PFLModel:
        """
        Aggregate the models.

        Args:
            models: Dictionary with the models (node: model,num_samples).

        Returns:
            A P2PFLModel with the aggregated.

        """
        # Check if there are models to aggregate
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.node_name}) Trying to aggregate models when there is no models")

        # Total Samples
        total_samples = sum([m.get_num_samples() for m in models])

        # Create a Zero Model using numpy
        first_model_weights = models[0].get_parameters()
        accum = [np.zeros_like(layer) for layer in first_model_weights]

        # Add weighted models
        for m in models:
            for i, layer in enumerate(m.get_parameters()):
                accum[i] = np.add(accum[i], layer * m.get_num_samples())

        # Normalize Accum
        accum = [np.divide(layer, total_samples) for layer in accum]

        # Get contributors
        contributors: List[str] = []
        for m in models:
            contributors = contributors + m.get_contributors()

        # Return an aggregated p2pfl model AAAAAAAAAA
        return models[0].build_copy(params=accum, num_samples=total_samples, contributors=contributors)

##############################################################################################################################    TEST

class FedAvgTFLite(Aggregator):

    def aggregate(self, models: List[KerasTfliteModel]) -> KerasTfliteModel:
        """
        Agrega los modelos TFLite.

        Args:
            models: Lista de modelos KerasTfliteModel.

        Returns:
            Un KerasTfliteModel con los pesos agregados.
        """
        if len(models) == 0:
            raise NoModelsToAggregateError("No hay modelos para agregar")

        total_samples = sum(m.get_num_samples() for m in models)


        aggregated_model = MLP(generate_tflite=False)

        for model in models:
            weights_bytes = model.get_tflite_parameters()
            
            weight = model.get_num_samples() / total_samples

            aggregated_model.aggregate(weights_bytes, weight)
            

        aggregated_weights_bytes = aggregated_model.extract_weights_from_tflite()



        # Obtener contribuyentes
        contributors = sum((m.get_contributors() for m in models), [])
        # Crear y devolver un nuevo modelo con los pesos agregados
        return models[0].build_copy(params=aggregated_weights_bytes, num_samples=total_samples, contributors=contributors)
        


