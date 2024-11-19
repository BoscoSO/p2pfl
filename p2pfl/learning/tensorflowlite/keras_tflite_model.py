#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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

"""Keras model abstraction for P2PFL."""

import os
import tempfile
from typing import Dict, List, Optional, Union

import numpy as np
import tensorflow as tf

from p2pfl.learning.exceptions import ModelNotMatchingError
from p2pfl.learning.p2pfl_model import P2PFLModel
from p2pfl.learning.tensorflowlite.temp.ModelTFLite import ModelTFLite


INPUT_SIZE = 224
IMG_SIZE = 28
NUM_CLASSES = 10
BATCH_SIZE_XS = 8
BATCH_SIZE_S = 16
BATCH_SIZE_M = 32
BATCH_SIZE_L = 64

#####################
#    KerasModel     #
#####################


class ModelNotBuiltError(Exception):
    """Raised when a model is not built."""

    pass


class KerasTfliteModel(P2PFLModel):
    """
    P2PFL model abstraction for TensorFlowLite/Keras.

    Args:
        model: The Keras model to encapsulate.
        params: Optional initial parameters (list of NumPy arrays or bytes).
        num_samples: Optional number of samples used for training.
        contributors: Optional list of contributor nodes.
        additional_info: Optional dictionary for extra information.

    """

    def __init__(
        self,
        model: tf.keras.Model,
        params: Optional[Union[List[np.ndarray], bytes]] = None,
        num_samples: Optional[int] = 900,
        contributors: Optional[List[str]] = None,
        additional_info: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize the KerasModel."""
        super().__init__(model, params, num_samples, contributors, additional_info)

        # Ensure the model is built
        if len(model.get_weights()) == 0:  # type: ignore
            raise ModelNotBuiltError(
                "Model must be built before creating a P2PFLMODEL! Please be sure that model.get_weights() return a non empty list."
            )
        
    def build_copy(self, **kwargs) -> "P2PFLModel":
        """
        Build a copy of the model.

        Args:
            **kwargs: Parameters of the model initialization.

        Returns:
            A copy of the model.

        """
        params = kwargs.get('params')
        num_samples = kwargs.get('num_samples')
        contributors = kwargs.get('contributors')
        #additional_info = kwargs.get('additional_info', self.get_additional_info())

        if isinstance(params, bytes):
            new_model = MLP(generate_tflite=False)

            new_model_obj = KerasTfliteModel(
                model=new_model,
                num_samples=num_samples,
                contributors=contributors,
            )
            new_model_obj.set_parameters(params)
            
            return new_model_obj
        else:
            new_model = MLP(generate_tflite=False)

            return KerasTfliteModel(
                model=new_model,
                num_samples=num_samples,
                contributors=contributors,
            )
           
    
    
    def get_tflite_parameters(self) -> bytes:
        """
        Get the parameters of the model.

        Returns:
            The parameters of the model

        """
        return self.model.extract_weights_from_tflite()
    
    def get_parameters(self) -> List[np.ndarray]:
        """
        Get the parameters of the model.

        Returns:
            The parameters of the model

        """
        return self.model.get_weights()

   
            
    def set_parameters(self, params: Union[List[np.ndarray], bytes]) -> None:
        """
        Set the parameters of the model.

        Args:
            params: The parameters of the model.

        Raises:
            ModelNotMatchingError: If parameters don't match the model.

        """
        #params = self.decode_parameters(params)
        try:
            if isinstance(params, bytes):
                self.model.apply_checkpoint_to_tflite(params)
            else:
                print("noes bytes")

        except ValueError as e:
            raise ModelNotMatchingError("Parameters don't match the model. Please check the model architecture and the parameters.") from e

    


    def encode_parameters(self, params: Optional[List[np.ndarray]] = None) -> bytes:
        """
        Encode the parameters of the model.

        Args:
            params: The parameters of the model.

        """
        #if params is None:
        #    params = self.model.get_weights()
        
      
        return self.model.extract_weights_from_tflite()
  
    def decode_parameters(self, data: bytes) -> List[np.ndarray]:
        """
        Decode the parameters of the model.

        Args:
            data: The parameters of the model.


        """
        return None


    def aggregate_model(self, checkpoint_bytes, weight):
        self.model.aggregate(checkpoint_bytes,weight)
        
####
# Example MLP
####


class MLP(tf.keras.Model):
    """Multilayer Perceptron (MLP) for MNIST classification using Keras."""

    def __init__(self, out_channels=10, lr_rate=0.0001, seed=None,  generate_tflite=True, **kwargs):
        """
        Initialize the MLP.

        Args:
            hidden_sizes (list): List of integers representing the number of neurons in each hidden layer.
            out_channels (int): Number of output classes (10 for MNIST).
            lr_rate (float): Learning rate for the Adam optimizer.
            seed (int, optional): Random seed for reproducibility.
            kwargs: Additional arguments.

        """
        super().__init__(**kwargs)
        
        if seed is not None:
            tf.random.set_seed(seed)
        

        self.model_path= "p2pfl/learning/tensorflowlite/temp/basic_model_batches.tflite"

        self.model = ModelTFLite()        

        if(generate_tflite == True):
            self.create_tflite()

        self.interpreter = tf.lite.Interpreter(self.model_path)
       

    def call(self, inputs):
        """Forward pass of the MLP."""
        return self.model(inputs)

############################################################################


    def create_tflite(self):
        """Convierte el modelo a TFLite y lo guarda en el path especificado."""

        new_model = ModelTFLite()
        new_model.set_weights(self.model.get_weights())

        temp_dir = tempfile.mkdtemp()
        temp_saved_model_dir = os.path.join(temp_dir, 'temp_saved_model')

        try:
            print("1. Guardando modelo...")
            tf.saved_model.save(new_model, temp_saved_model_dir, signatures={
                'train_fixed_batch_xs': new_model.train_fixed_batch_xs.get_concrete_function(),
                'train_fixed_batch_s': new_model.train_fixed_batch_s.get_concrete_function(),
                'train_fixed_batch_m': new_model.train_fixed_batch_m.get_concrete_function(),
                'train_fixed_batch_l': new_model.train_fixed_batch_l.get_concrete_function(),
                'infer': new_model.infer.get_concrete_function(),
                'save': new_model.save.get_concrete_function(),
                'restore': new_model.restore.get_concrete_function(),
                'aggregate': new_model.aggregate.get_concrete_function(),
            })
            
            print("2. Creando convertidor...")
            converter = tf.lite.TFLiteConverter.from_saved_model(temp_saved_model_dir)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter.experimental_enable_resource_variables = True
            converter.allow_custom_ops = True

            print("3. Convirtiendo modelo...")
            tflite_model = converter.convert()
        
            print("4. Guardando modelo TFLite...")
            with open(self.model_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"Modelo TFLite guardado en {self.model_path}")
            return True

        except Exception as e:
            print(f"Error durante la creación del modelo TFLite: {str(e)}")
            return False

        finally:
            print("5. Limpiando directorio temporal...")
            if os.path.exists(temp_dir):
                for root, dirs, files in os.walk(temp_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(temp_dir)
            self.interpreter = tf.lite.Interpreter(self.model_path)



   

############################################################################ CHECKPOINTS LOGIC


    def cargar_pesos_checkpoint_a_tf(self, checkpoint_bytes) -> List[np.ndarray]:

        temp_file_fd, temp_file_path = tempfile.mkstemp(suffix='.ckpt')
        os.close(temp_file_fd)  # Cerrar el descriptor de archivo

        try:
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(checkpoint_bytes)

            # Cargar los pesos del checkpoint
            reader = tf.train.load_checkpoint(temp_file_path)
            shape_from_key = reader.get_variable_to_shape_map()
            
        
            # Mapear y cargar los pesos en el modelo TensorFlow
            for layer in self.model.layers:
                if len(layer.weights) > 0:
                    pesos_capa = []
                    for peso in layer.weights:
                        nombre_peso = peso.name.split(':')[0]  # Eliminar ':0' si está presente
                        nombre_peso_checkpoint = nombre_peso + ':0'  # Añadir ':0' para buscar en el checkpoint
                        if nombre_peso_checkpoint in shape_from_key:
                            peso_checkpoint = reader.get_tensor(nombre_peso_checkpoint)
                            if peso_checkpoint.shape == peso.shape:
                                pesos_capa.append(peso_checkpoint)
                            else:
                                print(f"Advertencia: La forma no coincide para {nombre_peso}. Checkpoint: {peso_checkpoint.shape}, Modelo: {peso.shape}")
                                pesos_capa.append(peso.numpy())
                        else:
                            print(f"Advertencia: No se encontró {nombre_peso_checkpoint} en el checkpoint")
                            pesos_capa.append(peso.numpy())

                    try:
                        layer.set_weights(pesos_capa)
                    except ValueError as e:
                        print(f"Error al establecer pesos para la capa {layer.name}: {e}")

            print("Proceso de carga de pesos completado.")
            return self.model.get_weights()
        except Exception as e:
            print(f"Error al aplicar el checkpoint a h5: {str(e)}")
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def apply_checkpoint_to_tflite(self, checkpoint_bytes):
        #self.cargar_pesos_checkpoint_a_tf(checkpoint_bytes)
        self.interpreter.allocate_tensors()
        restore_signature = self.interpreter.get_signature_runner('restore')

        print(f"apply_checkpoint_to_tflite: {len(checkpoint_bytes)} bytes")
        temp_file_fd, temp_file_path = tempfile.mkstemp(suffix='.ckpt')
        os.close(temp_file_fd)  # Cerrar el descriptor de archivo

        try:
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(checkpoint_bytes)

            checkpoint_path_array = np.array([temp_file_path], dtype=np.string_)

            restore_signature(checkpoint_path=checkpoint_path_array)

        except Exception as e:
            print(f"Error al aplicar el checkpoint: {str(e)}")
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)


    def extract_weights_from_tflite(self) -> bytes:
        self.interpreter.allocate_tensors()
        save_signature = self.interpreter.get_signature_runner('save')
        
        temp_file_fd, temp_file_path = tempfile.mkstemp(suffix='.ckpt')
        os.close(temp_file_fd)  # Cerrar el descriptor de archivo

        try:
            # Convertir la ruta del archivo temporal a un array NumPy de strings
            checkpoint_path_array = np.array([temp_file_path], dtype=np.string_)
            
            save_signature(checkpoint_path=checkpoint_path_array)
        
            with open(temp_file_path, 'rb') as temp_file:
                weights_bytes = temp_file.read()
            
            print(f"extract_weights_from_tflite: {len(weights_bytes)} bytes")
            
            return weights_bytes
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)




    def aggregate(self, checkpoint_bytes, weight):
        self.interpreter.allocate_tensors()
        aggregate_signature = self.interpreter.get_signature_runner('aggregate')
        
        temp_file_fd, temp_file_path = tempfile.mkstemp(suffix='.ckpt')
        os.close(temp_file_fd)  # Cerrar el descriptor de archivo

        try:
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(checkpoint_bytes)

            checkpoint_path_array = np.array([temp_file_path], dtype=np.string_)
            weight_tensor = tf.constant(weight, dtype=tf.float32)
            
            aggregate_signature(checkpoint_path=checkpoint_path_array,weight=weight_tensor)

        except Exception as e:
            print(f"Error en aggregate con el checkpoint: {str(e)}")
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

