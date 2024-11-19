import os

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.initializers import Constant

tf.config.set_visible_devices([], 'GPU')

SAVED_MODEL_DIR = './saved_model_basic'
IMG_SIZE = 28
INPUT_SIZE = 32

BATCH_SIZE_XS = 8
BATCH_SIZE_S = 16
BATCH_SIZE_M = 32
BATCH_SIZE_L = 64

NUM_CLASSES = 10

class ModelTFLite(tf.keras.Model):
  def create_base_model(self, input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(24, kernel_size=(6, 6), strides=1)(inputs)
    x = layers.BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(rate=0.25)(x)
    
    x = layers.Conv2D(48, kernel_size=(5, 5), strides=2)(x)
    x = layers.BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(rate=0.25)(x)
    
    x = layers.Conv2D(64, kernel_size=(4, 4), strides=2)(x)
    x = layers.BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(rate=0.25)(x)
    
    return tf.keras.Model(inputs, x)

  def __init__(self, learning_rate=0.0001, **kwargs): 
    super().__init__(**kwargs)

    
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


    base_model = self.create_base_model((INPUT_SIZE, INPUT_SIZE, 3))

    self.model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='input'),
        
        layers.Resizing(INPUT_SIZE, INPUT_SIZE),
        layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x)),  
        
        base_model,
            
        layers.Flatten(),
        
        layers.Dense(200),
        layers.BatchNormalization(scale=False, beta_initializer=Constant(0.01)),
        layers.Activation('relu'),
        layers.Dropout(rate=0.25),
            

        layers.Dense(NUM_CLASSES, activation=None, name='output')
    ])

    self.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # si true quitar activation='softmax'
        metrics=['accuracy']
    )

    x_sample = tf.random.normal([1, 28, 28, 1], dtype=tf.float32)

    # Pasar el tensor por el modelo para construirlo
    _ = self.model(x_sample)


  


  def get_signatures(self):
    """Devuelve un diccionario con todas las firmas."""
    return {
        'train_fixed_batch_xs': self.train_fixed_batch_xs.get_concrete_function(),
        'train_fixed_batch_s': self.train_fixed_batch_s.get_concrete_function(),
        'train_fixed_batch_m': self.train_fixed_batch_m.get_concrete_function(),
        'train_fixed_batch_l': self.train_fixed_batch_l.get_concrete_function(),
        'infer': self.infer.get_concrete_function(),
        'save': self.save.get_concrete_function(),
        'restore': self.restore.get_concrete_function(),
    }
 

#############################################################TRAIN BATCH

  @tf.function(input_signature=[
      tf.TensorSpec([BATCH_SIZE_XS, IMG_SIZE, IMG_SIZE, 1], tf.float32),
      tf.TensorSpec([BATCH_SIZE_XS], tf.int32),
  ])
  def train_fixed_batch_xs(self, x, y):
    with tf.GradientTape() as tape:
        prediction = self.model(x)
        loss = self.model.loss(y, prediction)
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(prediction, axis=1), tf.int32), y), tf.float32))
    return {"loss": loss, "accuracy": accuracy}
  
  @tf.function(input_signature=[
      tf.TensorSpec([BATCH_SIZE_S, IMG_SIZE, IMG_SIZE, 1], tf.float32),
      tf.TensorSpec([BATCH_SIZE_S], tf.int32),
  ])
  def train_fixed_batch_s(self, x, y):
    with tf.GradientTape() as tape:
        prediction = self.model(x)
        loss = self.model.loss(y, prediction)
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(prediction, axis=1), tf.int32), y), tf.float32))
    return {"loss": loss, "accuracy": accuracy}
  

  @tf.function(input_signature=[
      tf.TensorSpec([BATCH_SIZE_M, IMG_SIZE, IMG_SIZE, 1], tf.float32),
      tf.TensorSpec([BATCH_SIZE_M], tf.int32),
  ])
  def train_fixed_batch_m(self, x, y):
    with tf.GradientTape() as tape:
        prediction = self.model(x)
        loss = self.model.loss(y, prediction)
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(prediction, axis=1), tf.int32), y), tf.float32))
    return {"loss": loss, "accuracy": accuracy}
  

  @tf.function(input_signature=[
      tf.TensorSpec([BATCH_SIZE_L, IMG_SIZE, IMG_SIZE, 1], tf.float32),
      tf.TensorSpec([BATCH_SIZE_L], tf.int32),
  ])
  def train_fixed_batch_l(self, x, y):
    with tf.GradientTape() as tape:
        prediction = self.model(x)
        loss = self.model.loss(y, prediction)
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(prediction, axis=1), tf.int32), y), tf.float32))
    return {"loss": loss, "accuracy": accuracy}
  

#############################################################


  @tf.function(input_signature=[
      tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 1], tf.float32),
  ])
  def infer(self, x):
    logits = self.model(x)
    probabilities = tf.nn.softmax(logits, axis=-1)
    return {
        "output": probabilities,
        "logits": logits
    }

  @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  def save(self, checkpoint_path):
    tensor_names = [weight.name for weight in self.model.weights]
    tensors_to_save = [weight.read_value() for weight in self.model.weights]
    tf.raw_ops.Save(
        filename=checkpoint_path, tensor_names=tensor_names,
        data=tensors_to_save, name='save')
    return {
        "checkpoint_path": checkpoint_path
    }

  @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  def restore(self, checkpoint_path):
    restored_tensors = {}
    for var in self.model.weights:
      restored = tf.raw_ops.Restore(
          file_pattern=checkpoint_path, tensor_name=var.name, dt=var.dtype,
          name='restore')
      var.assign(restored)
      restored_tensors[var.name] = restored
    return restored_tensors
  
  
  @tf.function(input_signature=[
    tf.TensorSpec(shape=[], dtype=tf.string),  # checkpoint_path
    tf.TensorSpec(shape=[], dtype=tf.float32)  # weight
  ])
  def aggregate(self, checkpoint_path, weight):
    
    current_weights = self.model.weights
    
    other_weights = self.restore(checkpoint_path)
    
    # Realizar la agregación
    for current, other in zip(current_weights, other_weights.values()):
        aggregated = current * (1 - weight) + other * weight
        current.assign(aggregated)
    
    return {"status": tf.constant("Aggregation completed")}
