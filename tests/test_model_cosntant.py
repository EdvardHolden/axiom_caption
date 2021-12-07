# File for testing what sort of constant I can store in the model
# as I need to store an axiom ordering
import os
import tensorflow as tf
from keras.layers import Dense
from enum import Enum, auto

model_dir = '../experiments/base_model'

class testEnum(Enum):
    test = auto()

class DummyModel(tf.keras.Model):

    def __init__(self, units, property, name='dummy', **kwargs):
        super(DummyModel, self).__init__(name=name, *kwargs)
        self.units = units
        self.property = property
        self.constant = tf.constant('my_constant', dtype=None, shape=None, name='Const')

        self.fc = Dense(units, activation='relu')
        self.out = Dense(2)

    def call(self, inputs, training=None):
        x = inputs
        x = self.fc(x, training=training)
        return self.out(x, training=training)

    def __str__(self):
        return f'DummyModel {self.units} {self.property}'



#model = DummyModel(10, 'test')
model = DummyModel(10, testEnum.test)

#model.compile()
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.predict([[1, 1, 1, 0, 0]])

# Initialise the checkpoint manager
checkpoint_path = os.path.join(model_dir, 'ckpt_dir')
print('OG model ', model)

# Save the model
model.save(checkpoint_path)
print('Model saved')
print(model.property)
print(model.constant)


print()
from model import load_model
#loaded_model = load_model(checkpoint_path)
loaded_model = tf.keras.models.load_model(checkpoint_path)
print('Loaded model ', loaded_model)
print(loaded_model.constant)
print(loaded_model.property)


