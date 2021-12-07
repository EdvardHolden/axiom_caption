import tensorflow as tf

x1 = tf.random.normal([32, 7, 7, 512])
# Text data - random integers
x2 = tf.random.uniform(shape=[32, 22], dtype=tf.int32, maxval=100)
# Try to load and predict


# Load model anbd predict
loaded_model = tf.keras.models.load_model('ckpt_dir')
#latest_checkpoint = tf.train.latest_checkpoint('ckpt_dir')
# loaded_model.load_weights(latest_checkpoint)
#print(f'Restored from {latest_checkpoint}.')
print()
print("Model predictions")
print(loaded_model.predict([x1, x2]))  # is working
# TODO: update!