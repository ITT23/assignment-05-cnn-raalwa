import sys
import time

import config
import numpy as np
import tensorflow as tf
from pynput.keyboard import Controller, Key
from transformer import Transformer

video_id = 0

if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

transformer = Transformer(video_id)
model = tf.keras.models.load_model("03-media_control/trained_gesture_model")

keyboard = Controller()

while True:
    image = transformer.get_transformed_image()
    if image is not None:
        prediction = model.predict(image)
        label = config.LABEL_NAMES[np.argmax(prediction)]
        if label == 'like':
            keyboard.press(Key.media_volume_up)
            keyboard.release(Key.media_volume_up)
            print('volume up')
        if label == 'dislike':
            keyboard.press(Key.media_volume_down)
            keyboard.release(Key.media_volume_down)
            print('volume down')
        if label == 'stop':
            keyboard.press(Key.media_play_pause)
            keyboard.release(Key.media_play_pause)
            print('stop/start')
    else:
        print("Waiting for board...")
    time.sleep(0.3)
