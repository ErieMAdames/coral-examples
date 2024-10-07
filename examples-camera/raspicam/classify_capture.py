# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo to classify Raspberry Pi camera stream using picamera2."""
import argparse
import collections
from collections import deque
import common
import numpy as np
import operator
import os
import tflite_runtime.interpreter as tflite
import time
from picamera2 import Picamera2
import cv2


Category = collections.namedtuple('Category', ['id', 'score'])

def get_output(interpreter, top_k, score_threshold):
    """Returns no more than top_k categories with score >= score_threshold."""
    scores = common.output_tensor(interpreter, 0)
    categories = [
        Category(i, scores[i])
        for i in np.argpartition(scores, -top_k)[-top_k:]
        if scores[i] >= score_threshold
    ]
    return sorted(categories, key=operator.itemgetter(1), reverse=True)

def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_v2_1.0_224_quant_edgetpu.tflite'
    default_labels = 'imagenet_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    args = parser.parse_args()

    with open(args.labels, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)

    interpreter = common.make_interpreter(args.model)
    interpreter.allocate_tensors()

    picam2 = Picamera2()
    # Set the configuration for camera preview at 640x480 resolution
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(config)

    width, height, _ = common.input_image_size(interpreter)

    picam2.start()

    try:
        fps = deque(maxlen=20)
        fps.append(time.time())

        while True:
            start_time = time.time()

            # Capture image from camera
            image = picam2.capture_array()

            # Resize and preprocess the image for the model
            resized_image = cv2.resize(image, (width, height))
            input_tensor = np.asarray(resized_image, dtype=np.uint8)
            common.input_tensor(interpreter)[:,:] = input_tensor

            # Run inference
            start_ms = time.time()
            interpreter.invoke()
            results = get_output(interpreter, top_k=3, score_threshold=0)
            inference_ms = (time.time() - start_ms) * 1000.0

            # Calculate FPS
            fps.append(time.time())
            fps_ms = len(fps) / (fps[-1] - fps[0])

            # Display results
            print('Inference: {:.2f}ms FPS: {:.1f}'.format(inference_ms, fps_ms))
            for result in results:
                print('{:.0f}% {}'.format(100 * result[1], labels[result[0]]))

    finally:
        picam2.stop()

if __name__ == '__main__':
    main()
