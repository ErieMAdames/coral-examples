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

"""An object tracking demo using a Raspberry Pi camera and Coral TPU."""

import argparse
import collections
import cv2
import numpy as np
import os
import time
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

# Helper function for bounding boxes
def draw_bounding_box(image, box, label, score):
    ymin, xmin, ymax, xmax = box
    im_height, im_width, _ = image.shape
    (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width),
                                  int(ymin * im_height), int(ymax * im_height))
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
    label = f'{label}: {int(score * 100)}%'
    cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Load labels from the file
def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

# Parse the model output
def get_output(interpreter, score_threshold=0.5):
    boxes = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    class_ids = interpreter.get_tensor(interpreter.get_output_details()[1]['index'])
    scores = interpreter.get_tensor(interpreter.get_output_details()[2]['index'])
    count = interpreter.get_tensor(interpreter.get_output_details()[3]['index'])[0]

    results = []
    for i in range(int(count)):
        if scores[i] >= score_threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': class_ids[i],
                'score': scores[i]
            }
            results.append(result)
    return results

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='.tflite model path')
    parser.add_argument('--labels', required=True, help='Label file path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Score threshold')
    args = parser.parse_args()

    # Load the TFLite model
    interpreter = Interpreter(model_path=args.model, experimental_delegates=[load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()

    # Load labels
    labels = load_labels(args.labels)

    # Setup Picamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()

            # Prepare input tensor
            _, height, width, _ = interpreter.get_input_details()[0]['shape']
            input_frame = cv2.resize(frame, (width, height))
            input_data = np.expand_dims(input_frame, axis=0)

            # Run inference
            interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
            interpreter.invoke()

            # Get output
            results = get_output(interpreter, score_threshold=args.threshold)

            # Draw bounding boxes and labels
            for result in results:
                label = labels[int(result['class_id'])]
                draw_bounding_box(frame, result['bounding_box'], label, result['score'])

            # Display frame with bounding boxes
            cv2.imshow('Object Tracker', frame)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
