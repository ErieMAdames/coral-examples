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

def draw_results(image, results, labels, fps, inference_ms):
    """Draw inference results, bounding boxes, and FPS on the image."""
    # Flip the image vertically
    image = cv2.flip(image, 0)

    # Draw FPS and inference time
    cv2.putText(image, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f'Inference: {inference_ms:.2f}ms', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw bounding boxes and labels
    for result in results:
        label = labels[result.id]
        score = result.score
        # Assuming result.bbox is in (xmin, ymin, xmax, ymax) format
        xmin, ymin, xmax, ymax = result.bbox
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(image, f'{label}: {100*score:.0f}%', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return image

def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_v2_1.0_224_quant_edgetpu.tflite'
    default_labels = 'imagenet_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
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

            # Draw results on the image
            annotated_image = draw_results(image, results, labels, fps_ms, inference_ms)

            # Display the image with annotations
            cv2.imshow("Inference Results", annotated_image)

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
