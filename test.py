import cv2
import onnxruntime as ort
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, default='./data/88_megaage_asian_32_age.jpg')
    parser.add_argument('--model-path', type=str, default='./age_googlenet.onnx')
    args = parser.parse_args()

    ages = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

    age_classifier = ort.InferenceSession(args.model_path)

    orig_image = cv2.imread(args.image_path)

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image_mean = np.array([104, 117, 123])
    image = image - image_mean
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    input_name = age_classifier.get_inputs()[0].name
    confidences = age_classifier.run(None, {input_name: image})[0]

    idx = confidences[0].argmax()
    confidence = confidences[0][idx]
    age = ages[idx]

    print(age + ': ' + str(confidence))
