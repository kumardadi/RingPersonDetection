import cv2
import numpy as np

# Load pre-trained object detection model
model = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')

# Load pre-trained classifier model
classifier = cv2.face.LBPHFaceRecognizer_create()
classifier.read('classifier.yml')

# Load label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy')

# Load training set and labels
training_set = np.load('training_set.npy')
labels = np.load('labels.npy')

# Load input image
image = cv2.imread('input_image.jpg')

# Perform object detection on input image
blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True)
model.setInput(blob)
output = model.forward()

# Find person in detected objects
for detection in output[0, 0, :, :]:
    if detection[1] == 1: # Person detected
        x1, y1, x2, y2 = (detection[3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])).astype(int)
        person_image = image[y1:y2, x1:x2]

        # Classify the detected person
        gray = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, (100, 100), interpolation=cv2.INTER_LINEAR)
        label_id, _ = classifier.predict(face)
        label = label_encoder.inverse_transform(label_id)

        if label not in label_encoder.classes_:
            # Prompt user for label
            label = input('Enter label for person: ')

            # Add new image with label to training set
            training_set = np.concatenate([training_set, face[np.newaxis]])
            labels = np.concatenate([labels, [label]])

            # Train the classifier with updated training set
            classifier.train(training_set, labels)

            # Save updated training set and classifier model
            np.save('training_set.npy', training_set)
            np.save('labels.npy', labels)
            classifier.write('classifier.yml')

        # Output detected person's label
        print('Detected person:', label)
