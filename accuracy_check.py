import pickle
import os
import matplotlib.image as mpimg
from mtcnn.mtcnn import MTCNN
import torch
from numpy import asarray
from numpy import expand_dims
from PIL import Image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
import time

TrainingImagePath='lfw'
TestImagePath='lfw-test/'
update_model = False
COSINE_LIMITER = 0.4

images = []
faces = []
torch.cuda.set_device(0)
detector = MTCNN()
required_size=(224, 224)
labels = []

# create a vggface model object
model = VGGFace(model='resnet50',
    include_top=False,
    input_shape=(224, 224, 3),
    pooling='avg')

def get_model_scores(faces):
    scores = []
    for face in faces:
        sample = asarray(face, 'float32')
        # prepare the data for the model
        sample = preprocess_input(sample, version=2)
        # perform prediction
        scores.append(model.predict(sample).flatten())
    return scores



with open("ModelScores.pkl", "rb") as readStream:
    model_scores = pickle.load(readStream)

with open("Labels.pkl", "rb") as readStream:
    labels = pickle.load(readStream)

total_counter = 0
correct_counter = 0
match_not_found_counter = 0

for root, dirs, files in os.walk(TestImagePath):
    for filename in files:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            start_time = time.time()
            test_image = mpimg.imread(os.path.join(root, filename))
            test_face = detector.detect_faces(test_image)[0]
            x1, y1, width, height = test_face['box']
            x2, y2 = x1 + width, y1 + height
            face_boundary = test_image[y1:y2, x1:x2]
            # resize pixels to the model size
            face_image = Image.fromarray(face_boundary)
            face_image = face_image.resize(required_size)
            face_array = asarray(face_image)
            face_image = [expand_dims(face_array, axis=0)]
            test_model_score = get_model_scores(face_image)[0]
            score_list = [(labels[x], float(cosine(model_scores[x], test_model_score))) for x in range(len(labels))]
            sorted_score = sorted(score_list, key=lambda x: x[1], reverse=False)
            total_counter += 1
            end_time = time.time()
            print("Time taken:", end_time - start_time, "seconds")
            if sorted_score[0][1] < COSINE_LIMITER:
                print("Matched with " + sorted_score[0][0] + ", Score: " + str(sorted_score[0][1]))
                if sorted_score[0][0] in filename:
                    correct_counter += 1
            else:
                print("Match not found, Score: " + str(sorted_score[0][1]))
                match_not_found_counter += 1


print("Accuracy: " + str(correct_counter/total_counter*100))
print("Not found Inaccuracy: " + str(match_not_found_counter/total_counter*100))