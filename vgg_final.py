import pickle
import os
import matplotlib.image as mpimg
from mtcnn.mtcnn import MTCNN
import torch
import contextlib
import io
from numpy import asarray
from numpy import expand_dims
from PIL import Image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine

TrainingImagePath='lfw'
TestImagePath='lfw-test/'
update_model = False

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

if update_model:
    for root, dirs, files in os.walk(TrainingImagePath):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                temp = []
                with contextlib.redirect_stdout(io.StringIO()):
                    image = mpimg.imread(os.path.join(root, filename))
                    images.append(image)
                    temp = detector.detect_faces(image)
                    for face in temp:
                        x1, y1, width, height = face['box']
                        x2, y2 = x1 + width, y1 + height
                        face_boundary = image[y1:y2, x1:x2]
                        face_image = Image.fromarray(face_boundary)
                        face_image = face_image.resize(required_size)
                        face_array = asarray(face_image)
                        face_image = expand_dims(face_array, axis=0)
                        faces.append(face_image)
                        labels.append(os.path.dirname(os.path.join(root, filename)).split('/')[-1])

    model_scores = get_model_scores(faces)
    # Saving the face map for future reference
    with open("ModelScores.pkl", 'wb') as fileWriteStream:
        pickle.dump(model_scores, fileWriteStream)
    
    with open("Labels.pkl", 'wb') as fileWriteStream:
        pickle.dump(labels, fileWriteStream)

with open("ModelScores.pkl", "rb") as readStream:
    model_scores = pickle.load(readStream)

with open("Labels.pkl", "rb") as readStream:
    labels = pickle.load(readStream)

test_image = TestImagePath + "Abdullah_al-Attiyah_0001.jpg"

test_image = mpimg.imread(test_image)

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

if sorted_score[0][1] < 0.4:
    print("Matched with " + sorted_score[0][0] + "\nScore: " + str(sorted_score[0][1]))
else:
    print("Match not found\nScore: " + str(sorted_score[0][1]))