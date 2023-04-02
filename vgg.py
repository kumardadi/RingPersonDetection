from keras.preprocessing.image import ImageDataGenerator
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mtcnn.mtcnn import MTCNN


# Understand more about ImageDataGenerator at below link
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# Defining pre-processing transformations on raw images of training data
# These hyper parameters helps to generate slightly twisted versions
# of the original image, which leads to a better model, since it learns
# on the good and bad mix of images

TrainingImagePath='lfw'

train_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

# Defining pre-processing transformations on raw images of testing data
# No transformations are done on the testing images
test_datagen = ImageDataGenerator()

# Generating the Training Data
training_set = train_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=256,
        class_mode='categorical')


# Generating the Testing Data
test_set = test_datagen.flow_from_directory(
        TrainingImagePath,
        target_size=(64, 64),
        batch_size=256,
        class_mode='categorical')

# Printing class labels for each face
test_set.class_indices

TrainClasses=training_set.class_indices
 
# Storing the face and the numeric tag for future reference
ResultMap={}
for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
    ResultMap[faceValue]=faceName
 
# Saving the face map for future reference
with open("ResultsMap.pkl", 'wb') as fileWriteStream:
    pickle.dump(ResultMap, fileWriteStream)


faces = []
detector = MTCNN(verbose = False, device = 'cuda')

for root, dirs, files in os.walk(TrainingImagePath):
    for filename in files:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            temp = detector.detect_faces(mpimg.imread(os.path.join(root, filename)))
            if temp > 1:
                print(os.path.join(root, filename))
            faces.append(temp) 

print(len(faces))