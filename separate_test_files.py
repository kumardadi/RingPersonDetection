import os
import shutil

source_dir='lfw'
destination_dir='lfw-test/'

for dirpath, dirnames, filenames in os.walk(source_dir):
    for fileIndex in range(len(filenames) - 1):
        # Move only the first image to destination folder
        shutil.move(os.path.join(dirpath, filenames[fileIndex]), os.path.join(destination_dir, filenames[fileIndex]))