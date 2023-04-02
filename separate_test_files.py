import os
import shutil

source_dir='lfw'
destination_dir='lfw-test/'

for dirpath, dirnames, filenames in os.walk(source_dir):
    # Check if subdirectory contains more than one image
    if len(filenames) > 1:
        # Move only the first image to destination folder
        shutil.move(os.path.join(dirpath, filenames[0]), os.path.join(destination_dir, filenames[0]))