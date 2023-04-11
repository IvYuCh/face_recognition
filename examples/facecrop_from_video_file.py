import os

import face_recognition
import cv2
from pathlib import Path
from tqdm import tqdm

# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open the input movie file
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--faces_dir_name", type=str, required=True)
parser.add_argument("--video_name", type=str, required=True)
parser.add_argument("--output_dir_name", type=str, required=True)
parser.add_argument("--frs_retr_ps", type=float, default=5.0, required=False)
args = parser.parse_args()
video = os.path.join('/video', args.video_name)
faces_dir = os.path.join('/faces_dir', args.faces_dir_name)
output_dir = os.path.join('/output_dir', args.output_dir_name)
input_movie = cv2.VideoCapture(video)
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(input_movie.get(cv2.CAP_PROP_FPS))
freq = min(args.frs_retr_ps / fps, 1)
print(freq)

# Create an output directory
os.makedirs(output_dir, exist_ok=True)

# Load some sample pictures and learn how to recognize them.
known_faces = []
known_faces_names = []
for face_file in os.listdir(faces_dir):
    image = face_recognition.load_image_file(face_file)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(face_encoding)
    print(face_file, Path(face_file).stem)
    known_faces_names.append((Path(face_file).stem))


# Initialize some variables
face_locations = []
face_encodings = []
frame_number = 0
retr_number = 0

for _ in tqdm(range(length)):
    # Grab a single frame of video
    ret, frame = input_movie.read()
    # Quit when the input video file ends
    if not ret:
        break
    frame_number += 1
    if (retr_number + 1) / frame_number > freq:
        continue
    retr_number += 1

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        name = None
        for i, m in enumerate(match):
            if m:
                name = known_faces_names[i]
                break
        face_names.append(name)

    # Write the results of cropped image
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # if unknown face skip cropping
        if not name:
            continue
        cropped_image = frame[top:bottom, left:right]
        filename = os.path.join(output_dir, f'{frame_number}_{name}.jpg')
        cv2.imwrite(filename, cropped_image)

# All done!
input_movie.release()
cv2.destroyAllWindows()
