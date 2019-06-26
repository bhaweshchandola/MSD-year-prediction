import face_recognition
import cv2
import pickle



def predict(frame, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    # if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
    #     raise Exception("Invalid image path: {}".format(X_img_path))

    # if knn_clf is None and model_path is None:
    #     raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # frame = X_img_path
    # Load image file and find face locations
    # X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(frame)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(frame, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    print("close",closest_distances)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    print("**********************")
    print("face", knn_clf.predict(faces_encodings))
    print("xface", X_face_locations)
    print("are",are_matches)
    print("close",closest_distances)
    print("************end***********")

    fourth = closest_distances
    print("fou1", type(fourth[0]))
    print("f_tols", fourth[0].tolist())
    print("sunm", sum(fourth[0].tolist(), []))
    fourth = list(map(lambda x : int(x*100), sum(fourth[0].tolist(), [])))
    print("fou2", fourth)


    # Predict classes and remove classifications that aren't within the threshold
    # return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)] orignal without percentage

    return [(pred, loc, clos) if rec else ("unknown", loc, clos) for pred, loc, rec, clos in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches, fourth)]


def show_prediction_labels_on_image(frame, predictions):
    """
    Shows the face recognition results visually.
    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    # pil_image = Image.open(img_path).convert("RGB")
    # draw = ImageDraw.Draw(pil_image)
    print("percentage is :", predictions)
    for name, (top, right, bottom, left), percen in predictions:
        print("percentage is :", predictions)
        # # Draw a box around the face using the Pillow module
        # draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # # There's a bug in Pillow where it blows up with non-UTF-8 text
        # # when using the default bitmap font
        # name = name.encode("UTF-8")

        # # Draw a label with a name below the face
        # text_width, text_height = draw.textsize(name)
        # draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        # draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

        # frame = img_path
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name + str(percen)+"%", (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

    # Remove the drawing library from memory as per the Pillow docs
    # del draw

    # Display the resulting image
    # pil_image.show()
    return frame



video_capture = cv2.VideoCapture(0)
# video_capture = cv2.VideoCapture('rtsp://test:test$1234@192.168.0.73:554/Streaming/channels/202/')

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    predictions = predict(frame, model_path="trained_knn_model.clf")

    # Print results on the console
    for name, (top, right, bottom, left), percen in predictions:
        print("- Found {} at ({}, {})".format(name, left, top))

    # Display results overlaid on an image
    frame = show_prediction_labels_on_image(frame, predictions)

    # # Find all the faces and face enqcodings in the frame of video
    # face_locations = face_recognition.face_locations(rgb_frame)
    # face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # # Loop through each face in this frame of video
    # for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    #     # See if the face is a match for the known face(s)
    #     matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    #     name = "Unknown"

    #     # If a match was found in known_face_encodings, just use the first one.
    #     if True in matches:
    #         first_match_index = matches.index(True)
    #         name = known_face_names[first_match_index]

    #     # Draw a box around the face
    #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    #     # Draw a label with a name below the face
    #     cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    #     font = cv2.FONT_HERSHEY_DUPLEX
    #     cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()