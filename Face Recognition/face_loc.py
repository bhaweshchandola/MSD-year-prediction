import face_recognition
file_name = "group.jpg"
image = face_recognition.load_image_file(file_name)
face_locations = face_recognition.face_locations(image)

print(face_locations)

import cv2
img = cv2.imread(file_name)
for top, right, bottom, left in face_locations:
    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
    # top *= 4
    # right *= 4
    # bottom *= 4
    # left *= 4
    print(top, bottom, left, right)
    #  rect = [x, y, w, h]
    #  y = top
    #  y+h = bottom
    #  x = left
    #  x+ w = right
    #  crop_image = img[x[1]:x[1]+x[3], x[0]:x[0]+x[2]]

    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

img  = cv2.resize(img, (1400,800))
cv2.imshow("as", img)
cv2.waitKey(0)
cv2.destroyAllWindows()