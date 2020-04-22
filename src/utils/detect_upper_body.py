import cv2
import imutils
import matplotlib.pyplot as plt

def get_upper_bodies(image):
    haar_upper_body_cascade = cv2.CascadeClassifier("src/utils/haarcascade_upperbody.xml") 

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert video to grayscale

    upper_body = haar_upper_body_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (50, 100), # Min size for valid detection, changes according to video size or body size in the video.
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the upper bodies
    ubodies = []
    for (x, y, w, h) in upper_body:
        crop_img = image[y:y+h, x:x+w]
        ubodies.append(crop_img)

    return ubodies
            
###########################################################################################
###########################################################################################

def real_time_upper_body():

    haar_upper_body_cascade = cv2.CascadeClassifier("src/utils/haarcascade_upperbody.xml")

    # Uncomment this for real-time webcam detection
    # video_capture = cv2.VideoCapture(0)

    # For real-time sample video detection
    video_capture = cv2.VideoCapture("data/subway.mp4")
    video_width = video_capture.get(3)
    video_height = video_capture.get(4)

    while True:
        ret, frame = video_capture.read()

        frame = imutils.resize(frame, width=1000) # resize original video for better viewing performance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert video to grayscale

        upper_body = haar_upper_body_cascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (50, 100), # Min size for valid detection, changes according to video size or body size in the video.
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the upper bodies
        for (x, y, w, h) in upper_body:
            crop_img = frame[y:y+h, x:x+w]
            # plt.imshow(crop_img)
            # plt.show()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1) # creates green color rectangle with a thickness size of 1
            cv2.putText(frame, "Upper Body Detected", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # creates green color text with text size of 0.5 & thickness size of 2
        cv2.imshow('Video', frame) # Display video

        # stop script when "q" key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release capture
    video_capture.release()
    cv2.destroyAllWindows()

real_time_upper_body()