from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2
import base64

# Opencv pre-trained SVM with HOG people features
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# Alternative detector fixme doesn't work: assertion error
# HOGCV.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())

def detector(image):
    """
    @image is a numpy array
    """

    image = imutils.resize(image, width=min(400, image.shape[1]))
    # clone = image.copy()

    (rects, weights) = HOGCV.detectMultiScale(image, winStride=(8, 8),
                                              padding=(32, 32), scale=1.05)

    # Applies non-max supression from imutils package to kick-off overlapped
    # boxes
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    # rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    return rects


def cameraDetect():
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = imutils.resize(frame, width=min(400, frame.shape[1]))
        result = detector(frame.copy())

        # shows the result
        for (xA, yA, xB, yB) in result:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (255, 255, 255), 2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cameraDetect()