import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp


# ----------------------------
# Load Trained Model
# ----------------------------

model = tf.keras.models.load_model(
    "models/mask_detector.keras"
)


# ----------------------------
# MediaPipe Face Detection
# ----------------------------

mp_face = mp.solutions.face_detection

face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)


# ----------------------------
# Image Size
# ----------------------------

IMG_SIZE = 128


# ----------------------------
# Colors
# ----------------------------

colors = {
    "Without Mask": (0, 255, 0),
    "With Mask": (0, 0, 255)
}


# ----------------------------
# Prediction Smoothing
# ----------------------------

last_prediction = 0.5


# ----------------------------
# Webcam
# ----------------------------

cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Cannot open webcam")
    exit()



# ----------------------------
# Detection Loop
# ----------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break


    h, w, _ = frame.shape


    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    results = face_detector.process(rgb)



    if results.detections:


        for detection in results.detections:


            bbox = detection.location_data.relative_bounding_box


            padding = 20


            x = int(bbox.xmin * w) - padding
            y = int(bbox.ymin * h) - padding

            bw = int(bbox.width * w) + (2 * padding)
            bh = int(bbox.height * h) + (2 * padding)



            # Keep inside frame

            x = max(0, x)
            y = max(0, y)


            if x+bw > w:
                bw = w-x


            if y+bh > h:
                bh = h-y



            face = frame[
                y:y+bh,
                x:x+bw
            ]



            if face.size == 0:
                continue



            # ----------------------------
            # Preprocessing
            # ----------------------------

            face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )


            face = cv2.resize(
                face,
                (IMG_SIZE, IMG_SIZE)
            )


            face = face.astype(
                "float32"
            ) / 255.0


            face = np.expand_dims(
                face,
                axis=0
            )



            # ----------------------------
            # Prediction
            # ----------------------------

            prediction = model.predict(
                face,
                verbose=0
            )[0][0]


            # Smooth prediction

            last_prediction = (
                0.8 * last_prediction +
                0.2 * prediction
            )


            prediction = last_prediction



            print(
                "RAW OUTPUT:",
                round(float(prediction),3)
            )



            # ----------------------------
            # Label Decision
            # ----------------------------
            #
            # If output is opposite,
            # swap labels here
            # ----------------------------


            if prediction >= 0.5:

                label = "Without Mask"
                confidence = prediction * 100


            else:

                label = "With Mask"
                confidence = (1-prediction) * 100




            color = colors[label]



            # ----------------------------
            # Draw Rectangle
            # ----------------------------


            cv2.rectangle(
                frame,
                (x,y),
                (x+bw,y+bh),
                color,
                2
            )



            cv2.putText(
                frame,
                f"{label} {confidence:.1f}%",
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )



    cv2.imshow(
        "Face Mask Detection",
        frame
    )



    key = cv2.waitKey(1) & 0xff


    if key == ord("q"):
        break



# ----------------------------
# Release
# ----------------------------

cap.release()

cv2.destroyAllWindows()

face_detector.close()