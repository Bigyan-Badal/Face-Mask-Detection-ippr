import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp


# =====================================
# PAGE SETTINGS
# =====================================

st.set_page_config(
    page_title="Face Mask Detection",
    layout="wide"
)


# =====================================
# LOAD TRAINED MODEL
# =====================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "models/mask_detector.keras"
    )


model = load_model()


# =====================================
# MEDIAPIPE SETTINGS
# =====================================

mp_face = mp.solutions.face_detection

IMG_SIZE = 128


# =====================================
# COLORS
# Same as detect.py
# =====================================

colors = {

    "Without Mask": (
        0,
        255,
        0
    ),

    "With Mask": (
        0,
        0,
        255
    )

}


# =====================================
# FACE MASK DETECTION FUNCTION
# Same working logic as detect.py
# =====================================

def detect_mask(
    image,
    previous_prediction=None
):

    # Create MediaPipe face detector

    face_detector = mp_face.FaceDetection(

        model_selection=0,

        min_detection_confidence=0.5

    )


    # Copy original image

    output_image = image.copy()


    # Get image height and width

    h, w, _ = image.shape


    # Convert image from BGR to RGB
    # MediaPipe requires RGB

    rgb_image = cv2.cvtColor(

        image,

        cv2.COLOR_BGR2RGB

    )


    # Detect face

    results = face_detector.process(

        rgb_image

    )


    # Store previous prediction

    current_prediction = previous_prediction


    # Check if a face is detected

    if results.detections:


        for detection in results.detections:


            # Get face bounding box

            bbox = (

                detection

                .location_data

                .relative_bounding_box

            )


            # Padding around face

            padding = 20


            # Calculate coordinates

            x = int(

                bbox.xmin * w

            ) - padding


            y = int(

                bbox.ymin * h

            ) - padding


            bw = int(

                bbox.width * w

            ) + (2 * padding)


            bh = int(

                bbox.height * h

            ) + (2 * padding)


            # Keep the face box inside image

            x = max(

                0,

                x

            )


            y = max(

                0,

                y

            )


            if x + bw > w:

                bw = w - x


            if y + bh > h:

                bh = h - y


            # Crop face

            face = image[

                y:y + bh,

                x:x + bw

            ]


            # Skip empty face

            if face.size == 0:

                continue


            # =================================
            # PREPROCESSING
            # Same as detect.py
            # =================================


            # Convert face from BGR to RGB

            face = cv2.cvtColor(

                face,

                cv2.COLOR_BGR2RGB

            )


            # Resize face

            face = cv2.resize(

                face,

                (

                    IMG_SIZE,

                    IMG_SIZE

                )

            )


            # Normalize pixel values

            face = (

                face.astype(

                    "float32"

                )

                / 255.0

            )


            # Add batch dimension

            face = np.expand_dims(

                face,

                axis=0

            )


            # =================================
            # MODEL PREDICTION
            # =================================

            raw_prediction = (

                model.predict(

                    face,

                    verbose=0

                )[0][0]

            )


            # =================================
            # PREDICTION SMOOTHING
            # Same as detect.py
            # =================================

            if current_prediction is None:

                current_prediction = 0.5


            current_prediction = (

                0.8

                * current_prediction

                +

                0.2

                * raw_prediction

            )


            prediction = current_prediction


            # =================================
            # LABEL DECISION
            # Same as detect.py
            # =================================

            if prediction >= 0.5:


                label = "Without Mask"


                confidence = (

                    prediction

                    * 100

                )


            else:


                label = "With Mask"


                confidence = (

                    1 - prediction

                ) * 100


            # Get label color

            color = colors[

                label

            ]


            # =================================
            # DRAW FACE BOX
            # =================================

            cv2.rectangle(

                output_image,

                (

                    x,

                    y

                ),

                (

                    x + bw,

                    y + bh

                ),

                color,

                2

            )


            # =================================
            # DISPLAY LABEL
            # =================================

            cv2.putText(

                output_image,

                f"{label} {confidence:.1f}%",

                (

                    x,

                    max(

                        30,

                        y - 10

                    )

                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                color,

                2

            )


    # Close MediaPipe

    face_detector.close()


    # Return result and prediction

    return (

        output_image,

        current_prediction

    )


# =====================================
# SIDEBAR
# =====================================

st.sidebar.title(

    "Face Mask Detection"

)


page = st.sidebar.radio(

    "Menu",

    [

        "Home",

        "Image Detection",

        "Webcam Detection"

    ]

)


# =====================================
# HOME PAGE
# No large gap between text and image
# =====================================

if page == "Home":


    st.markdown(

        """
        <div style="
        text-align: center;
        margin: 0;
        padding: 0;
        ">

        <h1 style="
        margin: 0;
        padding: 0;
        ">
        Welcome to Face Mask Detection System
        </h1>

        <p style="
        font-size: 20px;
        color: gray;
        margin: 0;
        padding: 0;
        ">
        Using Deep Learning, Convolutional Neural Network (CNN),
        TensorFlow, OpenCV, MediaPipe, and Streamlit.
        </p>

        </div>
        """,

        unsafe_allow_html=True

    )


    # Image starts immediately after the text

    st.image(

        "face_mask_figure.png",

        use_container_width=True

    )


# =====================================
# IMAGE DETECTION
# =====================================

elif page == "Image Detection":


    st.header(

        "Image Detection"

    )


    uploaded_file = st.file_uploader(

        "Choose an image",

        type=[

            "jpg",

            "jpeg",

            "png",

            "jfif"

        ]

    )


    if uploaded_file is not None:


        # Read uploaded image

        file_bytes = np.asarray(

            bytearray(

                uploaded_file.read()

            ),

            dtype=np.uint8

        )


        # Decode image

        image = cv2.imdecode(

            file_bytes,

            cv2.IMREAD_COLOR

        )


        if image is not None:


            # Detect mask

            result, _ = detect_mask(

                image,

                previous_prediction=None

            )


            # Convert BGR to RGB

            result_rgb = cv2.cvtColor(

                result,

                cv2.COLOR_BGR2RGB

            )


            # Display result

            st.image(

                result_rgb,

                use_container_width=True

            )


        else:


            st.error(

                "Image could not be opened."

            )


# =====================================
# WEBCAM DETECTION
# =====================================

elif page == "Webcam Detection":


    st.header(

        "Webcam Detection"

    )


    start_camera = st.checkbox(

        "Start Webcam"

    )


    camera_view = st.image(

        []

    )


    if start_camera:


        # Open webcam

        cap = cv2.VideoCapture(

            0

        )


        # Check webcam

        if not cap.isOpened():


            st.error(

                "Cannot open webcam."

            )


        else:


            # Stop button

            stop_button = st.button(

                "Stop Webcam"

            )


            # Initial prediction
            # Same as detect.py

            last_prediction = 0.5


            # Webcam loop

            while (

                start_camera

                and

                not stop_button

            ):


                success, frame = cap.read()


                if not success:


                    st.error(

                        "Cannot read webcam."

                    )


                    break


                # Detect mask

                result, last_prediction = (

                    detect_mask(

                        frame,

                        previous_prediction=

                        last_prediction

                    )

                )


                # Convert BGR to RGB

                result_rgb = cv2.cvtColor(

                    result,

                    cv2.COLOR_BGR2RGB

                )


                # Display webcam output

                camera_view.image(

                    result_rgb,

                    use_container_width=True

                )


            # Release webcam

            cap.release()