import streamlit as st
import cv2
import numpy as np


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes


def main():
    st.title("Face Finesse Fusion App")

    # Load face detection model
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    faceNet = cv2.dnn.readNet(faceModel, faceProto)

    # Load age and gender prediction models
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    # Streamlit sidebar
    option = st.selectbox("Choose input source", ("Live Video (Coming Soon)", "Upload Photos"))

    if option == "Live Video (Coming Soon)":
        st.subheader("Live Video Stream")
        st.info("Live Video feature is coming soon. Stay tuned!")

    elif option == "Upload Photos":
        st.subheader("Upload Photos")
        uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, 1)
                resultImg, faceBoxes = highlightFace(faceNet, frame)

                gender_preds = []
                age_preds = []

                for faceBox in faceBoxes:
                    face = frame[max(0, faceBox[1] - 20):
                                 min(faceBox[3] + 20, frame.shape[0] - 1), max(0, faceBox[0] - 20):min(faceBox[2] + 20,
                                 frame.shape[1] - 1)]

                    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                    genderNet.setInput(blob)
                    genderPreds = genderNet.forward()
                    gender = genderList[genderPreds[0].argmax()]
                    gender_preds.append(gender)

                    ageNet.setInput(blob)
                    agePreds = ageNet.forward()
                    age = ageList[agePreds[0].argmax()]
                    age_preds.append(age)

                    cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 255), 2, cv2.LINE_AA)

                c1, c2 = st.columns(2)
                with c1:
                    st.image(uploaded_file, caption="Original Image")
                with c2:
                    st.image(resultImg, channels="BGR", caption="Processed Image")
                    for i in range(len(gender_preds)):
                        st.write(f"Face {i + 1}:")
                        st.write("Gender: ", gender_preds[i])
                        st.write("Age: ", age_preds[i])

footer = """<style>
a:link , a:visited{
color: black;
font-weight: bold;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}


.footer a {
    color: #007bff;
    text-decoration: none;
    font-weight: bold;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
color: white;
text-align: center;
z-index:100;
background: rgba(255, 255, 255, 0.05);
backdrop-filter: blur(90%);
border-radius: 10px;
box-shadow: 0 8px 32px 0 rgba(255, 255, 255, 0.15);
backdrop-filter: blur( 4px );
-webkit-backdrop-filter: blur( 4px );
        }
}
</style>

<div class="footer">
<p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://github.com/Srish0218" target="_blank">Srishti Jaitly 🌸</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
