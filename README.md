# FaceFinesseFusion (Gender-and-Age-Detection)   <img alt="GitHub" src="https://img.shields.io/github/license/smahesh29/Gender-and-Age-Detection">


<h2>Objective :</h2>
The objective of FaceFinesseFusion is to provide a user-friendly and efficient platform for face detection and analysis. By leveraging deep learning models for gender and age prediction, the project aims to offer a seamless experience for users to upload images or stream live video and receive insights about the faces present in the content. The key objectives of the project include:

1. **Accurate Face Detection**: Utilize state-of-the-art face detection models to accurately locate faces within images or video frames.
2. **Gender Prediction:** Implement a gender prediction model to classify detected faces as male or female.
3. **Age Prediction:** Employ an age prediction model to estimate the age range of each detected face.
4. **Support for Multiple Faces:** Enable the detection and analysis of multiple faces within a single image or video frame.

By achieving these objectives, FaceFinesseFusion aims to serve as a versatile tool for various use cases, including demographic analysis, security monitoring, and entertainment applications.
<h2>About the Project :</h2>
<p>In this Python Project, I had used Deep Learning to accurately identify the gender and age of a person from a single image of a face. I used the models trained by <a href="https://talhassner.github.io/home/projects/Adience/Adience-data.html">Tal Hassner and Gil Levi</a>. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, I made this a classification problem instead of making it one of regression.</p>

<h2>How to Use:</h2>
- **Upload Photos:** Users can upload images containing faces and receive detailed analysis results, including gender and age predictions for each detected face.
- **Stream Live Video (Coming Soon):** Once available, users will be able to stream live video and receive real-time insights about the faces appearing in the video feed.

<h2>Dataset :</h2>
<p>For this python project, I had used the Adience dataset; the dataset is available in the public domain and you can find it <a href="https://www.kaggle.com/ttungl/adience-benchmark-gender-and-age-classification">here</a>. This dataset serves as a benchmark for face photos and is inclusive of various real-world imaging conditions like noise, lighting, pose, and appearance. The images have been collected from Flickr albums and distributed under the Creative Commons (CC) license. It has a total of 26,580 photos of 2,284 subjects in eight age ranges (as mentioned above) and is about 1GB in size. The models I used had been trained on this dataset.</p>

<h2>The contents of this Project :</h2>
<ul>
  <li>opencv_face_detector.pbtxt</li>
  <li>opencv_face_detector_uint8.pb</li>
  <li>age_deploy.prototxt</li>
  <li>age_net.caffemodel</li>
  <li>gender_deploy.prototxt</li>
  <li>gender_net.caffemodel</li>
  <li>a few pictures to try the project on</li>
  <li>detect.py</li>
 </ul>
 <p>For face detection, we have a .pb file- this is a protobuf file (protocol buffer); it holds the graph definition and the trained weights of the model. We can use this to run the trained model. And while a .pb file holds the protobuf in binary format, one with the .pbtxt extension holds it in text format. These are TensorFlow files. For age and gender, the .prototxt files describe the network configuration and the .caffemodel file defines the internal states of the parameters of the layers.</p>
 
 <h2>Usage :</h2>
To use your FaceFinesseFusion project, follow these steps:

### 1. Clone the Repository:

If you haven't already, clone the FaceFinesseFusion repository to your local machine using Git. You can do this by running the following command in your terminal or command prompt:

```bash
git clone https://github.com/Srish0218/FaceFinesseFusion.git
```

### 2. Run the Application:

Once the dependencies are installed, you can run the application. Use the following command:

```bash
streamlit run app.py
```
## Demo
> When you deploy this project this gives error related to cv2. So please wait for same web app using flask framework

## Contributing
Contributions are welcome! If you have ideas for new features or improvements, please feel free to open an issue or submit a pull request.

## Troubleshooting
If you encounter any issues, check the Troubleshooting guide for common problems and solutions.

## License
This project is licensed under the MIT License. Feel free to use and modify it as per your requirements.

## Credits
Developed with ❤ by [SRISHTI JAITLY](https://github.com/Srish0218).
If you have questions, encounter issues, or want to discuss features, please open an issue.
