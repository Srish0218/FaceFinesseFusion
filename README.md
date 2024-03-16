# Gender-and-Age-Detection   <img alt="GitHub" src="https://img.shields.io/github/license/smahesh29/Gender-and-Age-Detection">


<h2>Objective :</h2>
The objective of FaceFinesseFusion is to provide a user-friendly and efficient platform for face detection and analysis. By leveraging deep learning models for gender and age prediction, the project aims to offer a seamless experience for users to upload images or stream live video and receive insights about the faces present in the content. The key objectives of the project include:

1. **Accurate Face Detection**: Utilize state-of-the-art face detection models to accurately locate faces within images or video frames.
2. **Gender Prediction:** Implement a gender prediction model to classify detected faces as male or female.
3. **Age Prediction:** Employ an age prediction model to estimate the age range of each detected face.
4. **Support for Multiple Faces:** Enable the detection and analysis of multiple faces within a single image or video frame.
5. **User-Friendly Interface:** Develop an intuitive user interface that allows users to easily upload images, stream video, and view analysis results.
6. **Scalability and Performance:** Design the application to handle a large volume of images or video frames efficiently while maintaining high accuracy in face detection and analysis.

By achieving these objectives, FaceFinesseFusion aims to serve as a versatile tool for various use cases, including demographic analysis, security monitoring, and entertainment applications.
<h2>About the Project :</h2>
<p>In this Python Project, I had used Deep Learning to accurately identify the gender and age of a person from a single image of a face. I used the models trained by <a href="https://talhassner.github.io/home/projects/Adience/Adience-data.html">Tal Hassner and Gil Levi</a>. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, I made this a classification problem instead of making it one of regression.</p>

<h2>How to Use:</h2>
- **Upload Photos:** Users can upload images containing faces and receive detailed analysis results, including gender and age predictions for each detected face.
- **Stream Live Video (Coming Soon):** Once available, users will be able to stream live video and receive real-time insights about the faces appearing in the video feed.

<h2>Dataset :</h2>
<p>For this python project, I had used the Adience dataset; the dataset is available in the public domain and you can find it <a href="https://www.kaggle.com/ttungl/adience-benchmark-gender-and-age-classification">here</a>. This dataset serves as a benchmark for face photos and is inclusive of various real-world imaging conditions like noise, lighting, pose, and appearance. The images have been collected from Flickr albums and distributed under the Creative Commons (CC) license. It has a total of 26,580 photos of 2,284 subjects in eight age ranges (as mentioned above) and is about 1GB in size. The models I used had been trained on this dataset.</p>

<h2>Additional Python Libraries Required :</h2>
<ul>
  <li>OpenCV</li>
  
       pip install opencv-python
</ul>
<ul>
 <li>argparse</li>
  
       pip install argparse
</ul>

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
git clone <repository_url>
```

Replace `<repository_url>` with the URL of your GitHub repository.


### 3. Run the Application:

Once the dependencies are installed, you can run the application. Use the following command:

```bash
streamlit run app.py
```

This command will start the Streamlit application, and you will see the application interface in your web browser.

### 4. Choose Input Source:

In the application interface, you'll see options to choose the input source. You can select either "Upload Photos" or "Live Video (Coming Soon)".

- **Upload Photos**: Click on this option to upload one or more images containing faces. Once uploaded, the system will analyze the images and provide insights such as gender and age predictions for each detected face.

- **Live Video (Coming Soon)**: This option will be available soon. Stay tuned for updates on this feature.

### 5. Analyze Results:

After selecting the input source and uploading images (if applicable), the application will display the processed images with bounding boxes around detected faces. Additionally, it will provide detailed information about each detected face, including gender and age predictions.

### 6. Explore Additional Features:

Feel free to explore additional features and functionalities of the application, such as uploading different types of images, experimenting with various lighting conditions, and analyzing multiple faces in a single image.

### 7. Provide Feedback:

If you have any feedback, suggestions, or issues with the application, you can provide them directly within the application interface or by reaching out to the project maintainers.

By following these steps, you can effectively use your FaceFinesseFusion project to analyze faces in images and potentially live video streams, gaining valuable insights into the gender and age of detected individuals.

Please note that some features may still be under development, so be sure to check for updates and improvements regularly.

# Working:
[![Watch the video](https://img.youtube.com/vi/ReeccRD21EU/0.jpg)](https://youtu.be/ReeccRD21EU)

<h2>Examples :</h2>
<p><b>NOTE:- I downloaded the images from Google,if you have any query or problem i can remove them, i just used it for Educational purpose.</b></p>

    >python detect.py --image girl1.jpg
    Gender: Female
    Age: 25-32 years
    
<img src="Example/Detecting age and gender girl1.png">

    >python detect.py --image girl2.jpg
    Gender: Female
    Age: 8-12 years
    
<img src="Example/Detecting age and gender girl2.png">

    >python detect.py --image kid1.jpg
    Gender: Male
    Age: 4-6 years    
    
<img src="Example/Detecting age and gender kid1.png">

    >python detect.py --image kid2.jpg
    Gender: Female
    Age: 4-6 years  
    
<img src="Example/Detecting age and gender kid2.png">

    >python detect.py --image man1.jpg
    Gender: Male
    Age: 38-43 years
    
<img src="Example/Detecting age and gender man1.png">

    >python detect.py --image man2.jpg
    Gender: Male
    Age: 25-32 years
    
<img src="Example/Detecting age and gender man2.png">

    >python detect.py --image woman1.jpg
    Gender: Female
    Age: 38-43 years
    
<img src="Example/Detecting age and gender woman1.png">
              
