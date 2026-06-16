# 1st code
from flask import Flask, request, jsonify, render_template
import numpy as np
from keras.models import load_model
import cv2
from io import BytesIO
import base64
from PIL import Image

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model (1).h5')

# Define a dictionary to map predicted labels to expression names
expressions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Recommended posts for each emotion
recommended_posts = {
    'Angry': {
        'instagram': 'https://www.instagram.com/explore/tags/angry/',
        'twitter': 'https://twitter.com/search?q=angry',
        'youtube': 'https://www.youtube.com/results?search_query=angry'
    },
    'Disgust': {
        'instagram': 'https://www.instagram.com/explore/tags/disgusting/',
        'twitter': 'https://twitter.com/search?q=disgusting',
        'youtube': 'https://www.youtube.com/results?search_query=disgusting'
    },
    'Fear': {
        'instagram': 'https://www.instagram.com/explore/tags/fear/',
        'twitter': 'https://twitter.com/search?q=fear',
        'youtube': 'https://www.youtube.com/results?search_query=fear'
    },
    'Happy': {
        'instagram': 'https://www.instagram.com/explore/tags/happyposts/',
        'twitter': 'https://twitter.com/search?q=happy',
        'youtube': 'https://www.youtube.com/results?search_query=happy'
    },
    'Sad': {
        'instagram': 'https://www.instagram.com/explore/tags/sadposts/',
        'twitter': 'https://twitter.com/search?q=sad',
        'youtube': 'https://www.youtube.com/results?search_query=sad'
    },
    'Surprise': {
        'instagram': 'https://www.instagram.com/explore/tags/surprise/',
        'twitter': 'https://twitter.com/search?q=surprise',
        'youtube': 'https://www.youtube.com/results?search_query=surprise'
    },
    'Neutral': {
        'instagram': 'https://www.instagram.com/explore/tags/neutral/',
        'twitter': 'https://twitter.com/search?q=neutral',
        'youtube': 'https://www.youtube.com/results?search_query=neutral'
    }
}

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/detect', methods=['POST'])
def detect_emotion():
    # Get the image and platform data from the request
    data = request.get_json()
    image_data = data.get('image')
    platform = data.get('platform')  # Get the selected platform

    # Convert base64 string to image
    image_data = image_data.split(',')[1]  # Remove the data URL part
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes)).convert('RGB')

    # Convert image to OpenCV format
    image = np.array(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return jsonify({'emotion': 'No face detected', 'percentage': 0, 'post_url': ''})

    # Process the first detected face
    (x, y, w, h) = faces[0]
    face = gray_image[y:y + h, x:x + w]
    resized_face = cv2.resize(face, (48, 48))
    three_channel_face = cv2.cvtColor(resized_face, cv2.COLOR_GRAY2RGB)
    normalized_face = three_channel_face / 255.0
    input_face = np.expand_dims(normalized_face, axis=0)

    # Predict the emotion
    predictions = model.predict(input_face)
    max_index = np.argmax(predictions[0])
    emotion = expressions[max_index]
    percentage = round(predictions[0][max_index] * 100, 2)

    # Get the recommended post URL based on the selected platform
    post_url = recommended_posts.get(emotion, {}).get(platform, '')

    return jsonify({
        'emotion': emotion,
        'percentage': percentage,
        'post_url': post_url
    })

if __name__ == '__main__':
    app.run(debug=True)