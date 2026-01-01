from flask import Flask, request, jsonify
import mysql.connector
import base64
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
print("Received POST request to /recognize")


def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="aslalphabetdata"
    )

def get_sign_coordinates(letter):
    db_connection = connect_to_db()
    cursor = db_connection.cursor()
    cursor.execute("SELECT * FROM aslalphabet WHERE letter = %s", (letter,))
    result = cursor.fetchone()
    cursor.close()
    db_connection.close()

    if result:
        return result[2:]
    return None

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.json
    image_data = base64.b64decode(data['image'].split(',')[1])

    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            letter = compare_hand_to_database(hand_landmarks, "A")  # Example for letter 'A'
            if letter:
                return jsonify({"letter": letter})
    
    return jsonify({"letter": "Unknown"})

def compare_hand_to_database(hand_landmarks, letter):
    height, width, _ = hand_landmarks.shape
    normalized_coords = [(landmark.x * width, landmark.y * height) for landmark in hand_landmarks.landmark]

    stored_coords = get_sign_coordinates(letter)
    if not stored_coords:
        return None

    for i in range(0, len(stored_coords), 2):
        real_x, real_y = normalized_coords[i // 2]
        stored_x, stored_y = stored_coords[i], stored_coords[i + 1]
        distance = ((real_x - stored_x) ** 2 + (real_y - stored_y) ** 2) ** 0.5
        if distance > 0.05:
            return None
    
    return letter

if __name__ == '__main__':
    app.run(debug=True)
