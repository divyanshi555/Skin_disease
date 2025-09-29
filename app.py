import os
import numpy as np
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras import layers

# ---------------------------
# Flask app initialization
# ---------------------------
app = Flask(__name__)

# ---------------------------
# Load the model
# ---------------------------
MODEL_PATH = "best_skin_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class names
class_names = ['Acne', 'Normal Skin', 'Ringworm']

# TTA augmentation
tta_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# ---------------------------
# Disease information
# ---------------------------
disease_info = {
    "Normal Skin": {
        "description": """Your skin appears healthy with no visible acne, redness, or infection.""",
        "treatment": """No medical treatment is required. Maintain regular skincare routines.""",
        "remedy": """- Cleanse your face twice daily
- Stay hydrated
- Use sunscreen to protect from UV damage
- Eat a balanced diet and get enough sleep"""
    },
    "Acne": {
        "description": """Acne is a common skin condition caused by clogged hair follicles 
with oil, bacteria, and dead skin cells. Appears as pimples, blackheads, or whiteheads.""",
        "treatment": """- Over-the-counter topical creams like benzoyl peroxide, salicylic acid
- Dermatologist-prescribed treatments for severe cases (e.g., retinoids, antibiotics)""",
        "remedy": """- Wash your face gently twice a day
- Avoid touching or popping pimples
- Limit oily and sugary foods
- Natural remedies: aloe vera gel, tea tree oil"""
    },
    "Ringworm": {
        "description": """Ringworm is a contagious fungal infection that causes red, circular, itchy patches on the skin.""",
        "treatment": """- Topical antifungal creams (e.g., clotrimazole, miconazole)
- Oral antifungal medication if infection is widespread or severe""",
        "remedy": """- Keep affected areas clean and dry
- Avoid sharing personal items like towels or clothes
- Natural remedies: turmeric paste, coconut oil
- Wash hands frequently to prevent spreading"""
    }
}


# ---------------------------
# Image classification function
# ---------------------------
def classify_image(image_path, model, class_names, tta_steps=1):
    """
    Predict the class of a single image and return only the predicted class.
    """
    # Load image and convert to array
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array_exp = tf.expand_dims(img_array, axis=0)  # shape (1,224,224,3)

    # Collect predictions
    preds = []
    for _ in range(tta_steps):
        if tta_steps > 1:
            aug_img = tta_augmentation(img_array_exp, training=True)
            pred = model.predict(aug_img, verbose=0)
        else:
            pred = model.predict(img_array_exp, verbose=0)
        preds.append(pred)

    # Average predictions if using TTA
    preds = np.mean(preds, axis=0)[0]
    predicted_class = class_names[np.argmax(preds)]

    return predicted_class


# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict')
def predict_page():
    return render_template("predict.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Save uploaded file
    upload_folder = os.path.join("static", "uploads")
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    # Classify the image
    predicted_class= classify_image(file_path, model, class_names, tta_steps=5)

    # Get disease info
    info = disease_info[predicted_class]

    return render_template(
        "result.html",
        prediction=predicted_class,
        image_name="uploads/" + file.filename,
        description=info["description"],
        treatment=info["treatment"],
        remedy=info["remedy"]
    )

# ---------------------------
# Run Flask app
# ---------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render assigns a port
    app.run(host="0.0.0.0", port=port)

