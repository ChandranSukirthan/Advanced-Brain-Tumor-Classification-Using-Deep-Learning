import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import cv2
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# ── Build model architecture (must EXACTLY match training) ────────────────────
base_model = VGG19(include_top=False, input_shape=(240, 240, 3))
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model_03 = Model(base_model.inputs, output)
model_03.load_weights('vgg_unfrozen.h5')

app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')

# ── Class mapping from training (flow_from_directory alphabetical order) ──────
# Keras assigns indices alphabetically from the folder names:
#   tumorous_and_nontumorous/train/nontumorous  →  class 0
#   tumorous_and_nontumorous/train/tumorous     →  class 1
CLASS_NAMES = {
    0: "No Brain Tumor",   # nontumorous
    1: "Yes Brain Tumor",  # tumorous
}


def get_className(class_idx: int) -> str:
    """Map numeric class index to human-readable label."""
    return CLASS_NAMES.get(class_idx, "Unknown")


def getResult(img_path: str):
    """
    Preprocess image and run model inference.

    Preprocessing matches training (ImageDataGenerator rescale=1./255):
      1. Read with OpenCV  →  BGR to RGB
      2. Resize to 240×240
      3. Normalize [0,1]   ← critical: must match rescale=1./255 used during training
      4. Add batch dimension

    Returns
    -------
    class_idx   : int   (0 = No Tumor, 1 = Yes Tumor)
    confidence  : float (0–100 %)
    raw_probs   : list  [prob_class0, prob_class1]
    """
    # Read and convert colour space
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Cannot read image: {img_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    image = cv2.resize(image, (240, 240))

    # Normalise pixel values to [0, 1]  ← matches rescale=1./255 in training
    image = image.astype(np.float32) / 255.0

    # Add batch dimension → (1, 240, 240, 3)
    input_img = np.expand_dims(image, axis=0)

    # Inference
    predictions = model_03.predict(input_img, verbose=0)   # shape: (1, 2)

    raw_probs  = predictions[0].tolist()                   # [p0, p1]
    class_idx  = int(np.argmax(predictions, axis=1)[0])    # scalar Python int
    confidence = float(np.max(predictions) * 100)          # percent

    return class_idx, confidence, raw_probs


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    f = request.files['file']
    if not f or f.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded file to system temp directory to avoid Flask reload triggers
    import tempfile
    upload_folder = os.path.join(tempfile.gettempdir(), 'neuro_classify_uploads')
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, secure_filename(f.filename))
    f.save(file_path)

    try:
        class_idx, confidence, raw_probs = getResult(file_path)
        label = get_className(class_idx)

        return jsonify({
            'result':      label,
            'confidence':  round(confidence, 2),
            'class_index': class_idx,
            'probs':       [round(p * 100, 2) for p in raw_probs],  # [no_tumor%, tumor%]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
