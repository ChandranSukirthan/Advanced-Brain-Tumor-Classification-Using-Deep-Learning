"""
Diagnostic script: Tests the loaded model against augmented_data/yes and augmented_data/no
to determine actual accuracy and reveal if class labels are swapped.
"""
import os
import sys
import numpy as np
import cv2

# ── Force TF to be quiet ─────────────────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# ── Build model ───────────────────────────────────────────────────────────────
print("Loading model...")
base_model = VGG19(include_top=False, input_shape=(240, 240, 3))
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model = Model(base_model.inputs, output)
model.load_weights('vgg_unfrozen.h5')
print("Model loaded.\n")

# ── Preprocessing functions ───────────────────────────────────────────────────
def preprocess_rescale(img_path):
    """Matches training: rescale=1./255"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (240, 240))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def preprocess_raw(img_path):
    """No normalization (raw 0-255)"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (240, 240))
    img = img.astype(np.float32)
    return np.expand_dims(img, axis=0)

# ── Test both preprocessing methods on a sample of images ────────────────────
aug_no_dir  = 'augmented_data/no'
aug_yes_dir = 'augmented_data/yes'

no_files  = [f for f in os.listdir(aug_no_dir)  if f.endswith('.jpg')][:30]
yes_files = [f for f in os.listdir(aug_yes_dir) if f.endswith('.jpg')][:30]

print("=" * 65)
print(f"{'METHOD':<20} {'TRUE CLASS':<14} {'CORRECT':<10} {'ACCURACY'}")
print("=" * 65)

for preprocess_fn, name in [(preprocess_rescale, "rescale /255"), (preprocess_raw, "raw (no norm)")]:
    correct_no = 0
    correct_yes = 0

    for fname in no_files:
        path = os.path.join(aug_no_dir, fname)
        try:
            inp = preprocess_fn(path)
            pred = model.predict(inp, verbose=0)
            class_idx = int(np.argmax(pred, axis=1)[0])
            if class_idx == 0:   # 0 = nontumorous = correct for no-tumor images
                correct_no += 1
        except Exception as e:
            pass

    for fname in yes_files:
        path = os.path.join(aug_yes_dir, fname)
        try:
            inp = preprocess_fn(path)
            pred = model.predict(inp, verbose=0)
            class_idx = int(np.argmax(pred, axis=1)[0])
            if class_idx == 1:   # 1 = tumorous = correct for yes-tumor images
                correct_yes += 1
        except Exception as e:
            pass

    acc_no  = correct_no  / len(no_files)  * 100
    acc_yes = correct_yes / len(yes_files) * 100
    acc_all = (correct_no + correct_yes) / (len(no_files) + len(yes_files)) * 100

    print(f"{name:<20} {'No Tumor (0)':<14} {correct_no}/{len(no_files):<6}   {acc_no:.1f}%")
    print(f"{name:<20} {'Yes Tumor (1)':<14} {correct_yes}/{len(yes_files):<6}   {acc_yes:.1f}%")
    print(f"{name:<20} {'OVERALL':<14} {correct_no+correct_yes}/{len(no_files)+len(yes_files):<6}   {acc_all:.1f}%")
    print("-" * 65)

print("\nDone. Use the preprocessing method with higher overall accuracy in app.py")
print("If 'No Tumor' accuracy is ~0% and 'Yes Tumor' ~100%, labels are SWAPPED (class 0=tumor, 1=no-tumor)")
