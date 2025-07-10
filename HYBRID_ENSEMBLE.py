import os
import cv2
import numpy as np
from tqdm import tqdm

IMAGE_SIZE = 299  # Required for Inception and EfficientNet

def load_images_from_folders(folder1, folder2, label1=0, label2=1):
    images = []
    labels = []

    # Folder 1 (e.g., carbide)
    for file in tqdm(os.listdir(folder1), desc="Loading Carbide"):
        path = os.path.join(folder1, file)
        try:
            img = cv2.imread(path)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            images.append(img)
            labels.append(label1)
        except:
            print(f"Failed to load {path}")

    # Folder 2 (e.g., non-carbide)
    for file in tqdm(os.listdir(folder2), desc="Loading Non-Carbide"):
        path = os.path.join(folder2, file)
        try:
            img = cv2.imread(path)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            images.append(img)
            labels.append(label2)
        except:
            print(f"Failed to load {path}")

    return np.array(images), np.array(labels)
carbide_path = r"C:\Users\student\Desktop\Dataset\data\with_carbide"
noncarbide_path = r"C:\Users\student\Desktop\Dataset\data\without_carbide"
X_images, y_labels = load_images_from_folders(carbide_path, noncarbide_path)
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
import cv2
import numpy as np

def preprocess_images(X_images):
    X_incep = []
    X_eff = []

    for img in X_images:
        img_resized = cv2.resize(img, (299, 299))
        X_incep.append(inception_preprocess(img_resized))
        X_eff.append(efficientnet_preprocess(img_resized))

    return np.array(X_incep), np.array(X_eff)
X_incep, X_eff = preprocess_images(X_images)
from tensorflow.keras.applications import InceptionV3, EfficientNetB0

# Load pre-trained models
inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_shape=(299, 299, 3))
efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(299, 299, 3))

# Preprocess input
X_incep, X_eff = preprocess_images(X_images)

# Extract features
features_incep = inception_model.predict(X_incep, verbose=1)
features_eff = efficientnet_model.predict(X_eff, verbose=1)

# Concatenate features
X_features_hybrid = np.concatenate([features_incep, features_eff], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_features_hybrid, y_labels, test_size=0.2, random_state=42, stratify=y_labels)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=150, max_depth=6)
knn = KNeighborsClassifier(n_neighbors=5)
logreg = LogisticRegression(max_iter=1000)
gb = GradientBoostingClassifier(n_estimators=100)
# Stack them into an ensemble
estimators = [
    ('xgb', xgb),
   
    ('knn', knn),
    ('logreg', logreg),
    ('gb', gb)
]

# Final estimator that learns from base predictions
final_estimator = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

stacked_model = StackingClassifier(estimators=estimators, final_estimator=final_estimator, passthrough=True)
stacked_model.fit(X_train, y_train)
import pickle
with open("stacked_model.pkl","wb") as f:
    pickle.dump(stacked_model,f)
from sklearn.metrics import classification_report, accuracy_score

y_pred = stacked_model.predict(X_test)

print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Carbide', 'Non-Carbide']))

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# Label mapping
label_map = {0: 'with carbide', 1: 'without carbide'}

def preprocess_single_image(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # For displaying with matplotlib
    img_resized = cv2.resize(img, (299, 299))

    # Expand dims and preprocess
    incep_input = inception_preprocess(np.expand_dims(img_resized.copy(), axis=0))
    eff_input = efficientnet_preprocess(np.expand_dims(img_resized.copy(), axis=0))

    return incep_input, eff_input, img_rgb

def predict_and_display(img_path):
    incep_input, eff_input, img_rgb = preprocess_single_image(img_path)

    # Extract features
    feature_incep = inception_model.predict(incep_input)
    feature_eff = efficientnet_model.predict(eff_input)
    hybrid_features = np.concatenate([feature_incep, feature_eff], axis=1)

    # Get prediction
    prediction = stacked_model.predict(hybrid_features)[0]
    prediction_proba = stacked_model.predict_proba(hybrid_features)[0]
    predicted_label = label_map[prediction]

    # Display image with prediction
    plt.figure(figsize=(6, 4))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f'Predicted: {predicted_label}\nConfidence: {prediction_proba[prediction]:.2%}', fontsize=14)
    plt.tight_layout()
    plt.show()

    # Optional: print probabilities for both classes
    print("Confidence Scores:")
    for i, prob in enumerate(prediction_proba):
        print(f"{label_map[i]}: {prob:.2%}")

# Example usage
image_path = r"C:\Users\student\Downloads\p_banana.jpg"
predict_and_display(image_path)
