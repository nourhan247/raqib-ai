import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


# ===========================
# Configuration
# ===========================

IMG_SIZE = (260, 260)

CLASS_NAMES = [
    "Damaged Road",
    "Normal Road",
    "Damaged Home",
    "Normal Building",
    "Big Trash",
    "Small Trash"
]

MODEL_PATH = "disaster_6class_model_final.keras"

model = tf.keras.models.load_model(MODEL_PATH)


# ===========================
# Image Preprocessing
# ===========================

def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)
    image = np.array(image).astype("float32")

    # Same preprocessing used during training
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return image


# ===========================
# Road Analysis
# ===========================

def analyze_road(image_path):

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blur, 60, 170)

    # Connect nearby cracks
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find damaged regions
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    total_area = 0
    largest_pothole = 0
    crack_length = 0
    crack_count = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        # Ignore tiny noise
        if area < 30:
            continue

        crack_count += 1

        total_area += area

        largest_pothole = max(largest_pothole, area)

        crack_length += cv2.arcLength(cnt, False)

    image_area = image.shape[0] * image.shape[1]

    damage_percentage = (total_area / image_area) * 100

    pothole_percentage = (largest_pothole / image_area) * 100

    crack_density = crack_length / image_area

    # Texture roughness
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture = laplacian.var()

    # Normalize each feature
    damage_score = min(damage_percentage * 4, 100)

    pothole_score = min(pothole_percentage * 10, 100)

    crack_score = min(crack_density * 6000, 100)

    texture_score = min(texture / 15, 100)

    count_score = min(crack_count * 3, 100)

    # Final severity
    severity_score = (
        0.35 * damage_score +
        0.25 * pothole_score +
        0.20 * crack_score +
        0.10 * texture_score +
        0.10 * count_score
    )

    return {
        "damage_percentage": round(damage_percentage, 2),
        "largest_pothole_percentage": round(pothole_percentage, 2),
        "crack_length": round(crack_length, 2),
        "crack_count": crack_count,
        "texture_score": round(texture, 2),
        "severity_score": round(min(severity_score, 100), 2)
    }


# ===========================
# Building Analysis
# ===========================

def analyze_building(image_path):

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Detect possible damaged regions
    _, mask = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    total_area = 0
    largest_region = 0
    contour_complexity = 0
    region_count = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 50:
            continue

        region_count += 1

        total_area += area

        largest_region = max(largest_region, area)

        perimeter = cv2.arcLength(cnt, True)

        if area > 0:
            contour_complexity += (perimeter ** 2) / (4 * np.pi * area)

    image_area = image.shape[0] * image.shape[1]

    damage_percentage = (total_area / image_area) * 100

    largest_damage_percentage = (largest_region / image_area) * 100

    # Texture roughness
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture = laplacian.var()

    # Brightness (collapsed buildings tend to be darker)
    brightness = np.mean(gray)

    # -------------------------
    # Normalize features
    # -------------------------

    damage_score = min(damage_percentage * 3, 100)

    largest_score = min(largest_damage_percentage * 8, 100)

    texture_score = min(texture / 20, 100)

    complexity_score = min(contour_complexity * 5, 100)

    brightness_score = max(0, (150 - brightness) / 150 * 100)

    # -------------------------
    # Final severity score
    # -------------------------

    severity_score = (
        0.40 * damage_score +
        0.20 * largest_score +
        0.20 * texture_score +
        0.10 * complexity_score +
        0.10 * brightness_score
    )

    return {
        "damage_percentage": round(damage_percentage, 2),
        "largest_damage_percentage": round(largest_damage_percentage, 2),
        "damaged_regions": region_count,
        "texture_score": round(texture, 2),
        "contour_complexity": round(contour_complexity, 2),
        "brightness": round(brightness, 2),
        "severity_score": round(min(severity_score, 100), 2)
    }


# ===========================
# Trash Analysis
# ===========================

def analyze_trash(image_path):

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Automatic threshold
    _, mask = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((5,5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    total_area = 0
    largest_object = 0
    object_count = 0

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 40:
            continue

        object_count += 1

        total_area += area

        largest_object = max(largest_object, area)

    image_area = image.shape[0] * image.shape[1]

    debris_percentage = (total_area / image_area) * 100

    largest_object_percentage = (largest_object / image_area) * 100

    if object_count > 0:
        average_size = total_area / object_count
    else:
        average_size = 0

    # -------------------------
    # Normalize features
    # -------------------------

    debris_score = min(debris_percentage * 4, 100)

    largest_score = min(largest_object_percentage * 8, 100)

    object_score = min(object_count * 5, 100)

    average_score = min(average_size / 300, 100)

    # -------------------------
    # Final severity score
    # -------------------------

    severity_score = (
        0.45 * debris_score +
        0.25 * largest_score +
        0.20 * object_score +
        0.10 * average_score
    )

    return {
        "debris_percentage": round(debris_percentage, 2),
        "largest_object_percentage": round(largest_object_percentage, 2),
        "object_count": object_count,
        "average_object_size": round(average_size, 2),
        "severity_score": round(min(severity_score, 100), 2)
    }


# ===========================
# Prediction
# ===========================

def predict_image(image_path):

    image = preprocess_image(image_path)

    probabilities = model.predict(image, verbose=0)[0]

    predicted_index = np.argmax(probabilities)

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(probabilities[predicted_index])

    # -----------------------
    # Dynamic Severity
    # -----------------------

    if predicted_class == "Damaged Road":

        metrics = analyze_road(image_path)
        damage_percentage = metrics["damage_percentage"]
        severity_score = metrics["severity_score"]

    elif predicted_class == "Damaged Home":

        metrics = analyze_building(image_path)
        damage_percentage = metrics["damage_percentage"]
        severity_score = metrics["severity_score"]

    elif predicted_class in ["Big Trash", "Small Trash"]:

        metrics = analyze_trash(image_path)
        damage_percentage = metrics["debris_percentage"]
        severity_score = metrics["severity_score"]

    else:

        metrics = {}
        damage_percentage = 0
        severity_score = 0

    # Severity label



    if severity_score < 40:
        severity_label = "Low"

    elif severity_score < 70:
        severity_label = "Medium"

    else:
        severity_label = "High"

    return {

        "predicted_class": predicted_class,

        "confidence_score": round(confidence * 100, 2),

        "damage_percentage": round(damage_percentage, 2),

        "severity_score": round(severity_score, 2),

        "severity_label": severity_label,

        "metrics": metrics,

        "all_probabilities": {
            CLASS_NAMES[i]: round(float(probabilities[i]), 4)
            for i in range(len(CLASS_NAMES))
        }

    } 