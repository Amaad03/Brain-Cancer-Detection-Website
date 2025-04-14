import os
import numpy as np
from keras._tf_keras.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2

#FinalModel = 93%
#model_test.keras = 85.58%


model = load_model('vggtest.keras')
test_dir = 'dataset/Testing'


class_names = sorted(os.listdir(test_dir))
print(f"Class names: {class_names}")


def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (244, 244))
    img = img / 255.0
    img = np.stack((img,)*3, axis=-1)
    img_array = np.expand_dims(img, axis=0)
    return img_array

total_images = 0
correct_predictions = 0

for class_name in class_names:
    class_path = os.path.join(test_dir, class_name)
    for image_name in os.listdir(class_path):
        img_path = os.path.join(class_path, image_name)
        img_array = preprocess_image(img_path)
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        print(f"Raw prediction for {image_name}: is {predicted_class_name}")
        
     
        actual_class_name = None
        if "no" in image_name:
            actual_class_name = 'notumor'
        elif "pi" in image_name or "piTr" in image_name:
            actual_class_name = 'pituitary'
        elif "gl" in image_name:
            actual_class_name = 'glioma'
        elif "me" in image_name:
            actual_class_name = 'meningioma'
        

        if predicted_class_name == actual_class_name:
            correct_predictions += 1
        total_images += 1


accuracy = (correct_predictions / total_images) * 100
print(f"Accuracy: {accuracy:.2f}%")
print(f"Total Images: {total_images}")

