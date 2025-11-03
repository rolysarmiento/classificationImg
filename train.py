from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Cargar modelo preentrenado
model = ResNet50(weights='imagenet')

# Carpeta de imágenes
carpeta = 'img_train'

for nombre in os.listdir(carpeta):
    ruta = os.path.join(carpeta, nombre)
    if ruta.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Cargar imagen y adaptarla al tamaño del modelo
        img = image.load_img(ruta, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Predicción
        preds = model.predict(x)
        descripcion = decode_predictions(preds, top=1)[0][0][1]  # nombre del objeto

        print(f"Imagen: {nombre} → Predicción: {descripcion}")
