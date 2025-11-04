from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Para el DataSet
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import array_to_img, save_img


# Cargar modelo preentrenado
model = ResNet50(weights='imagenet')

# Carpeta de imágenes
carpeta = 'img_train'

# generamos las imagenes 
os.makedirs(carpeta, exist_ok=True)
(x_train, y_train), _ = cifar10.load_data()
clases = ['avion', 'auto', 'pajaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camion']
for i in range(10):
    img = array_to_img(x_train[i])  # Convierte el array a imagen
    clase = clases[y_train[i][0]]
    save_path = os.path.join(carpeta, f"{clase}_{i}.jpg")
    img.save(save_path)

print("✅ Imágenes guardadas en la carpeta 'img_train'")


# analizamos las imagenes con RESNET50
for nombre in os.listdir(carpeta):
    ruta = os.path.join(carpeta, nombre)
    if ruta.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = image.load_img(ruta, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        descripcion = decode_predictions(preds, top=1)[0][0][1]  # nombre del objeto

        print(f"Imagen: {nombre} → Predicción: {descripcion}")

        
