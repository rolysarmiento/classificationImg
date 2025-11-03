# ============================================================================
# CLASIFICACIÓN DE IMÁGENES CON PYTHON
# Tutorial Paso a Paso con Diferentes Enfoques
# ============================================================================

"""
Este tutorial cubre:
1. Métodos Clásicos (k-NN, SVM, Random Forest)
2. Redes Neuronales Convolucionales (CNN)
3. Transfer Learning con modelos pre-entrenados
"""

# ============================================================================
# PARTE 1: INSTALACIÓN DE LIBRERÍAS NECESARIAS
# ============================================================================

"""
Ejecuta en tu terminal:

pip install tensorflow keras opencv-python scikit-learn numpy matplotlib pillow scikit-image

"""

# ============================================================================
# PARTE 2: IMPORTAR LIBRERÍAS
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ============================================================================
# PARTE 3: CARGAR Y PREPARAR DATOS
# ============================================================================

def cargar_dataset_ejemplo():
    """
    Carga un dataset de ejemplo. En este caso usaremos CIFAR-10
    CIFAR-10: 60,000 imágenes de 32x32 en 10 clases
    
    Para tu propio dataset:
    - Organiza las imágenes en carpetas por clase
    - Usa ImageDataGenerator de Keras o carga manualmente
    """
    print("📦 Cargando dataset CIFAR-10...")
    
    # Cargar CIFAR-10 (incluido en Keras)
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Nombres de las clases
    nombres_clases = ['avión', 'auto', 'pájaro', 'gato', 'ciervo', 
                      'perro', 'rana', 'caballo', 'barco', 'camión']
    
    print(f"✅ Dataset cargado:")
    print(f"   - Entrenamiento: {X_train.shape[0]} imágenes")
    print(f"   - Prueba: {X_test.shape[0]} imágenes")
    print(f"   - Tamaño de imagen: {X_train.shape[1]}x{X_train.shape[2]}")
    print(f"   - Clases: {len(nombres_clases)}")
    
    return X_train, y_train, X_test, y_test, nombres_clases


def visualizar_imagenes(X, y, nombres_clases, n_imagenes=10):
    """
    Visualiza algunas imágenes del dataset
    """
    plt.figure(figsize=(15, 3))
    for i in range(n_imagenes):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X[i])
        plt.title(nombres_clases[y[i][0]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# ============================================================================
# PARTE 4: EXTRACCIÓN DE CARACTERÍSTICAS PARA MÉTODOS CLÁSICOS
# ============================================================================

def extraer_caracteristicas_hog(imagenes):
    """
    Extrae características HOG (Histogram of Oriented Gradients)
    HOG captura la forma y estructura de los objetos
    """
    from skimage.feature import hog
    
    print("🔍 Extrayendo características HOG...")
    caracteristicas = []
    
    for img in imagenes:
        # Convertir a escala de grises
        img_gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Extraer HOG
        features = hog(img_gris, 
                      orientations=9,  # Número de orientaciones
                      pixels_per_cell=(8, 8),  # Tamaño de celda
                      cells_per_block=(2, 2),  # Celdas por bloque
                      visualize=False)
        
        caracteristicas.append(features)
    
    return np.array(caracteristicas)


def extraer_caracteristicas_simple(imagenes):
    """
    Método simple: aplanar las imágenes
    Convierte cada imagen en un vector 1D
    """
    print("🔍 Aplanando imágenes...")
    # Normalizar píxeles entre 0 y 1
    imagenes_norm = imagenes.astype('float32') / 255.0
    # Aplanar: convertir de (32, 32, 3) a (3072,)
    caracteristicas = imagenes_norm.reshape(imagenes_norm.shape[0], -1)
    return caracteristicas


# ============================================================================
# PARTE 5: MÉTODO CLÁSICO 1 - K-NEAREST NEIGHBORS (k-NN)
# ============================================================================

def entrenar_knn(X_train, y_train, X_test, y_test, k=5):
    """
    k-NN: Clasifica basándose en los k vecinos más cercanos
    - Simple y fácil de entender
    - Bueno para datasets pequeños
    - Lento en predicción con muchos datos
    """
    print(f"\n🤖 Entrenando k-NN (k={k})...")
    
    # Aplanar las etiquetas
    y_train_flat = y_train.ravel()
    y_test_flat = y_test.ravel()
    
    # Extraer características
    X_train_features = extraer_caracteristicas_simple(X_train)
    X_test_features = extraer_caracteristicas_simple(X_test)
    
    # Usar solo un subconjunto para velocidad (opcional)
    n_muestras = 5000
    X_train_features = X_train_features[:n_muestras]
    y_train_flat = y_train_flat[:n_muestras]
    
    # Crear y entrenar el modelo
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_features, y_train_flat)
    
    # Evaluar
    y_pred = knn.predict(X_test_features)
    accuracy = accuracy_score(y_test_flat, y_pred)
    
    print(f"✅ Precisión k-NN: {accuracy*100:.2f}%")
    
    return knn, accuracy


# ============================================================================
# PARTE 6: MÉTODO CLÁSICO 2 - SUPPORT VECTOR MACHINE (SVM)
# ============================================================================

def entrenar_svm(X_train, y_train, X_test, y_test):
    """
    SVM: Encuentra el hiperplano que mejor separa las clases
    - Muy efectivo en espacios de alta dimensión
    - Bueno con margen de separación claro
    - Puede ser lento con muchos datos
    """
    print(f"\n🤖 Entrenando SVM...")
    
    # Aplanar las etiquetas
    y_train_flat = y_train.ravel()
    y_test_flat = y_test.ravel()
    
    # Extraer características HOG (mejor que píxeles crudos para SVM)
    # Usar un subconjunto para velocidad
    n_muestras = 3000
    X_train_hog = extraer_caracteristicas_hog(X_train[:n_muestras])
    X_test_hog = extraer_caracteristicas_hog(X_test[:1000])
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_hog)
    X_test_scaled = scaler.transform(X_test_hog)
    
    # Crear y entrenar SVM
    svm = SVC(kernel='rbf',  # Kernel de función de base radial
              C=10,  # Parámetro de regularización
              gamma='scale')
    
    svm.fit(X_train_scaled, y_train_flat[:n_muestras])
    
    # Evaluar
    y_pred = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test_flat[:1000], y_pred)
    
    print(f"✅ Precisión SVM: {accuracy*100:.2f}%")
    
    return svm, accuracy


# ============================================================================
# PARTE 7: MÉTODO CLÁSICO 3 - RANDOM FOREST
# ============================================================================

def entrenar_random_forest(X_train, y_train, X_test, y_test):
    """
    Random Forest: Conjunto de árboles de decisión
    - Robusto y preciso
    - Maneja bien datos no lineales
    - Menos propenso a overfitting que un solo árbol
    """
    print(f"\n🌲 Entrenando Random Forest...")
    
    # Aplanar las etiquetas
    y_train_flat = y_train.ravel()
    y_test_flat = y_test.ravel()
    
    # Extraer características
    X_train_features = extraer_caracteristicas_simple(X_train)
    X_test_features = extraer_caracteristicas_simple(X_test)
    
    # Usar un subconjunto
    n_muestras = 5000
    X_train_features = X_train_features[:n_muestras]
    y_train_flat = y_train_flat[:n_muestras]
    
    # Crear y entrenar Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,  # Número de árboles
        max_depth=20,  # Profundidad máxima de cada árbol
        random_state=42,
        n_jobs=-1  # Usar todos los cores
    )
    
    rf.fit(X_train_features, y_train_flat)
    
    # Evaluar
    y_pred = rf.predict(X_test_features)
    accuracy = accuracy_score(y_test_flat, y_pred)
    
    print(f"✅ Precisión Random Forest: {accuracy*100:.2f}%")
    
    return rf, accuracy


# ============================================================================
# PARTE 8: RED NEURONAL CONVOLUCIONAL (CNN) DESDE CERO
# ============================================================================

def crear_cnn_simple(input_shape, num_clases):
    """
    Crea una CNN simple para clasificación de imágenes
    
    Arquitectura:
    - Conv2D: Extrae características locales
    - MaxPooling: Reduce dimensionalidad
    - Dropout: Previene overfitting
    - Dense: Capas completamente conectadas para clasificación
    """
    model = models.Sequential([
        # Primera capa convolucional
        # Aplica 32 filtros de 3x3 para detectar características básicas
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),  # Reduce a la mitad
        
        # Segunda capa convolucional
        # 64 filtros para características más complejas
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Tercera capa convolucional
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Aplanar para capas densas
        layers.Flatten(),
        
        # Capas completamente conectadas
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),  # Apaga aleatoriamente 50% de neuronas
        
        # Capa de salida
        layers.Dense(num_clases, activation='softmax')  # Probabilidades por clase
    ])
    
    return model


def entrenar_cnn(X_train, y_train, X_test, y_test, epochs=10):
    """
    Entrena una CNN desde cero
    """
    print(f"\n🧠 Entrenando CNN desde cero...")
    
    # Normalizar imágenes
    X_train_norm = X_train.astype('float32') / 255.0
    X_test_norm = X_test.astype('float32') / 255.0
    
    # Convertir etiquetas a one-hot encoding
    # Ejemplo: 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)
    
    # Crear modelo
    model = crear_cnn_simple(X_train_norm.shape[1:], 10)
    
    # Compilar modelo
    model.compile(
        optimizer='adam',  # Algoritmo de optimización
        loss='categorical_crossentropy',  # Función de pérdida para multiclase
        metrics=['accuracy']
    )
    
    # Mostrar arquitectura
    model.summary()
    
    # Entrenar
    history = model.fit(
        X_train_norm, y_train_cat,
        epochs=epochs,
        batch_size=64,  # Procesar 64 imágenes a la vez
        validation_data=(X_test_norm, y_test_cat),
        verbose=1
    )
    
    # Evaluar
    test_loss, test_acc = model.evaluate(X_test_norm, y_test_cat, verbose=0)
    print(f"✅ Precisión CNN: {test_acc*100:.2f}%")
    
    return model, history, test_acc


# ============================================================================
# PARTE 9: TRANSFER LEARNING CON MODELOS PRE-ENTRENADOS
# ============================================================================

def crear_modelo_transfer_learning(modelo_base_nombre='ResNet50', num_clases=10):
    """
    Transfer Learning: Usar un modelo pre-entrenado en ImageNet
    
    Modelos disponibles:
    - ResNet50: 50 capas, muy preciso
    - VGG16: Arquitectura clásica, más simple
    - InceptionV3: Usa convoluciones de diferentes tamaños
    
    Ventajas:
    - Aprende más rápido (usa conocimiento previo)
    - Necesita menos datos
    - Mayor precisión
    """
    print(f"\n🚀 Creando modelo con Transfer Learning ({modelo_base_nombre})...")
    
    # Seleccionar modelo base
    if modelo_base_nombre == 'ResNet50':
        modelo_base = ResNet50(weights='imagenet', 
                               include_top=False,  # Sin capa de clasificación
                               input_shape=(32, 32, 3))
    elif modelo_base_nombre == 'VGG16':
        modelo_base = VGG16(weights='imagenet', 
                           include_top=False, 
                           input_shape=(32, 32, 3))
    elif modelo_base_nombre == 'InceptionV3':
        modelo_base = InceptionV3(weights='imagenet', 
                                 include_top=False, 
                                 input_shape=(75, 75, 3))  # InceptionV3 necesita min 75x75
    
    # Congelar las capas del modelo base
    # No entrenaremos estas capas, solo usaremos sus características
    modelo_base.trainable = False
    
    # Construir modelo completo
    model = models.Sequential([
        modelo_base,
        layers.GlobalAveragePooling2D(),  # Reduce dimensiones
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_clases, activation='softmax')
    ])
    
    return model


def entrenar_transfer_learning(X_train, y_train, X_test, y_test, 
                               modelo_nombre='ResNet50', epochs=5):
    """
    Entrena con Transfer Learning
    """
    print(f"\n🎯 Entrenando con Transfer Learning ({modelo_nombre})...")
    
    # Preparar datos
    X_train_norm = X_train.astype('float32') / 255.0
    X_test_norm = X_test.astype('float32') / 255.0
    
    # Preprocesar según el modelo
    if modelo_nombre in ['ResNet50', 'VGG16']:
        X_train_prep = keras.applications.resnet50.preprocess_input(X_train_norm * 255)
        X_test_prep = keras.applications.resnet50.preprocess_input(X_test_norm * 255)
    
    # One-hot encoding
    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)
    
    # Crear modelo
    model = crear_modelo_transfer_learning(modelo_nombre, 10)
    
    # Compilar
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Entrenar solo con un subconjunto para velocidad
    n_muestras = 5000
    history = model.fit(
        X_train_prep[:n_muestras], y_train_cat[:n_muestras],
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test_prep, y_test_cat),
        verbose=1
    )
    
    # Evaluar
    test_loss, test_acc = model.evaluate(X_test_prep, y_test_cat, verbose=0)
    print(f"✅ Precisión Transfer Learning: {test_acc*100:.2f}%")
    
    return model, history, test_acc


# ============================================================================
# PARTE 10: COMPARACIÓN DE TODOS LOS MODELOS
# ============================================================================

def comparar_modelos():
    """
    Ejecuta y compara todos los modelos
    """
    print("=" * 70)
    print("CLASIFICACIÓN DE IMÁGENES - COMPARACIÓN DE MÉTODOS")
    print("=" * 70)
    
    # Cargar datos
    X_train, y_train, X_test, y_test, nombres_clases = cargar_dataset_ejemplo()
    
    # Visualizar algunas imágenes
    visualizar_imagenes(X_train, y_train, nombres_clases)
    
    # Diccionario para guardar resultados
    resultados = {}
    
    # 1. k-NN
    try:
        _, acc_knn = entrenar_knn(X_train, y_train, X_test, y_test)
        resultados['k-NN'] = acc_knn
    except Exception as e:
        print(f"❌ Error en k-NN: {e}")
    
    # 2. SVM
    try:
        _, acc_svm = entrenar_svm(X_train, y_train, X_test, y_test)
        resultados['SVM'] = acc_svm
    except Exception as e:
        print(f"❌ Error en SVM: {e}")
    
    # 3. Random Forest
    try:
        _, acc_rf = entrenar_random_forest(X_train, y_train, X_test, y_test)
        resultados['Random Forest'] = acc_rf
    except Exception as e:
        print(f"❌ Error en Random Forest: {e}")
    
    # 4. CNN desde cero
    try:
        _, _, acc_cnn = entrenar_cnn(X_train, y_train, X_test, y_test, epochs=5)
        resultados['CNN'] = acc_cnn
    except Exception as e:
        print(f"❌ Error en CNN: {e}")
    
    # 5. Transfer Learning
    try:
        _, _, acc_tl = entrenar_transfer_learning(X_train, y_train, X_test, y_test, 
                                                   'ResNet50', epochs=3)
        resultados['Transfer Learning (ResNet50)'] = acc_tl
    except Exception as e:
        print(f"❌ Error en Transfer Learning: {e}")
    
    # Mostrar resultados
    print("\n" + "=" * 70)
    print("📊 RESULTADOS FINALES")
    print("=" * 70)
    for modelo, accuracy in sorted(resultados.items(), key=lambda x: x[1], reverse=True):
        print(f"{modelo:30s}: {accuracy*100:6.2f}%")
    
    # Visualizar comparación
    plt.figure(figsize=(10, 6))
    plt.bar(resultados.keys(), [v*100 for v in resultados.values()])
    plt.ylabel('Precisión (%)')
    plt.title('Comparación de Modelos de Clasificación')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return resultados


# ============================================================================
# PARTE 11: USAR TU PROPIO DATASET
# ============================================================================

def cargar_dataset_personalizado(ruta_dataset):
    """
    Carga tu propio dataset de imágenes
    
    Estructura esperada:
    ruta_dataset/
        clase1/
            imagen1.jpg
            imagen2.jpg
        clase2/
            imagen1.jpg
            imagen2.jpg
    """
    print(f"📁 Cargando dataset desde: {ruta_dataset}")
    
    # Usar ImageDataGenerator de Keras
    datagen = ImageDataGenerator(
        rescale=1./255,  # Normalizar
        validation_split=0.2  # 20% para validación
    )
    
    # Cargar datos de entrenamiento
    train_generator = datagen.flow_from_directory(
        ruta_dataset,
        target_size=(224, 224),  # Redimensionar imágenes
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    # Cargar datos de validación
    val_generator = datagen.flow_from_directory(
        ruta_dataset,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, val_generator


# ============================================================================
# PARTE 12: GUARDAR Y CARGAR MODELOS
# ============================================================================

def guardar_modelo(model, nombre_archivo):
    """
    Guarda un modelo entrenado
    """
    model.save(nombre_archivo)
    print(f"💾 Modelo guardado en: {nombre_archivo}")


def cargar_modelo_guardado(nombre_archivo):
    """
    Carga un modelo previamente guardado
    """
    model = keras.models.load_model(nombre_archivo)
    print(f"📂 Modelo cargado desde: {nombre_archivo}")
    return model


def predecir_imagen(model, imagen_path, nombres_clases):
    """
    Predice la clase de una nueva imagen
    """
    # Cargar imagen
    img = cv2.imread(imagen_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    
    # Preprocesar
    img_array = np.expand_dims(img, axis=0)
    img_array = img_array.astype('float32') / 255.0
    
    # Predecir
    predicciones = model.predict(img_array)
    clase_predicha = np.argmax(predicciones[0])
    confianza = predicciones[0][clase_predicha]
    
    print(f"Clase predicha: {nombres_clases[clase_predicha]}")
    print(f"Confianza: {confianza*100:.2f}%")
    
    # Visualizar
    plt.imshow(img)
    plt.title(f"{nombres_clases[clase_predicha]} ({confianza*100:.1f}%)")
    plt.axis('off')
    plt.show()
    
    return clase_predicha, confianza


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Ejecutar comparación completa
    resultados = comparar_modelos()
    
    # Para usar tu propio dataset:
    # train_gen, val_gen = cargar_dataset_personalizado('ruta/a/tu/dataset')
    
    # Para entrenar solo un modelo específico:
    # X_train, y_train, X_test, y_test, nombres_clases = cargar_dataset_ejemplo()
    # model, history, acc = entrenar_cnn(X_train, y_train, X_test, y_test, epochs=10)
    # guardar_modelo(model, 'mi_modelo_cnn.h5')
    
    print("\n✨ ¡Tutorial completado!")