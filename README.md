🧠 1. Redes Neuronales Convolucionales (CNN)

Son modelos diseñados para analizar imágenes imitando cómo el cerebro detecta patrones visuales.
Detectan bordes, texturas y formas automáticamente sin que tengamos que programarlo.

🔹 LeNet (1998)

Uno de los primeros modelos CNN creados por Yann LeCun.

Usado originalmente para reconocer dígitos escritos a mano (MNIST).

Estructura simple (2 capas convolucionales + 2 capas fully connected).

Ideal para datasets pequeños y tareas básicas.

🔹 AlexNet (2012)

Marcó un gran avance en visión por computadora (ganó ImageNet 2012).

Más profunda que LeNet (8 capas).

Usa ReLU (activación rápida) y Dropout (evita sobreajuste).

Ideal para clasificación de imágenes a gran escala.

🔹 VGG (2014)

Se caracteriza por usar bloques de convoluciones 3x3 repetidas.

Modelos conocidos: VGG16 y VGG19 (por la cantidad de capas).

Muy usada por su simplicidad y rendimiento sólido, aunque consume mucha memoria.

🔹 ResNet (2015)

Introduce conexiones residuales, que permiten redes muy profundas (50, 101 o más capas).

Evita el problema de “desvanecimiento del gradiente”.

Modelos populares: ResNet50, ResNet101.

Muy usada en Transfer Learning.

🔹 Inception (GoogLeNet)

Usa módulos Inception, que combinan convoluciones de diferentes tamaños (1x1, 3x3, 5x5) en paralelo.

Aprende características a distintas escalas.

Muy eficiente en tiempo y precisión.

🔹 MobileNet

Diseñada para dispositivos móviles o de baja potencia.

Usa convoluciones “depthwise separable” (más ligeras).

Ideal para apps en Android, IoT o proyectos con recursos limitados.

🔹 EfficientNet (2019)

Optimiza simultáneamente profundidad, anchura y resolución del modelo.

Más precisa y ligera que redes anteriores.

Ideal para proyectos modernos que buscan alto rendimiento con bajo costo computacional.

⚙️ 2. Transfer Learning (Aprendizaje por Transferencia)

Es una técnica donde se usa un modelo ya entrenado (por ejemplo, ResNet50 o VGG16) y se ajusta a un nuevo dataset.

Ejemplo: tomar una red pre-entrenada con millones de imágenes (ImageNet) y adaptarla para clasificar radiografías o tipos de frutas.

Ventajas:

Requiere menos datos y tiempo.

Mejora la precisión con poco entrenamiento adicional.

📊 3. Métodos Clásicos

Antes del boom de las CNN, la clasificación de imágenes se hacía extrayendo características manualmente y usando algoritmos clásicos de machine learning.

🔹 k-NN (k-Nearest Neighbors)

Clasifica una imagen comparándola con sus “vecinos” más cercanos en el espacio de características.

Simple pero eficaz en datasets pequeños.

🔹 SVM (Support Vector Machine)

Encuentra el “hiperplano” que mejor separa las clases.

Muy útil para imágenes con pocas características relevantes.

🔹 Random Forest

Usa un conjunto de árboles de decisión para clasificar imágenes.

Robusto y fácil de usar, aunque menos potente que las CNN para datos complejos.

🧩 Técnicas de Extracción de Características

Antes de usar algoritmos como SVM o k-NN, era necesario extraer características visuales.

🔹 HOG (Histogram of Oriented Gradients)

Describe bordes y direcciones de gradientes en la imagen.

Muy usado para detección de personas o vehículos.

🔹 SIFT (Scale-Invariant Feature Transform)

Detecta puntos clave en la imagen (invariante a escala e iluminación).

Ideal para reconocimiento de objetos.

🔹 SURF (Speeded-Up Robust Features)

Versión optimizada de SIFT, más rápida.

Detecta y describe regiones distintivas en las imágenes.