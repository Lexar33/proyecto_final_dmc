import numpy as np

import keras_ocr
import matplotlib.pyplot as plt

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

# Get a set of three example images
images = [keras_ocr.tools.read('recorte_objeto.jpg')]

# Each list of predictions in prediction_groups is a list of
# (word, box) tuples.
#prediction_groups = pipeline.recognize(images)

# Plot the predictions
#fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
#for ax, image, predictions in zip(axs, images, prediction_groups):
#    keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)

#plt.show()



# 3. Realizar la predicción
prediction_groups = pipeline.recognize(images)

# 4. Mostrar resultados (Ajuste para evitar el TypeError)
fig, ax = plt.subplots(figsize=(10, 10))

# Como solo hay una imagen, accedemos directamente a la primera predicción
keras_ocr.tools.drawAnnotations(image=images[0], 
                                predictions=prediction_groups[0], 
                                ax=ax)

# Opcional: imprimir en consola para verificar
print("\n--- Texto Detectado ---")
for text, box in prediction_groups[0]:
    print(f"Palabra: {text}")