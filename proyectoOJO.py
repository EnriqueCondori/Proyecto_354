# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 20:48:38 2020

@author: Usuario
"""


import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import tensorflow.keras.optimizers as Optimizer

 
img_sizeAlto=2592
img_sizeAncho=1728

normal_folder_path="entrenamiento/normal"
normal=[]
for img in os.listdir(normal_folder_path):
    img = cv2.imread(os.path.join(normal_folder_path,img))
    img_resize= cv2.resize(img,(img_sizeAlto,img_sizeAncho))
    normal.append(img_resize)
    
    
mem_folder_path="entrenamiento/membrana epirretiniana macular"
mem=[]
for img in os.listdir(mem_folder_path):
    img = cv2.imread(os.path.join(mem_folder_path,img))
    img_resize= cv2.resize(img,(img_sizeAlto,img_sizeAncho))
    mem.append(img_resize)
    
    
drusas_folder_path="entrenamiento/drusas"
drusas=[]
for img in os.listdir(drusas_folder_path):
    img = cv2.imread(os.path.join(drusas_folder_path,img))
    img_resize= cv2.resize(img,(img_sizeAlto,img_sizeAncho))
    drusas.append(img_resize)
    

catarata_folder_path="entrenamiento/catarata"
catarata=[]
for img in os.listdir(catarata_folder_path):
    img = cv2.imread(os.path.join(catarata_folder_path,img))
    img_resize= cv2.resize(img,(img_sizeAlto,img_sizeAncho))
    catarata.append(img_resize)
        

orc_folder_path="entrenamiento/oclusion de la vena central de la retina"
orc=[]
for img in os.listdir(orc_folder_path):
    img = cv2.imread(os.path.join(orc_folder_path,img))
    img_resize= cv2.resize(img,(img_sizeAlto,img_sizeAncho))
    orc.append(img_resize)
    

diabetes_folder_path="entrenamiento/ritinopatia diabetica" 
diabetes=[]
for img in os.listdir(diabetes_folder_path):
    img = cv2.imread(os.path.join(diabetes_folder_path,img))
    img_resize= cv2.resize(img,(img_sizeAlto,img_sizeAncho))
    diabetes.append(img_resize)
    

dm_folder_path="entrenamiento/degeneracion macular seca relacionada con la edad" 
dm=[]
for img in os.listdir(dm_folder_path):
    img = cv2.imread(os.path.join(dm_folder_path,img))
    img_resize= cv2.resize(img,(img_sizeAlto,img_sizeAncho))
    dm.append(img_resize)
    
    
    
membrana_folder_path="entrenamiento/membrana epiretinal" 
membrana=[]
for img in os.listdir(membrana_folder_path):
    img = cv2.imread(os.path.join(membrana_folder_path,img))
    img_resize= cv2.resize(img,(img_sizeAlto,img_sizeAncho))
    membrana.append(img_resize)
    
    
glaucoma_folder_path="entrenamiento/glaucoma" 
glaucoma=[]
for img in os.listdir(glaucoma_folder_path):
    img = cv2.imread(os.path.join(glaucoma_folder_path,img))
    img_resize= cv2.resize(img,(img_sizeAlto,img_sizeAncho))
    glaucoma.append(img_resize)
    
    
hipertensiva_folder_path="entrenamiento/ritinopatia hipertensiva" 
hipertensiva=[]
for img in os.listdir(hipertensiva_folder_path):
    img = cv2.imread(os.path.join(hipertensiva_folder_path,img))
    img_resize= cv2.resize(img,(img_sizeAlto,img_sizeAncho))
    hipertensiva.append(img_resize)
    
    
miopia_folder_path="entrenamiento/miopia patologica" 
miopia=[]
for img in os.listdir(miopia_folder_path):
    img = cv2.imread(os.path.join(miopia_folder_path,img))
    img_resize= cv2.resize(img,(img_sizeAlto,img_sizeAncho))
    miopia.append(img_resize)
    

pigmentacion_folder_path="entrenamiento/pigmentacion retiniana" 
pigmentacion=[]
for img in os.listdir(pigmentacion_folder_path):
    img = cv2.imread(os.path.join(pigmentacion_folder_path,img))
    img_resize= cv2.resize(img,(img_sizeAlto,img_sizeAncho))
    pigmentacion.append(img_resize)
    
    
vitrea_folder_path="entrenamiento/degeneracion vitrea" 
vitrea=[]
for img in os.listdir(vitrea_folder_path):
    img = cv2.imread(os.path.join(vitrea_folder_path,img))
    img_resize= cv2.resize(img,(img_sizeAlto,img_sizeAncho))
    vitrea.append(img_resize)

#ver la imagen

print(normal[4].shape)
plt.figure()
plt.imshow(np.squeeze(normal[4]))
plt.colorbar()
plt.grid(False)
plt.show()


images=np.concatenate([normal, mem, drusas, catarata, orc, diabetes, dm, membrana, glaucoma, hipertensiva, miopia, pigmentacion, vitrea])
Images= np.array(images)
etiquetasN= np.repeat(0,541)
etiquetasM=np.repeat(1,31)
etiquetasD=np.repeat(2,50)
etiquetasC=np.repeat(3,47)
etiquetasO=np.repeat(4,2)
etiquetasDIA=np.repeat(6,2)
etiquetasDM=np.repeat(5,60)
etiquetasME=np.repeat(7,27)
etiquetasG=np.repeat(8,14)
etiquetasH=np.repeat(9,33)
etiquetasMIO=np.repeat(10,28)
etiquetasP=np.repeat(11,4)
etiquetasV=np.repeat(12,10)

class_names=['Normal','membrana epirretiniana macular', 'drusas', 'catarata', 'oclusión de la vena central de la retina',
             'retinopatía diabética', 'degeneración macular seca relacionada con la edad', 'membrana epiretinal',
             'glaucoma', 'retinopatía hipertensiva', 'miopía patológica', 'pigmentación retiniana', 'degeneración vítrea']

labels= np.concatenate([etiquetasN, etiquetasM, etiquetasD, etiquetasC, etiquetasO, etiquetasDIA,
                        etiquetasDM, etiquetasME, etiquetasG, etiquetasH, etiquetasMIO, etiquetasP,
                        etiquetasV])
Labels=np.array(labels)


#plt.figure(figsize=(10,10))
#for i in range(20):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
 #   plt.yticks([])
  #  plt.grid(False)
#    plt.imshow(Images[i])
    #, cmap=plt.cm.binary
 #   plt.xlabel(class_names[Labels[i]])
#plt.show()


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(2592, 1728,3)),
    keras.layers.Dense(128, activation='relu'),
    
    keras.layers.Dense(13, activation='softmax'),
    
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(Images, Labels, epochs=5)
trained=model.fit(Images, Labels, epochs=5)


img=Images[20]
print(img.shape)
img=(np.expand_dims(img,0))
print(img.shape)

plt.figure()
plt.imshow(Images[20]) 
plt.colorbar()
plt.grid(False)
plt.show()


#prediccion
#predictions_single = model.predict(img)
#print(predictions_single)
#print(np.sum(predictions_single))
#print(np.argmax(predictions_single))
#print(class_names[np.argmax(predictions_single)])

img=cv2.imread(entrenamiento/prueba/'879_right.jpg')
img_cvt=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img.cvt)
plt.show()

predictions_single = model.predict(img)
print(predictions_single)
print(np.sum(predictions_single))
print(np.argmax(predictions_single))
print(class_names[np.argmax(predictions_single)])

