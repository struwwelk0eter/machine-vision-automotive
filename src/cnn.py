from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from matplotlib import pyplot as plt
import cv2
import numpy as np
import keras.callbacks as call
from keras.models import load_model


#MODEL
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # 3D in 1D 

model.add(Dense(16))
model.add(Activation('relu'))

model.add(Dense(16))
model.add(Activation('relu'))

model.add(Dense(2))
model.add(Activation('softmax'))

#Show summary
model.summary()


#Compile model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

#Define batch size
batch_size = 16

#Modify the training data
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range = 25,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        r'C:\train',  #directory
        target_size=(150, 150),  
        batch_size=batch_size,
        class_mode='binary') 

#imgs, labels = next(train_generator)
#print (imgs)

#show 5 example images
#c = 0
#for i in imgs:
#    plt.imshow(i[1,:]),plt.title('Original')
##    plt.xticks([]), plt.yticks([])
#    c = c+1
#    plt.show()
#    if c==5:
#        break
#    cv2.imshow('des', i)
#    cv2.waitKey(0)


validation_generator = test_datagen.flow_from_directory(
        r'C:\validation',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

#Tensorboard to supervise the training process
tensorboard = call.TensorBoard(log_dir=r'C:\Users\model',
                                          write_graph=True,
                                          write_images=True,)
#Fit the model
model.fit_generator(
        train_generator,
        steps_per_epoch=140 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps= 40 // batch_size,
#       shuffle = True,
        verbose = 1,
        callbacks=[tensorboard]) #Tensorboard connection


#save
model.save(r'C:\path')

#load
#model_loaded=load_model(r'C:\path')

test_generator = test_datagen.flow_from_directory(
        r'C:\path',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

#test_img = cv2.imread(r'C:\path')
#x = np.expand_dims(test_img, axis=0)
#print (x.shape)

#Predict
prediction = model.predict_generator(test_generator, steps = 1)
print (prediction)
