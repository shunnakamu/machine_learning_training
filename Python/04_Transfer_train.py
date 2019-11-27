from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Input
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import CSVLogger

if __name__ == '__main__':
    n_categories=2
    batch_size=64
    epochs=5
    train_dir='drive/My Drive/dataset/trainData'
    validation_dir='drive/My Drive/dataset/validationData'
    file_name='vgg16_fine'

    base_model = VGG16(weights='imagenet', include_top=False,
                       input_tensor=Input(shape=(224, 224, 3)))

    # add new layers instead of FC networks
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    prediction = Dense(n_categories, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=prediction)

    # fix weights before VGG16 14layers
    for layer in base_model.layers[:15]:
        layer.trainable = False

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # save model
    json_string = model.to_json()
    open(file_name + '.json', 'w').write(json_string)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    hist = model.fit_generator(train_generator,
                               epochs=epochs,
                               verbose=1,
                               validation_data=validation_generator)

    # save weights
    model.save(file_name + '.h5')