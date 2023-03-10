if __name__ == '__main__':
    import os
    import shutil
    import time
    from image_downloader import download_and_preprocess_images, delete_downloaded_images
    from image_classifier import load_model, classify_images, delete_trained_data

    model_name = 'EfficientNetB0'
    num_classes = 20
    class_labels = ['cat', 'dog', 'elephant', 'horse', 'lion',
                    'panda', 'sheep', 'snake', 'squirrel', 'zebra',
                    'cow', 'deer', 'fox', 'giraffe', 'goat',
                    'hippo', 'kangaroo', 'monkey', 'rhinoceros', 'tiger']

    query = 'animal'
    num_images_list = [10, 100, 1000, 10000, 100000]
    results = []
    for num_images in num_images_list:
        print(f'Downloading and preprocessing {num_images} images...')
        save_dir = f'images_{num_images}'
        download_and_preprocess_images(query, num_images, save_dir)
        print('')

        print('Training the model...')
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2,
            preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
        train_generator = train_datagen.flow_from_directory(
            save_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=True,
            subset='training')
        validation_generator = train_datagen.flow_from_directory(
            save_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            shuffle=True,
            subset='validation')
        model = load_model(model_name, num_classes)
        model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=5)
        print('')

        print('Classifying test images...')
        start_time = time.time()
        test_dir = f'test_images_{num_images}'
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        for i in range(10):
            download_and_preprocess_images(query, 10, test_dir)
            predictions = classify_images(model, test_dir, class_labels)
            accuracy = sum([1 for pred in predictions if pred[1] == 'animal']) / len(predictions) * 100
            print(f'Accuracy with {num_images} training images and {len(predictions)} test images: {accuracy:.2f}%')
        elapsed_time = time.time() - start_time
        print(f'Elapsed time for {num_images} images: {elapsed_time:.2f} seconds')
        results.append((num_images, accuracy, elapsed_time))
        delete_downloaded_images(test_dir)
        print('')

        print('Deleting downloaded images...')
        delete_downloaded_images(save_dir)
        print('')

        print('Deleting trained data...')
        data_dir = os.path.join(os.getcwd(), 'data')
        delete_trained_data(data_dir)
        print('')

    print(f'Results: {results}')
