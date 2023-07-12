import cv2
import numpy as np
import matplotlib.pyplot as plt
from Eigenfaces import process_and_train, classify_image, create_database_from_folder, reconstruct_image
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from time import time
import glob
from sklearn.metrics import classification_report

# online detection: True
# offline Detection: False
onlineDetection = False

if onlineDetection:
    # image size
    N = 64

    # cascade classifier for face detection in images
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # create database from images in eigenfaces folder
    labels, train, num_images = create_database_from_folder(glob.glob('eigenfaces/*.png'))

    # generate a database and train the classifier on it (train is transposed to get row images representation)
    u, num_eigenfaces, avg = process_and_train(labels, train.T, num_images, N, N)

    # gets the video source
    video_capture = cv2.VideoCapture(0)

    # run until 'q' is pressed
    while True:
        # Capture frame-by-frame
        _, frame = video_capture.read()

        # get gray value image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect face with haar cascade
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(N - 10, N - 10))

        # draw rectangle around face
        for (x, y, w, h) in faces:
            # extract face from frame and gray value image
            face = frame[y: y + h, x: x + w]
            image = gray[y: y + h, x: x + w]

            # equalize histogram
            image = cv2.equalizeHist(image)
            # resize to NxN
            image = cv2.resize(image, (N, N))

            # draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # time measurement
            start = time()
            # predict face and store label as string
            pred = str(classify_image(image, u, avg, num_eigenfaces, N, N))

            # creating a label to draw it (Name, and time in ms
            name = str(pred).split("'")[1].split("_")[0] + "_" + str((time() - start) * 1000).split(".")[0] + " ms"
            # draw the label to the frame
            cv2.putText(frame, name.split("_")[0], (int(x), int(y + h + 25)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, name.split("_")[1], (int(x), int(y + h + 50)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 1, cv2.LINE_AA)

        # show the video
        cv2.imshow('Video', frame)

        # break the loop and release webcam
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release camera
    video_capture.release()
    cv2.destroyAllWindows()

else:
    ### SET UP
    # load the dataset and extract dimensions
    lfw_dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    n_images, h, w = lfw_dataset.images.shape

    # extract samples and labels (images and names)
    X = lfw_dataset.data
    y = lfw_dataset.target
    labels = lfw_dataset.target_names
    n_labels = labels.shape[0]

    print("Dataset parameters:")
    print("Total number of images: ", format(n_images))
    print("Total number of people: ", format(n_labels))

    ### EIGENFACE COMPUTATION & TRAINING
    # Splits the data set by using test_train_split()
    # The test training set should be 25% of the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    n_train_images = X_train.shape[0]
    n_test_images = X_test.shape[0]
    print("Training n_images: ", format(n_train_images))

    # trains the classifier on generated features
    t0 = time()
    u, num_eigenfaces, avg = process_and_train(y_train, X_train, n_train_images, h, w)
    print("Training done in %0.3fs" % (time() - t0))

    ### RECONSTRUCTION
    # reconstruct a test image using the eigenfaces and a varying number of eigenfaces
    reco_input = np.copy(X_test[0])
    y_reco_10 = reconstruct_image(np.copy(reco_input), u, avg, 10, h, w)
    y_reco_100 = reconstruct_image(np.copy(reco_input), u, avg, 100, h, w)
    y_reco_full = reconstruct_image(np.copy(reco_input), u, avg, num_eigenfaces, h, w)

    plt.suptitle('Reconstruction pipeline')
    plt.subplot(1, 4, 1)
    plt.imshow(reco_input.reshape(h, w), cmap='gray')
    plt.title('Input image')

    plt.subplot(1, 4, 2)
    plt.imshow(y_reco_10, cmap='gray')
    plt.title('Reco with\n10 eigenfaces')

    plt.subplot(1, 4, 3)
    plt.imshow(y_reco_100, cmap='gray')
    plt.title('Reco with\n100 eigenfaces')

    plt.subplot(1, 4, 4)
    plt.imshow(y_reco_full, cmap='gray')
    plt.title('Reco with\nall eigenfaces')

    plt.axis('off')
    plt.show()



    ### CLASSIFICATION
    # classify the training set using the trained classifier
    t0 = time()

    # get predictions
    y_pred = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        y_pred[i] = classify_image(X_test[i], u, avg, num_eigenfaces, h, w)

    # evaluation
    end = (time() - t0)
    print("Prediction of", n_test_images, "faces done in %0.3fs" % end)
    print("Prediction time per face %0.3fs" % (end / n_test_images))

    print(classification_report(y_test, y_pred, target_names=labels))
