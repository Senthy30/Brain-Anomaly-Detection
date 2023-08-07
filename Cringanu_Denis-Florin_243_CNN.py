import csv
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2
import random
from queue import Queue

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter

import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, AveragePooling2D
from tensorflow.keras.models import load_model
from keras import metrics

# libraries -------------------------------

# optimizare pentru GPU pentru a nu utiliza mai multe memorie decat are
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

# functie folosita in trecut pentru a crea path-urile catre fisierele de date si labels
def UpdateFilename():
    global prefixPath
    global trainDataFilename, trainLabelsFilename
    global validationDataFilename, validationLabelsFilename
    global testDataFilename

    # check if prefixPath + "data_array/" exists
    if not os.path.exists(prefixPath + "data_array/"):
        os.mkdir(prefixPath + "data_array/")

    # check if prefixPath + "labels_array/" exists
    if not os.path.exists(prefixPath + "labels_array/"):
        os.mkdir(prefixPath + "labels_array/")

    # check fi prefixPath + "modified_images" exists
    if not os.path.exists(prefixPath + "modified_images/"):
        os.mkdir(prefixPath + "modified_images/")

    trainDataFilename = prefixPath + "data_array/" + trainDataFilename
    trainLabelsFilename = (prefixPath + trainLabelsFilename[0], prefixPath + "labels_array/" + trainLabelsFilename[1])

    validationDataFilename = prefixPath + "data_array/" + validationDataFilename
    validationLabelsFilename = (prefixPath + validationLabelsFilename[0], prefixPath + "labels_array/" + validationLabelsFilename[1])

    testDataFilename = prefixPath + "data_array/" + testDataFilename

prefixPath = ""

trainDataFilename = "train_data_array.npy"
trainLabelsFilename = ("train_labels.txt", "train_labels.npy")

validationDataFilename = "validation_data_array.npy"
validationLabelsFilename = ("validation_labels.txt", "validation_labels.npy")

testDataFilename = "test_data_array.npy"

# 60 60
dimensions = (128, 128, 1)

trainInterval = (1, 15001)
validationInterval = (15001, 17001)
testInterval = (17001, 22150)

# arrays in care au fost stocate in trecut imaginile si labels, inainte de folosirea ImageDataGenerator
trainData = None
trainLabels = None
validationData = None
validationLabels = None
testData = None

trainGenerator = None
validationGenerator = None
testGenerator = None

UpdateFilename()

# functie folosia in trecut pentru a citi imaginle si le a le scrie in fisierele .npy pentru o citi mai rapida a acestora
def ReloadData():
    global dimensions
    global prefixPath
    global trainDataFilename, trainLabelsFilename
    global validationDataFilename, validationLabelsFilename
    global testDataFilename

    # creez numele la fisierele .npy pentru train
    trainDataOutputFile = open(trainDataFilename.split('.')[0] + "_input.npy", "wb")
    trainLabelsInputFile = open(trainLabelsFilename[0], "r")
    trainLabelsOutputFile = open(trainLabelsFilename[1].split('.')[0] + "_input.npy", "wb")

    # creez numele la fisierele .npy pentru validation
    validationDataOutputFile = open(validationDataFilename.split('.')[0] + "_input.npy", "wb")
    validationLabelsInputFile = open(validationLabelsFilename[0], "r")
    validationLabelsOutputFile = open(validationLabelsFilename[1].split('.')[0] + "_input.npy", "wb")

    # creez numele la fisierele .npy pentru test
    testdataOutputFile = open(testDataFilename.split('.')[0] + "_input.npy", "wb")

    # citesc labels pentru train
    print("Writing train labels...\n")
    for line in tqdm(trainLabelsInputFile.readlines()):
        line = line.replace('\n', '').split(',')
        if line[0] == 'id':
            continue
        np.save(trainLabelsOutputFile, line[1])
    print()

    # citesc labels pentru validation
    print("Writing validation labels...\n")
    for line in tqdm(validationLabelsInputFile.readlines()):
        line = line.replace('\n', '').split(',')
        if line[0] == 'id':
            continue
        np.save(validationLabelsOutputFile, line[1])
    print()

    imagePrefix = prefixPath + "data/"
    listImagesName = os.listdir(prefixPath + "data/")
    currentCntImage = 1

    # citesc imaginile, le fac resize si le scriu in fisierele .npy corespunzatoare
    print("Writing images data...\n")
    for imageName in tqdm(listImagesName):
        # citesc imaginea
        image = cv2.imread(imagePrefix + imageName, cv2.IMREAD_GRAYSCALE)
        # fac resize la noua dimensiune
        image = cv2.resize(image, (dimensions[0], dimensions[1]), interpolation = cv2.INTER_AREA)
        
        # in functie de intervalul in care se afla currentCntImage, o scriu in fisierul corespunzator
        if currentCntImage < trainInterval[1]:
            data = np.ravel(image)
            np.save(trainDataOutputFile, data)
        elif currentCntImage < validationInterval[1]:
            data = np.ravel(image)
            np.save(validationDataOutputFile, data)
        else:
            data = np.ravel(image)
            np.save(testdataOutputFile, data)

        currentCntImage += 1
    print()

# functie folosita pentru a citi imaginile de train si de validation din fisierele cu imagini modificate
def ReadTestData():
    global testData, validationData, validationLabels

    testData = None
    validationData = None
    validationLabels = None

    # creez un array de 0-uri cu dimensiunea corespunzatoare imaginilor de test, respectiv validation
    testData = np.zeros((5149, dimensions[0], dimensions[1], 1))
    validationData = np.zeros((2000, dimensions[0], dimensions[1], 1))

    # path-ul pana la imaginile modificate
    pathToTestImages = "modified_images/" + str(dimensions[0]) + "x" + str(dimensions[1]) + "/test/test/"
    # listez toate imaginile din folderul cu imagini de test
    listImagesName = os.listdir(pathToTestImages)
    currentIdx = 0

    # parcurg fiecare nume de imagine din folder
    for imageName in tqdm(listImagesName):
        # citesc imaginea
        image = cv2.imread(pathToTestImages + imageName, cv2.IMREAD_GRAYSCALE)
        # dupa care ii fac reshape la dimensiunea dorita si o normalizez
        image = image.reshape((dimensions[0], dimensions[1], 1)) / 255.0
        # o adaug in array-ul de imagini de test
        testData[currentIdx] = image
        currentIdx += 1

    # fac acelasi lucru si pentru imaginile de validation
    pathToValidationImages = "modified_images/" + str(dimensions[0]) + "x" + str(dimensions[1]) + "/validation_predict/0/"
    listImagesName = os.listdir(pathToValidationImages)
    currentIdx = 0
    for imageName in tqdm(listImagesName):
        image = cv2.imread(pathToValidationImages + imageName, cv2.IMREAD_GRAYSCALE)
        image = image.reshape(dimensions[0], dimensions[1], 1) / 255.0
        validationData[currentIdx] = image
        currentIdx += 1

    # citesc labels pentru validation si train
    ReadLabels()

# functie folosita pentru a citi labelurile imaginilor de train si de validation
def ReadLabels():
    global trainLabels, validationLabels

    # creez un array de 0-uri cu dimensiunea corespunzatoare de validation, respectiv train
    validationLabels = np.zeros((2000))
    trainLabels = np.zeros((15000))

    # deschid fisierul cu labels
    f = open("validation_labels.txt", "r")
    currentIdx = 0
    # parcurg fiecare linie din fisier
    for line in f:
        line = line.split(',')
        # sar peste prima linie
        if line[0] == 'id':
            continue
        
        # adaug in array labelul citit
        validationLabels[currentIdx] =  int(line[1])
        currentIdx += 1

    # procedez la fel si pentru train
    f = open("train_labels.txt", "r")
    currentIdx = 0
    for line in f:
        line = line.split(',')
        if line[0] == 'id':
            continue

        trainLabels[currentIdx] =  int(line[1])
        currentIdx += 1

# functie folosita pentru a scrie rezultatele in fisierul de output
def WriteOutput(startPoint, testPredictions):
    # open the file in the write mode
    f = open('sample_submission.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(["id", "class"])
    i = startPoint
    for value in testPredictions:
        writer.writerow(["{:06d}".format(i), value])
        i += 1

    f.close()

# functie folosita pentru a crea imaginile cu dimensiuni diferite, plecand de la imaginile deja modificate, dar cu dimensiunea originala
def CreateNewDimensionImages():
    # dimensiunea noua pe care vreau sa o creez
    newDimensions = (128, 128)
    # calea de unde citesc imaginile modificate cu dimensiunea originala
    pathToOriginalDir = "modified_images/224x224_original/"
    # calea unde voi scrie noile imagini modificate
    pathToNewDir = "modified_images/" + str(newDimensions[0]) + "x" + str(newDimensions[1]) + "/"

    # verific daca folderul unde voi scrie noile imagini exista, daca nu il creez
    if not os.path.exists(pathToNewDir):
        os.makedirs(pathToNewDir)

    if not os.path.exists(pathToNewDir + "validation_predict/"):
        os.makedirs(pathToNewDir + "validation_predict/")
    if not os.path.exists(pathToNewDir + "validation_predict/0/"):
        os.makedirs(pathToNewDir + "validation_predict/0/")

    # listez toate directoarele din folderul de unde citesc imaginile
    listDirName = os.listdir(pathToOriginalDir)
    # trec prin fiecare director, si anume train, validation si test
    for dirName in listDirName:
        # verific daca folderele de train, validation si test exista unde vreau sa scriu imaginile, daca nu le creez
        if not os.path.exists(pathToNewDir + dirName):
            os.makedirs(pathToNewDir + dirName)

        # listez fiecare clasa din folderul respectiv
        listClassName = os.listdir(pathToOriginalDir + dirName + "/")
        # trec prin fiecare clasa
        for className in listClassName:
            # verific daca folderul cu clasa exista in folderul unde vreau sa scriu imaginile, daca nu il creez
            if not os.path.exists(pathToNewDir + dirName + "/" + className):
                os.makedirs(pathToNewDir + dirName + "/" + className)

            pathToImages = dirName + "/" + className + "/"
            # listez toate imaginile din folderul cu clasa
            listImagesName = os.listdir(pathToOriginalDir + pathToImages)

            # trec prin fiecare imagine
            for imageName in tqdm(listImagesName):
                # citesc imaginea
                image = cv2.imread(pathToOriginalDir + pathToImages + imageName)
                # scot noise-ul
                image =  cv2.fastNlMeansDenoising(image, None, 12, 5, 3)
                # fac resize la imagine
                image = cv2.resize(image, newDimensions)
                # scriu imaginea in noul folder creat
                cv2.imwrite(pathToNewDir + pathToImages + imageName, image)

                if "validation" in dirName:
                    cv2.imwrite(pathToNewDir + "validation_predict/0/" + imageName, image)

# functie foloista pentru a modifica imaginile originale si a le modificate in felul urmator:
# scot outlinerul prezent in unele imagini
# centrez imaginea
# rotesc imaginea astfel incat sa fie simetrica fata de axa Oy
def ModifyImages():
    global iInterval
    global prefixPath, dimensions
    global trainData, trainLabels, trainInterval
    global validationData, validationLabels, validationInterval
    global testData, testInterval

    # variabile ce le voi folosit la Algorithm Fill
    di = [-1, 0, 1, 0]
    dj = [0, 1, 0, -1]
    q = Queue()

    nrMaxImages = 22150
    cnt = [0, 0, 0]

    # verific daca folderul unde voi scrie imaginile modificate exista, daca nu il creez
    if not os.path.exists(prefixPath + "modified_images/"):
        os.makedirs(prefixPath + "modified_images/")

    # lista cu directoarele pe care trebuie sa le verific daca exista, daca nu le creez
    listOfDirectoriesToCheckIfExists = ["", "/train", "/validation", "/test", "/train/0", "/train/1", "/validation/0", "/validation/1", "/test/test", "/validation_predict/0"]
    for directory in listOfDirectoriesToCheckIfExists:
        if not os.path.exists(prefixPath + "modified_images/224x224_original" + directory):
            os.makedirs(prefixPath + "modified_images/224x224_original" + directory)

    # citesc labels
    ReadLabels()

    # trec prin indexul fiecarei imagini
    for k in tqdm(range(1, nrMaxImages)):
        # citesc imaginile din data
        image_ = cv2.imread("data/" + "{:06d}".format(k) + ".png", cv2.IMREAD_GRAYSCALE)
        # cresc constrastul imaginii
        image_ = cv2.convertScaleAbs(image_, alpha = 2.5, beta = 0)

        # stochez valoarea minima din imagine dupa care daca valoarea unui pixel este mai mica decat
        # aceasta valoare, il transform in 0
        minPixelVal = np.min(image_[0][0])
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                if image_[i][j] <= minPixelVal:
                    image_[i][j] = 0

        # aplic threshold peste imagine si o transform in black-white
        image = cv2.threshold(cv2.blur(image_, (2, 2), 1), 5, 255, cv2.THRESH_BINARY)[1]

        # matrice folosita pentru a verifica daca am trecut printr-o pozitie sau nu
        wasSeen = [[False for j in range(dimensions[1])] for i in range(dimensions[0])]
        
        # caut un punct de start aproape de mijloc astfel incat valoarea pixelului sa fie 255 (alb)
        startPoint  = (0, 0)
        for i in range(-20, 21):
            for j in range(-20, 21):
                if image[dimensions[0] // 2 + i][dimensions[1] // 2 + j] == 255:
                    startPoint = (dimensions[0] // 2 + i, dimensions[1] // 2 + j)
                    break
        
        # aceasta variabila ma va ajuta sa tin x-ul cel mai mic al creierului si x-ul cel mai mare al creierului
        minMaxX = startPoint
        # incep algoritmul de Fill si parcurg vecinii unei pozitii atat timp cand acel vecin
        # nu a fost inca vizitat si valoarea pixelului lui este 255 (alb)
        # adaug in coada pozitia de start
        q.put(startPoint)
        # marchez pozitia de start ca fiind vizitata
        wasSeen[startPoint[0]][startPoint[1]] = True

        # cat timp coada inca nu este goala
        while q.empty() == False:
            # extrag din coada pozitia curenta
            (posi, posj) = q.get()
            # ma uit la vecinii pozitiei curente
            for t in range(4):
                # calculez pozitia vecinului
                ni = posi + di[t]
                nj = posj + dj[t]
                # verific conditiile prezentate mai sus
                if ni >= 0 and ni < dimensions[0] and nj >= 0 and nj < dimensions[1] and wasSeen[ni][nj] == False and image[ni][nj] == 255:
                    # adaug in coada si marchez ca vizitat
                    q.put((ni, nj))
                    wasSeen[ni][nj] = True
                    if nj < minMaxX[0]:
                        minMaxX = (nj, minMaxX[1])
                    if nj > minMaxX[1]:
                        minMaxX = (minMaxX[0], nj)
        
        # parcurg din nou fiecare pixel din imagine, iar daca pixelul nu a fost vizitat, il transform in 0
        # daca y-ul pixelului este mai mic decat jumatatea inaltimei imaginii, iar x-ul se afla in intervalul minMaxX, il las asa cum este
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                if j >= minMaxX[0] and j <= minMaxX[1] and i <= dimensions[0] // 2:
                    continue
                if wasSeen[i][j] == False :
                    image_[i][j] = 0
                    image[i][j] = 0

        # calculez momentele imaginii pentru a afla centrul al imaginii
        momentsShape = cv2.moments(image)
        if momentsShape["m00"] != 0:
            # extract centrul imaginii
            centralX = int(momentsShape["m10"] / momentsShape["m00"])
            centralY = int(momentsShape["m01"] / momentsShape["m00"])

            # calculez cu cat trebuie sa mut imaginea pentru a fi centrata
            newX = dimensions[0] // 2 - centralX
            newY = dimensions[1] // 2 - centralY

            # calculez matricea de translatatie
            translationMatrix = np.float32([[1, 0, newX], [0, 1, newY]])   

            # translatez atat imaginea initiala cat si imaginea modificata
            image_ = cv2.warpAffine(image_, translationMatrix, (dimensions[0], dimensions[1])) 
            image = cv2.warpAffine(image, translationMatrix, (dimensions[0], dimensions[1])) 

        # creez o noua masca cu threshold pentru imaginea curenta
        image = cv2.threshold(cv2.GaussianBlur(image, (3, 3), 1), 3, 255, cv2.THRESH_BINARY)[1]
        # gasesc contururile imaginii
        contours, _ = cv2.findContours (image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # extract conturul de arie maxima
        try:
            c = max(contours, key = cv2.contourArea)
        except:
            c = []
        
        # daca conturul are mai mult de 5 puncte 
        if len(c) >= 5:
            # incerc sa gasesc elipsa ce incadreaza conturul
            (x, y), (MA, ma), angle = cv2.fitEllipse(c)

            # normalizez unghiului elipsei
            if angle >= 90:
                angle -= 90
            else:
                angle += 88

            # creez matricea de translatatie pentru a roti imaginea
            matrixRotation = cv2.getRotationMatrix2D((x, y), angle - 90, 1)

            # rotesc imaginea
            image_ = cv2.warpAffine(image_, matrixRotation, (image_.shape[1], image_.shape[0]), cv2.INTER_CUBIC)

        # scriu imaginea in directorul corespunzator in functie de valoarea lui k
        if k < trainInterval[1]:
            if trainLabels[k - 1] == 0:
                cv2.imwrite("modified_images/224x224_original/train/0/" + '{:06}'.format(k) + ".png", image_)
            else:
                cv2.imwrite("modified_images/224x224_original/train/1/" + '{:06}'.format(k) + ".png", image_)
        elif k < validationInterval[1]:
            if validationLabels[k - validationInterval[0]] == 0:
                cv2.imwrite("modified_images/224x224_original/validation/0/" + '{:06}'.format(k) + ".png", image_)
            else:
                cv2.imwrite("modified_images/224x224_original/validation/1/" + '{:06}'.format(k) + ".png", image_)
            cv2.imwrite("modified_images/224x224_original/validation_predict/0/" + '{:06}'.format(k) + ".png", image_)
        else:
            cv2.imwrite("modified_images/224x224_original/test/test/" + '{:06}'.format(k) + ".png", image_)

model = None
reduce_lr = None

# functie folosita pentru a crea modelul
def CreateModelAndCompile():
    global model
    global dimensions

    global reduce_lr

    # creez un model de tip Sequential
    model = Sequential()

    # blocul 1
    # adaug 2 straturi de convolutie cu 64 de filtre, 1 straturi de max pooling si 1 strat de dropout
    model.add(Conv2D(64, (3, 3), activation = "relu", input_shape = dimensions))
    model.add(Conv2D(64, (3, 3), activation = "relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # blocul 2
    # adaug 2 straturi de convolutie cu 128 de filtre, 1 straturi de max pooling si 1 strat de dropout
    model.add(Conv2D(128, (3, 3), activation = "relu"))
    model.add(Conv2D(128, (3, 3), activation = "relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # blocul 3
    # adaug 2 straturi de convolutie cu 256 de filtre, 1 straturi de max pooling si 1 strat de dropout
    model.add(Conv2D(256, (3, 3), activation = "relu"))
    model.add(Conv2D(256, (3, 3), activation = "relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # aplic flatten pe rezultatul blocului 3
    model.add(Flatten())
    
    # adaug 1 strat dense cu 128 de neuroni si un strat de dropout
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))

    # adaug stratul de output cu 1 neuron si functia de activare sigmoid
    model.add(Dense(1,activation="sigmoid"))

    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.75, patience = 15, mode='auto', verbose = 1, min_lr=0.000075)

    # model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # compilez modelul folosind functia de loss binary_crossentropy, optimizerul Adam si metrica accuracy si F1Score
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tfa.metrics.F1Score(num_classes=2, average='micro', threshold=0.5)])

# functie folosita pentru a incarca modelul
def LoadModel():
    global prefixPath, modelEpoch
    global model

    model = load_model(prefixPath + "models/model_" + '{:02}'.format(modelEpoch) + ".h5")

# functie folosita pentru a antrena modelul
def PredictUsingCNN(maxNumEpochs = 0):
    global prefixPath, load_model_name
    global dimensions
    global trainData, trainLabels
    global validationData, validationLabels
    global testData
    global model
    global reduce_lr

    # verific daca fisierul models exista
    if not os.path.exists(prefixPath + "models"):
        os.makedirs(prefixPath + "models")

    # creez callback-ul pentru checkpoint
    checkpoint_path = "models/model_{epoch:02d}.h5"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False, save_freq='epoch')

    if maxNumEpochs != 0:
        # antrenez modelul
        # model.fit(trainGenerator, epochs = maxNumEpochs, steps_per_epoch = len(trainGenerator), validation_data = validationGenerator, callbacks=[checkpoint_callback, reduce_lr], class_weight = {0: 1., 1: 1.5})
        model.fit(trainGenerator, epochs = maxNumEpochs, steps_per_epoch = len(trainGenerator), validation_data = validationGenerator, callbacks=[checkpoint_callback], class_weight = {0: 1., 1: 1.5})

# functie folosita pentru a da predict pe generatorul de validation
def PredictOnValidationUsingGenerator():
    global model

    validationPrediction = model.predict(validationGenerator)
    validationPrediction = np.round(validationPrediction.reshape(len(validationPrediction))).astype(int)

    accuracy = accuracy_score(validationLabels, validationPrediction)
    f1_score_val = f1_score(validationLabels, validationPrediction)
    matrix = confusion_matrix(validationLabels, validationPrediction)
    print("Accuracy for:", accuracy)
    print("f1_score for:", f1_score_val)
    print("Confusion matrix:\n", matrix)

# functie folosita pentru a da predict pe generatorul de test
def PredictOnTestUsingGenerator():
    global model

    testPrediction = model.predict(testGenerator)
    testPrediction = np.round(testPrediction.reshape(len(testPrediction))).astype(int)
    WriteOutput(17001, testPrediction)

# ModifyImages()

# CreateNewDimensionImages()

"""
trainData = np.zeros((15000, dimensions[0], dimensions[1]))

imagesList = os.listdir("modified_images/128x128/train/0/")
currentIndex = 0
for imageName in tqdm(imagesList):
    image = cv2.imread("modified_images/128x128/train/0/" + imageName, cv2.IMREAD_GRAYSCALE)
    trainData[currentIndex] = image
    currentIndex += 1

imagesList = os.listdir("modified_images/128x128/train/1/")
for imageName in tqdm(imagesList):
    image = cv2.imread("modified_images/128x128/train/1/" + imageName, cv2.IMREAD_GRAYSCALE)
    trainData[currentIndex] = image
    currentIndex += 1

trainData = np.expand_dims(trainData, axis = -1)

trainGenerator = trainDataGen.flow_from_directory(pathToImages + "train/", target_size = (dimensions[0], dimensions[1]), batch_size = 48, class_mode = 'binary')
validationGenerator = testDataGen.flow_from_directory(pathToImages + "validation/", target_size = (dimensions[0], dimensions[1]), batch_size = 48, class_mode = 'bina90ry')
testGenerator = testDataGen.flow_from_directory(pathToImages + "test/", target_size = (dimensions[0], dimensions[1]), batch_size = 48, class_mode = None)

"""

# creez ImageDataGeneratorul pentru train si validation, respectiv test
trainDataGen = ImageDataGenerator(horizontal_flip = True, rescale = 1. / 255, rotation_range = 15, width_shift_range = 0.1, height_shift_range = 0.1, shear_range = 0.1, zoom_range = 0.15, fill_mode = "nearest")                                         
testDataGen = ImageDataGenerator(rescale = 1. / 255)

# calea catre imaginile de train, validation si test
pathToImages = "modified_images/" + str(dimensions[0]) + "x" + str(dimensions[1]) + "/"

trainGenerator = trainDataGen.flow_from_directory(pathToImages + "train/", color_mode='grayscale', target_size = (dimensions[0], dimensions[1]), batch_size = 90, class_mode = 'binary', shuffle = True)
validationGenerator = testDataGen.flow_from_directory(pathToImages + "validation/", color_mode='grayscale', target_size = (dimensions[0], dimensions[1]), batch_size = 90, class_mode = 'binary', shuffle = False)
testGenerator = testDataGen.flow_from_directory(pathToImages + "test/", color_mode='grayscale', target_size = (dimensions[0], dimensions[1]), batch_size = 90, class_mode = None, shuffle = False)

CreateModelAndCompile()

# modelEpoch = 1
# LoadModel()

model.summary()

maxNumEpochs = 1
print("Training model...")
PredictUsingCNN(maxNumEpochs)
print("Done training model!")

print("Reading test and validation data...")
ReadTestData()
print("Done reading test and validation data!")

# tested on 181, 195
validationGenerator = testDataGen.flow_from_directory(pathToImages + "validation_predict/", color_mode='grayscale', target_size = (dimensions[0], dimensions[1]), batch_size = 90, class_mode = 'binary', shuffle = False)

PredictOnValidationUsingGenerator()
PredictOnTestUsingGenerator()