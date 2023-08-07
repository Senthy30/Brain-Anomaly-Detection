import csv
import numpy as np
import cv2
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

widthImage = 224
heightImage = 224
# dimensiunile imaginilor originale

wasSetted = False
imageRepeats = 9

# functie pe care o folosesc pentru data augmentation
# fiecare imagine pe care o modific, o voi salva apoi in fisierul f sub forma unui numpy array 1D
def AdjustImage(resized, f):
    cntRepeats = 0
    # ajustez brightness-ul imaginii cu diferite valori
    for k_gradient in range(-2, 4):
        image = cv2.add(resized, k_gradient * 4)

        data = np.ravel(image)
        np.save(f, data)
        cntRepeats += 1
    
    # ajustez blurul imaginii cu diferite valori
    for k_blur in range(0, 3):
        image = cv2.GaussianBlur(resized, (2 * k_blur + 1, 2 * k_blur + 1), 0)

        data = np.ravel(image)
        np.save(f, data)
        cntRepeats += 1

    # returnez numarul de copii a imaginii
    return cntRepeats

# functie folosita pentru a citi imaginile, a le procesa, fara a face undersampling
def ReadImagesWriteNumPyArray(startPoint, endPoint, outputFilename, scale_percent = 100, adjustImage = False):
    global imageRepeats, widthImage, heightImage, wasSetted
    global prefixPath

    length = endPoint - startPoint
    currentProcent = 0
    lastProcent = -1

    # numele fisierului unde vreau sa scriu numpy arrays corespunzatoare fiecarei imagini
    outputFilename = prefixPath + "data_array/" + outputFilename
    with open(outputFilename, 'wb') as f:
        for i in range(startPoint, endPoint):
            # iau numele imaginii
            imageFilename = prefixPath + "data/" + "{:06d}".format(i) + ".png"
            # citesc imaginea
            image = cv2.imread(imageFilename, cv2.IMREAD_GRAYSCALE)

            # calculez noul width and height
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            
            # fac resize la imagine
            resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

            if wasSetted == False:
                wasSetted = True
                widthImage = resized.shape[0]
                heightImage = resized.shape[1]

            # daca adjustImage == True, atunci salvez imaginile in fisier cu data augmentation, iar daca nu, le salvez doar in format original
            if adjustImage == True:
                # ajustez imaginea
                cntRepeats = AdjustImage(resized, f)

                if cntRepeats != imageRepeats:
                    imageRepeats = cntRepeats
            else:
                data = np.ravel(resized)
                np.save(f, data)

            # calculez cat % din total am procesat
            currentProcent = int((i - startPoint + 1) / length * 100)
            if currentProcent != lastProcent:
                print("Complete: " + str(currentProcent) + "%")
                lastProcent = currentProcent

# functie folosita pentru a citi imaginile, a le procesa si a face undersampling
# functie este la fel ca cea de sus, doar ca mai apare un for si numarul de imagini cu label 0 sau 1 procesate
# astfel incat sa pot selecta doar un anumit numar de imaginii cu 0 si 1
def ReadImagesWriteNumPyArrayBalancing(startPoint, endPoint, outputFilenameTrain, outputFilenameLabels, scale_percent = 100, maxCntAllowed = (-1, -1)):
    global imageRepeats, widthImage, heightImage, wasSetted
    global trainLabelsOne, trainIntervalY
    global prefixPath

    length = endPoint - startPoint
    currentProcent = 0
    lastProcent = -1

    outputFilenameTrain = prefixPath + "data_array/" + outputFilenameTrain
    outputFilenameLabels = prefixPath + "labels_array/" + outputFilenameLabels
    with open(outputFilenameTrain, 'wb') as f:
        with open(outputFilenameLabels, "wb") as g:
            cnt_0 = 0
            cnt_1 = 0
            cnt_0_valid = 0
            cnt_1_valid = 0

            for i in range(startPoint, endPoint):
                # daca maxCntAllowed[0] == -1, atunci inseamna ca voi selecta toate imaginile, nu mai fac undersampling
                if maxCntAllowed[0] != -1:
                    # verific daca label-ul imaginii curente este 0 sau 1
                    # dupa ce am verific, incrementez cnt-ul corespunzator si verific sa nu depasesc limita impusa
                    # daca depasesc, trec la urmatoare imagine
                    if trainLabelsOne[i - 1] == 0:
                        cnt_0 += 1
                        if cnt_0 >= maxCntAllowed[0]:
                            continue
                        cnt_0_valid += 1
                    elif trainLabelsOne[i - 1] == 1:
                        cnt_1 += 1
                        if cnt_1 >= maxCntAllowed[1]:
                            continue
                        cnt_1_valid += 1

                imageFilename = prefixPath + "data/" + "{:06d}".format(i) + ".png"
                image = cv2.imread(imageFilename, cv2.IMREAD_GRAYSCALE)

                width = int(image.shape[1] * scale_percent / 100)
                height = int(image.shape[0] * scale_percent / 100)
                dim = (width, height)
                
                resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

                if wasSetted == False:
                    wasSetted = True
                    widthImage = resized.shape[0]
                    heightImage = resized.shape[1]

                cntRepeats = AdjustImage(resized, f)

                if cntRepeats != imageRepeats:
                    imageRepeats = cntRepeats

                for k in range(imageRepeats):
                    np.save(g, trainLabelsOne[i - 1])

                currentProcent = int((i - startPoint + 1) / length * 100)
                if currentProcent != lastProcent:
                    print("Complete: " + str(currentProcent) + "%")
                    lastProcent = currentProcent

            trainIntervalY = cnt_0_valid + cnt_1_valid + 1


# aceasta functie are scopul de a citi fiecare label corespunzator imaginii si a le scrie intr-un fisier .npy
# pentru a citi mai repede cand este nevoie
def WriteLabelsAsNPY(inputFilename, outputFilename, interval, numberOfRepeats = 1):
    global prefixPath

    # fisierul de unde citesc labels
    inputFilename = prefixPath + inputFilename
    # fisierul unde le voi scrie
    outputFilename = prefixPath + "labels_array/" + outputFilename

    inputFile = open(inputFilename, "r")
    # sar peste prima linie
    str_line = inputFile.readline()
    with open(outputFilename, 'wb') as f:
        # citesc linia
        str_line = inputFile.readline()
        currentLine = 0
        # cat timp nu am ajuns la finalul fisierului
        while str_line:
            # scot \n din linie si dau split dupa ,
            str_line = str_line.replace("\n", "").split(',')

            currentLine += 1
            if(currentLine < interval[0]):
                str_line = inputFile.readline()
                continue
            if(currentLine >= interval[1]):
                break
            
            # salvez acelasi label de numberOfRepeats ori deoarece am facut data augmentation
            for k in range(numberOfRepeats):
                np.save(f, np.array(str_line[1]))

            # citesc urmatoarea linie
            str_line = inputFile.readline()

# functie folosita pentru a afisa imaginile intr-un fisier si a vedea transformarile efectuate 
def ReadNumPyArrayWriteImages(startPoint, endPoint, inputFilename):
    global widthImage, heightImage
    global prefixPath

    # check if restored_images folder exists
    if not os.path.exists(prefixPath + 'restored_images/'):
        os.makedirs(prefixPath + 'restored_images/')

    inputFilename = prefixPath + "data_array/" + inputFilename
    with open(inputFilename, 'rb') as f:
        for i in range(startPoint, endPoint):
            np_array = np.load(f)
            image = np.reshape(np_array, (widthImage, heightImage))
            cv2.imwrite(prefixPath + 'restored_images/' + "{:06d}".format(i) + '.png', image)


# functie folosita pentru a citi imaginile din fisierul "inputFilename".npy
def ReadNumPyArray(startPoint, endPoint, inputFilename):
    np_array = []
    with open(inputFilename, 'rb') as f:
        for i in range(startPoint, endPoint):
            try:
                np_array.append(np.load(f))
            except:
                break
    return np.array(np_array)

# functie folosita pentru a citi labels din fisierul "inputFilename".npy
def ReadNumPyArrayLabels(startPoint, endPoint, inputFilename):
    l_array = []
    with open(inputFilename, 'rb') as f:
        for i in range(startPoint, endPoint):
            try:
                np_array = np.load(f)
                l_array.append(np_array)
            except:
                break
    return np.array(l_array)

# functie folosita pentru a da predict folosind modelul KNN
# parametru getAccuracy este folosit pentru a vedea daca momentan testez pe validation sau pe test
def PredictOn(trainDataFilename, trainLabelsFilename, testDataFilename, testLabelsFilename, trainInterval, testInterval, getAccuracy = True):
    global imageRepeats

    # citesc train data
    print("Loading train data...")
    trainData = ReadNumPyArray(trainInterval[0], trainInterval[1], trainDataFilename)
    print("Complete!\n\nLoading train labels...")
    # citesc train labels
    trainLabels = ReadNumPyArrayLabels(trainInterval[0], trainInterval[1], trainLabelsFilename)
    print("Complete!\n\nLoading test data...")

    # citesc test data care poate fi validation sau test
    testData = ReadNumPyArray(testInterval[0], testInterval[1], testDataFilename)
    print("Complete!\n\nLoading test labels...")
    if getAccuracy == True:
        # citesc test labels daca momentan testez pe validation
        testLabels = ReadNumPyArrayLabels(testInterval[0], testInterval[1], testLabelsFilename).astype(int)

    print("Complete!\n\nRunning predict...")
    multiplier = 1
    # for-ul a fost creat pentru a testa diferite valori de num_neighbors
    for num_neighbors in [21]:
        new_num_neighbors = (multiplier * num_neighbors)
        # daca numarul de vecini este par, il scad cu 1, deoarece in cazul unui vot egal, nu vreau sa alega random una dintre clase
        if new_num_neighbors % 2 == 0:
            new_num_neighbors -= 1

        # initializez modelul KNN cu metrica euclidiana
        print("Creating KNN with " + str(new_num_neighbors) + " neighbors...")
        knn = KNeighborsClassifier(n_neighbors = new_num_neighbors, metric='euclidean')
        print("Complete!\n\nRunning fitting data...")
        # antrenez modelul pe train data
        knn.fit(trainData, trainLabels)
        print("Complete!\n\nRunning predict...")
        # dau predict pe test data
        testPredictions = knn.predict(testData)
        print("Complete!\n")

        if getAccuracy == True:
            # daca momentan testez pe validation, calculez accuracy, f1_score, confusion matrix, recall si precision

            accuracy = accuracy_score(testLabels, testPredictions)
            f1_score_val = f1_score(testLabels, testPredictions)
            matrix = confusion_matrix(testLabels, testPredictions)

            disp = ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels = knn.classes_)
            disp.plot()
            plt.show()

            print("Confusion matrix for " + str(new_num_neighbors) + ":\n", matrix)
            print("Accuracy for " + str(new_num_neighbors) + ":", accuracy)
            print("f1_score for " + str(new_num_neighbors) + ":", f1_score_val)
            print("Recall for " + str(new_num_neighbors) + ":", recall_score(testLabels, testPredictions, average="binary"))
            print("Precision for " + str(new_num_neighbors) + ":", precision_score(testLabels, testPredictions, average="binary"))

    if getAccuracy == False:
        # daca momentan testez pe test, scriu rezultatele in fisierul "sample_submission.csv"
        WriteOutput(17001, testPredictions)

def WriteOutput(startPoint, testPredictions):
    # deschid fisierul pentru scriere
    f = open('sample_submission.csv', 'w', newline='')

    # creez un nou writer
    writer = csv.writer(f)
    # scriu prima linie
    writer.writerow(["id", "class"])
    i = startPoint
    # pentru fiecare predictie, scriu id-ul si valoarea
    for value in testPredictions:
        writer.writerow(["{:06d}".format(i), value])
        i += 1

    f.close()

# functie folosita pentru a citi labels din fisierul "train_labels.txt" si a le salva in variabila globala "trainLabelsOne"
def ReadTrainLabelsTxt():
    global trainLabelsOne
    trainLabelsOne = []

    f = open("train_labels.txt", "r")
    for line in f.readlines():
        line = line.replace("\n", "").split(',')
        if line[0] == 'id':
            continue

        trainLabelsOne.append(int(line[1]))

# daca balacingClasses este True, atunci fac undersampling
balacingClasses = True
# daca reloadData este True, atunci citesc din nou fiecare imagine, o procesez, fac data augmentation si o scriu in fisierul .npy
reloadData = False
# daca typeOfTest == 0, atunci testez pe validation
# daca typeOfTest == 1, atunci testez tot pe train cu un anumit interval de imagini
# daca typeOfTest == 2, atunci testez pe test
typeOfTest = 0

trainIntervalX = 1
trainIntervalY = 15001
testIntervalX = 15001
testIntervalY = 17001

# numarul maxim de imagini pe care le citesc din fiecare clasa cand fac undersampling
maxCntAllowed = (3000, 2400)

trainDataFilename = "train_data_array_smaller.npy"
trainLabelsFilename = ("train_labels.txt", "train_labels.npy")
testDataFilename = "validation_data_array_smaller.npy"
testLabelsFilename = ("validation_labels.txt", "validation_labels.npy")

# prefixul pana la folderul in care se afla toate fisierele
prefixPath = ""

# check if data_array folder exists
if not os.path.exists(prefixPath + "data_array"):
    os.makedirs(prefixPath + "data_array")

# check if labels_array folder exists
if not os.path.exists(prefixPath + "labels_array"):
    os.makedirs(prefixPath + "labels_array")


#WriteLabelsAsNPY(trainLabelsFilename[0], "train_labels_one.npy", (1, 15001), 1) 
trainLabelsOne = None
ReadTrainLabelsTxt()

if typeOfTest == 1:
    
    trainIntervalX = 1
    trainIntervalY = 15001
    testIntervalX = 1
    testIntervalY = 15001
    testLabelsFilename = ("train_labels.txt", "validation_labels.npy")

elif typeOfTest == 2:

    testIntervalX = 17001
    testIntervalY = 22150
    testDataFilename = "test_data_array_smaller.npy"

if reloadData == True and balacingClasses == False:

    ReadImagesWriteNumPyArray(trainIntervalX, trainIntervalY, trainDataFilename, 25, True)

    WriteLabelsAsNPY(trainLabelsFilename[0], trainLabelsFilename[1], (trainIntervalX, trainIntervalY), imageRepeats)  

    ReadImagesWriteNumPyArray(testIntervalX, testIntervalY, testDataFilename, 25)
    if typeOfTest == 1:
        WriteLabelsAsNPY(testLabelsFilename[0], testLabelsFilename[1], (testIntervalX, testIntervalY))  
    elif typeOfTest == 0:
        WriteLabelsAsNPY(testLabelsFilename[0], testLabelsFilename[1], (1, 2001))  
    
    #ReadNumPyArrayWriteImages(1, 100, trainDataFilename)

elif reloadData == True and balacingClasses == True:
    ReadImagesWriteNumPyArrayBalancing(trainIntervalX, trainIntervalY, trainDataFilename, trainLabelsFilename[1], 25, maxCntAllowed)

    ReadImagesWriteNumPyArray(testIntervalX, testIntervalY, testDataFilename, 25)
    if typeOfTest == 1:
        WriteLabelsAsNPY(testLabelsFilename[0], testLabelsFilename[1], (testIntervalX, testIntervalY))  
    elif typeOfTest == 0:
        WriteLabelsAsNPY(testLabelsFilename[0], testLabelsFilename[1], (1, 2001))  

trainDataFilename = prefixPath + "data_array/train_data_array_smaller.npy"
trainLabelsFilename = prefixPath + "labels_array/train_labels.npy"
trainInterval = (trainIntervalX * imageRepeats, trainIntervalY * imageRepeats)
# imageRepeats = 9
# trainInterval = (1, 41742 - 8)

testDataFilename = prefixPath + "data_array/validation_data_array_smaller.npy"
if typeOfTest == 2:
    testDataFilename = prefixPath + "data_array/test_data_array_smaller.npy"
testLabelsFilename = prefixPath + "labels_array/validation_labels.npy"
testInterval = (testIntervalX, testIntervalY)

PredictOn(trainDataFilename, trainLabelsFilename, testDataFilename, testLabelsFilename, trainInterval, testInterval, typeOfTest != 2) 