import numpy as np
import pdb


def loadDataSet():
    postingList = []
    classVec = []
    with open('./text.csv', 'r') as f:
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            line = line.split(',')
            classVec.append(int(line[-1].rstrip()))
            postingList.append(line[0:-1])
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    reVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            reVec[vocabList.index(word)] += 1
    return reVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbsive = sum(trainCategory) / numTrainDocs
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbsive


def classfiNB(vec2Classfy, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classfy * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classfy * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    pdb.set_trace()
    trainMat = []
    for postDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    pdb.set_trace()
    testEntry = ['love', 'my', 'stupid', 'dog']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(classfiNB(thisDoc, p0V, p1V, pAb))


def main():
    testingNB()


if __name__ == '__main__':
    main()
