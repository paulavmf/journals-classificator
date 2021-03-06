# #
#Year 2017/2018
#Text Mining
#Master's Degree in Intelligent Systems and Numerical Applications of Engineering
#University of Las Palmas de Gran Canaria
#Paula Victoria Moreno Fajardo


import nltk
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.decomposition import PCA
import glob



#OBTAIN LISTING TEXT (each item is a document)

# for titles
def GetText_1linea(revista, cadena1):
    revista = revista[1:12]
    texto = []
    for f in revista:
        revista = open(f, 'r')
        n = len(revista.readlines())
        revista.seek(0)
        for i in range(0, n):
            linea = revista.readline()
            if linea.find(cadena1) != -1:
                inicio = linea.find(cadena1) + len(cadena1)
                texto.append(linea[inicio:-3])

        revista.close()
    return texto

#for abstract and keywords
def GetText_parrafo(revista, cadena1, cadena2):
    texto = []
    revista = revista[1:12]
    for f in revista:
        revista = open(f, 'r')
        n = len(revista.readlines())
        revista.seek(0)
        j = 0
        for i in range(0, n):
            linea = revista.readline()
            if linea.find(cadena1) != -1:
                j = j + 1
                inicio = linea.find(cadena1) + len(cadena1)
                para = (linea[inicio:])
                while para.find(cadena2) == -1:
                    para = para + revista.readline()
                texto.append(para)

        revista.close()

    return texto



#PARSING TEXT
# get only words from the text. no punctuation signs, no stop And lemmatized words

def tratamiento(texto):

    # paso a minusculas
    for i in range(len(texto)):
        texto[i] = texto[i].lower()

    # set de stopwords
    stopWords = set(stopwords.words('english'))
    # expresion regular para obtener solo palabras
    tokenizer = RegexpTokenizer(r'\w+')
    # LEMATIZADOR
    porter = nltk.PorterStemmer()

    for i in range(len(texto)):
        TITLE = []
        tokens = tokenizer.tokenize(texto[i])
        for w in tokens:
            w = porter.stem(w)
            if w not in stopWords:
                TITLE.append(w)

        texto[i] = " ".join(TITLE)

    return texto



# returns me a single matrix with all the documents and their label in the last column

def MatrixDocs(textoP, textoS):
    labels = []
    for i in range(len(textoP)):
        labels.append(1)
    for i in range(len(textoS)):
        labels.append(2)

    print("DOCS EN REVISTA PATTERN:", len(textoP, ))
    print("DOSC EN REVISTA SYSTEM:", len(textoS))
    textoP.extend(textoS)

    return textoP, labels


#CLASIFICATION


def clasification(data, gnb, ratio):
    TRAIN = data[:round(len(data) * ratio), :-1]
    labels_TRAIN = data[:round(len(data) * ratio), -1]

    TEST = data[round(len(data) * ratio):, :-1]

    labels_TEST = data[round(len(data) * ratio):, -1]

    result = np.zeros(len(labels_TEST))
    gnb.fit(TRAIN, labels_TRAIN)
    for i in range(len(TEST)):
        result[i] = gnb.predict([TEST[i]])

    acc = sum((result == labels_TEST)) / len(labels_TEST)

    return acc



#ATTRIBUTE SELECTION
def sffs(X, y, m, nombres, clf,
         ratio):  # Docs matrix, Vocabulary, number of attributes, vocabulary, classification method
     # Get number of features
    (numSamples, numFeatures) = X.shape

    # Name-position dictionary
    featurePos = dict()
    for i in range(numFeatures):
	# le da un numero (posicion) a cada atributo
        featurePos[nombres[i]] = i  
    print("featurePos", featurePos)

    # Matrix with selected attributes
    accuracy = []
    featureSol = []
    i = 0
    S = np.zeros((numSamples, 1))
    SF = np.zeros((numSamples, 1))
    while len(featureSol) < m:
        acc = []
        (_, c) = S.shape
        for f in nombres:
            pos = featurePos[f]  # Index
            # in the first iteration I create the first row, from the second iteration I insert rows

            if i == 0:
                SF[:, i] = X[:, pos]
            else:
                SF = np.insert(S, c, values=X[:, pos], axis=1)
            clf = clf.fit(SF[: round(len(X) * ratio)], y[: round(len(X) * ratio)])
            acc.append(clf.score(SF[round(len(X) * ratio):], y[round(len(X) * ratio):]))
        selecF = np.argmax(acc)
        # Updated with the result
        featureSol.append(nombres[selecF])
        del nombres[selecF]
        # in the first iteration create the first row, from the second iteration insert rows

        if i == 0:
            S[:, i] = X[:, selecF]
        else:
            S = np.insert(S, c, values=X[:, selecF], axis=1)
        accuracyF = acc[selecF]
        # print('accuracyF: {}'.format(accuracyF))
        # I go backwards as long as the accuracy backwards is greater than forwards

        accuracyB = 100

        if i > 0:
	    # compare the result with the previous result
            while accuracyB > accuracyF and len(featureSol) > 2:

                acc = []
                 # try every possibility going backwards
                for f in featureSol:
                    pos = list(featureSol).index(f)
                    SB = np.delete(S, pos, 1)
                    clf = clf.fit(SB[: round(len(X) * ratio)], y[: round(len(X) * ratio)])
                    acc.append(clf.score(SB[round(len(X) * ratio):], y[round(len(X) * ratio):]))
                selecB = np.argmax(acc)
                accuracyB = acc[selecB]

                # print('accuracyB: {}'.format(accuracyB))
                # update
                if accuracyB > accuracyF:
                    nombres.append(featureSol[selecB])
                    del featureSol[selecB]
                    S = np.delete(S, selecB, 1)
                    accuracyF = accuracyB 
                    # in this way returns forward if accuracy stops increasing

        i = i + 1
    if accuracyB > accuracyF:
        print(accuracyB)
    else:
        print(accuracyF)


    return featureSol


def JournalMultiClasisfier(TEXTO, nombre, labelsTitles, ratio, rep, metodo, REDUCTION):
    print("%%%%%%%%%%%%%OUTPUTS FOR:", nombre, "USING PCA%%%%%%%%%%%%%%%")

    # diccionario con los parametros para la vectorizacion
    ngram = {}
    ngram["1 Palabra"] = TfidfVectorizer()
    ngram["1 y 2 Palabras"] = TfidfVectorizer(ngram_range=(1, 2), max_df=1.0, min_df=1)

    for g in ngram:

        print("%%%%%%%%%%%USING", g, "%%%%%%%%%%%%%")
        vectorizer = ngram[g]

        T = vectorizer.fit_transform(TEXTO).toarray()
        Vocabulary = list(vectorizer.vocabulary_.keys())

        # REDUCCION DE LA DIMENSIONALIDAD
        if REDUCTION == True:
            print("DIMENSIONALITY REDUCTION")
            pca = PCA(n_components= 10)
            T = pca.fit_transform(T)

        Doc, V = T.shape
        print("NUM DOSC", Doc)
        print("NUM OF WORDS", V)

        # PONGO LOS LABELS EN LA ULTIMA COLUMNA
        TL = np.insert(T, V, values=labelsTitles, axis=1)

        accT = []

        for m in metodo:
            print("METHOD: ", m)

            for r in ratio:
                for i in range(rep):
                    # MEZCLO LA MATRIZ
                    np.random.shuffle(TL)
                    accT.append(clasification(TL, metodo[m], r))

            print("accuracy:", sum(accT) / len(accT))
            print("................................................................")
            print("................................................................")

            #utilizo seleccion de 50 atributos SFFS SOLO PARA EL ARTÍCULOS, utilizando Naive Bayes y 1 palabra de granulidad
            if REDUCTION == False and nombre== "ARTICULOS" and m =='Decision Tree' and g =="1 Palabra":
                print("attribute selection, METHOD SFFS")
                res = sffs(TL[:,:-1], TL[:,-1], 10, Vocabulary,metodo[m],0.6)
                print("the selected words in", nombre, "ARE", res)

        print("######################################################################################")
        print("######################################################################################")
    return 0


if __name__ == '__main__':
    # METODOS
    gnb = GaussianNB()
    dt = tree.DecisionTreeClassifier()
    msv = svm.SVC()
    knc = neighbors.KNeighborsClassifier()

    metodo = {}
    metodo['Naive Bayes'] = gnb
    metodo['Decision Tree'] = dt
    metodo["maquina de soporte virtual"] = msv
    metodo["k-vecinos"] = knc

    # PORCENTAJE TRAINING
    ratio = [0.6, 0.7, 0.8]
    # repeticiones por porcentaje
    rep = 50

    cadena_inicioT = "title={"
    cadena_inicioK = "keywords={"
    cadena_inicioA = "abstract={"
    cadena_final = "}"

    pattern = sorted(glob.glob('pattern/Pattern_*.txt'))
    system = sorted(glob.glob('system/system_*.txt'))

    titleP = GetText_1linea(pattern, cadena_inicioT)
    titleS = GetText_1linea(system, cadena_inicioT)

    abstractP = GetText_parrafo(pattern, cadena_inicioA, cadena_final)
    abstractS = GetText_parrafo(system, cadena_inicioA, cadena_final)

    KWP = GetText_parrafo(pattern, cadena_inicioK, cadena_final)
    KWS = GetText_parrafo(system, cadena_inicioK, cadena_final)
    print(titleP[1])

    # creo una sola matriz para

    TITLES, labelsTitles = MatrixDocs(titleP, titleS)
    KW, labelsKW = MatrixDocs(KWP, KWS)
    ABSTRACTS, labelsAbstracts = MatrixDocs(abstractP, abstractS)

    print("title before processing:", TITLES[0])

    # paso a lowcase,tokenizo,quito stopwords y lematizo
    TITLES = tratamiento(TITLES)
    ABSTRACTS = tratamiento(ABSTRACTS)
    KW = tratamiento(KW)
    print("title after processing:", TITLES[0])

    ARTICULOS = []
    for i in range(len(labelsTitles)):
        ARTICULOS.append(TITLES[i] + " " + KW[i] + " " + ABSTRACTS[i])

    print("articles:", ARTICULOS[0])
    #Elejir Reduction = True para aplicar PCA
    #si no se aplica PCA, se aplicará SFFS para ARTICULOS y 1 PALABRA tomando 50 atributos

    JournalMultiClasisfier(TITLES, "TITULOS", labelsTitles, ratio, rep, metodo, REDUCTION=False)
    JournalMultiClasisfier(ABSTRACTS, "ABSTRACTS", labelsAbstracts, ratio, rep, metodo, REDUCTION=False)
    JournalMultiClasisfier(KW, "KEYWORDS", labelsKW, ratio, rep, metodo, REDUCTION=False)
    JournalMultiClasisfier(ARTICULOS, "ARTICULOS", labelsKW, ratio, rep, metodo, REDUCTION=False)

