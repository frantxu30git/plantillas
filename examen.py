# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import getopt
import sys
import numpy as np
import pandas as pd
import sklearn as sk
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import os.path
import pickle
from os import path
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
#La variable "k" se inicializa con el valor 1 y la variable "d" con el valor 1.
# """ La variable "p" se establece como el directorio actual "./".
# La variable "f" se establece en "trainHalfHalf.csv" y "oFile" se establece en "output.out".
# El script luego verifica los argumentos de la línea de comandos y las opciones utilizando el módulo "getopt".
# Si se encuentra un error, se muestra un mensaje de error y se sale del programa.
# Si no hay errores, las opciones se asignan a las variables correspondientes.
# Si la opción "-h" se especifica, se imprime una lista de opciones válidas y se sale del programa.
# Luego, se construye la ruta del archivo de entrada (iFile) a partir de los valores de "p" y "f".


p='./'
f="datasetForTheExam_SubGrupo1.csv"
oFile="output.out"
# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'o:d:D:k:K:s:p:f:h:',['output=','k=','K=','d='])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    print('OPTIONS   :',options)

    for opt,arg in options:
        if opt in ('-o','--output'):
            oFile = arg
        elif opt == '-k':
            Kmin = int(arg)
        elif opt == '-K':
            Kmax = int(arg)
        elif opt ==  '-d':
            Dmin = int(arg)
        elif opt ==  '-D':
            Dmax = int(arg)
        elif opt == '-s':
            saltarPreproceso= arg
        elif opt in ('-p', '--path'):
            p = arg
        elif opt in ('-f', '--file'):
            f = arg
        elif opt in ('-h','--help'):
            print('-k Kmin \n- -K K max \n- -d distancemin\n -D distanceMax \n -a algoritmo (KNN o DT) \n -s saltar preprocesado (True o False) \n ')
            exit(1)

    if p == './':
        iFile=p+str(f)
    else:
        iFile = p+"/" + str(f)
    # astype('unicode') does not work as expected
# Finalmente, se define una función para convertir las variables a Unicode, dependiendo de la versión de Python que se esté utilizando. """
    def coerce_to_unicode(x):
        if sys.version_info < (3, 0):
            if isinstance(x, str):
                return unicode(x, 'utf-8')
            else:
                return unicode(x)
        else:
            return str(x)

    #Abrir el fichero .csv y cargarlo en un dataframe de pandas
    ml_dataset = pd.read_csv(iFile)
    #comprobar que los datos se han cargado bien. Cuidado con las cabeceras, la primera línea por defecto la considerara como la que almacena los nombres de los atributos
    # comprobar los parametros por defecto del pd.read_csv en lo referente a las cabeceras si no se quiere lo comentado

    #print(ml_dataset.head(5))
    
    ml_dataset = ml_daml_dataset = ml_dataset[['Perimeter', 'Area', 'kernelLength', 'KernelGrooveLength', 'Compactness', 'KernelWidth', 'AsymmetryCoeff', 'Class']]


    # Explica lo que se hace en este paso

    #se están definiendo dos listas, una para las características categóricas y otra para las características numéricas.
    categorical_features = []
    numerical_features = ['Perimeter', 'Area', 'kernelLength', 'KernelGrooveLength', 'Compactness', 'KernelWidth', 'AsymmetryCoeff']
    

#Inicializa una lista vacía llamada "text_features".
#Convierte cada una de las características categóricas en texto mediante la función "coerce_to_unicode()" 
# y lo guarda en la variable "ml_dataset[feature]".
#Hace lo mismo que en el paso anterior pero con las características de texto.
#Convierte cada una de las características numéricas en double o 
# en "epoch" si la característica es de tipo "datetime". Para esto, se utiliza la función "datetime_to_epoch()" 
# que convierte la fecha y hora en segundos transcurridos desde el 1 de enero de 1970.
    text_features = []
    for feature in categorical_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    for feature in text_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    for feature in numerical_features:
        if ml_dataset[feature].dtype == np.dtype('M8[ns]') or (
                hasattr(ml_dataset[feature].dtype, 'base') and ml_dataset[feature].dtype.base == np.dtype('M8[ns]')):
            ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
        else:
            ml_dataset[feature] = ml_dataset[feature].str.replace(',', '.').astype('double')


    #crea un diccionario target_map que asigna la etiqueta '0' a 0 y la etiqueta '1' a 1.
    target_map = {'1': 0, '2': 1, '3': 2}
    #En la segunda línea, se crea una nueva columna __target__ en el conjunto de datos ml_dataset que contiene 
    # la columna 'TARGET' convertida a una cadena de texto y luego mapeada según el diccionario target_map.
    ml_dataset['__target__'] = ml_dataset['Class'].map(str).map(target_map)    #Se elimina target del dataset
    del ml_dataset['Class']
    # Remove rows for which the target is unknown.
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    print(f)
    print(ml_dataset.head(5))

    #divide el dataset en dos partes: Train y test 
    #El parámetro "random_state=42" asegura que la división sea reproducible, mientras que el
    # parámetro "stratify=ml_dataset[['target']]" garantiza que la distribución de las clases de la variable objetivo (target) sea 
    # la misma en ambos conjuntos
    train, test = train_test_split(ml_dataset,test_size=0.2,random_state=42,stratify=ml_dataset[['__target__']])
    #coge los primeros cinco registros del dataset
    print(train.head(5))
    print(train['__target__'].value_counts())
    print(test['__target__'].value_counts())

    # rescale_features = {'Perimeter': 'AVGSTD', 'Area': 'AVGSTD', 'kernelLength': 'AVGSTD', 'KernelGrooveLength': 'AVGSTD','Compactness': 'AVGSTD','KernelWidth': 'AVGSTD','AsymmetryCoeff': 'AVGSTD'}
    # for (feature_name, rescale_method) in rescale_features.items():
    #     #el método de reescalado es "MINMAX", se calcula el valor mínimo y máximo de la variable correspondiente en el 
    #     # conjunto de entrenamiento. A continuación, se calcula la escala y el desplazamiento necesarios para llevar los 
    #     # valores de la variable en el rango [0, 1]. Si el método de reescalado es "ZSCORE", se utiliza el promedio y la desviación estándar
    #     #  de la variable correspondiente en el conjunto de entrenamiento para llevar los valores de la variable a una distribución normal estándar 
    #     # (con media cero y desviación estándar uno).
    #     if rescale_method == 'MINMAX':
    #         _min = train[feature_name].min()
    #         _max = train[feature_name].max()
    #         scale = _max - _min
    #         shift = _min
    #     else:
    #         shift = train[feature_name].mean()
    #         scale = train[feature_name].std()
    #     #Si la escala es cero, esto significa que la variable no tiene ninguna variabilidad, 
    #     # por lo que no es útil para el modelo y se elimina de los conjuntos de entrenamiento y prueba
    #     if scale == 0.:
    #         del train[feature_name]
    #         del test[feature_name]
    #         print('Feature %s was dropped because it has no variance' % feature_name)
    #     else:
    #     #Se utiliza la función "astype()" para convertir los valores reescalados a números de punto flotante de 64 bits para asegurar la precisión.
    #         print('Rescaled %s' % feature_name)
    #         train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
    #         test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale


    trainX = train.drop('__target__', axis=1)
     #trainY = train['__target__']

    testX = test.drop('__target__', axis=1)
     #testY = test['__target__']

    trainY = np.array(train['__target__'])
    testY = np.array(test['__target__'])

    # classes = pd.unique(trainY)
    # print('Clases únicas en los datos de entrenamiento:', classes)
    # sampling_strategy = {0: 1, 1: int(0.5), 2:int(0.5)}
    # undersample = RandomUnderSampler(sampling_strategy=sampling_strategy)
    # trainXUnder, trainYUnder = undersample.fit_resample(trainX, trainY)
    # testXUnder, testYUnder = undersample.fit_resample(testX, testY)





    # Explica lo que se hace en este paso
    #aplica el algoritmo KNN
    Kmax=Kmax+1
    Dmax=Dmax+1
    W=['uniform','distance']
    mResultado={'k':0,'d':0,'w':'','f-score':0}
    for k in range(int(Kmin),int(Kmax)):
        if(k%2!=0):
            for d in range(int(Dmin),int(Dmax)):
                for w in W:
                    #aplica el algoritmo KNN
                    print("la k es: " + str(k))
                    print("la d es: " + str(d))
                    print('la w es:' + w)
                    clf = KNeighborsClassifier(n_neighbors=int(k),
                                        weights=w,
                                        algorithm='auto',
                                        leaf_size=30,
                                        p=int(d))


                    #Balancea el resultado se asignará un peso mayor a las clases menos representadas en el conjunto de datos.
                    clf.class_weight = "balanced"

                    # Explica lo que se hace en este paso
                    #el clasificador se ajusta (fit) a los datos de entrenamiento (trainX, trainY), 
                    # lo que significa que se ajustará a los patrones en los datos y aprenderá a clasificar nuevos datos.
                    clf.fit(trainX, trainY)


                # Build up our result dataset

                # The model is now trained, we can apply it to our test set:

                    predictions = clf.predict(testX)
                    probas = clf.predict_proba(testX)

                    predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
                    cols = [
                        u'probability_of_value_%s' % label
                        for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
                    ]
                    probabilities = pd.DataFrame(data=probas, index=testX.index, columns=cols)

                # Build scored dataset
                    results_test = testX.join(predictions, how='left')
                    results_test = results_test.join(probabilities, how='left')
                    results_test = results_test.join(test['__target__'], how='left')
                    results_test = results_test.rename(columns= {'__target__': 'TARGET'})

                    i=0
                    for real,pred in zip(testY,predictions):
                        print(real,pred)
                        i+=1
                        if i>5:
                            break

                    print(f1_score(testY, predictions, average=None))
                    print(classification_report(testY,predictions))
                    print(confusion_matrix(testY, predictions, labels=[1,0]))
                    # Save results to CSV
                    report = classification_report(testY, predictions)
                    # macro_precision= precision_score(testY, predictions, average='macro')
                    # macro_recall= recall_score(testY, predictions, average='macro')
                    # f1_score_macro= f1_score(testY, predictions, average='macro')
                    # micro_precision= precision_score(testY, predictions, average='micro')
                    # micro_recall= recall_score(testY, predictions, average='micro')
                    # f1_score_micro= f1_score(testY, predictions, average='micro')
                    cr = classification_report(testY,predictions,output_dict=True)
                    precision = cr['0']['precision']
                    recall= cr['0']['recall']
                    # None_precision= precision_score(testY, predictions, average=None)
                    # None_recall= recall_score(testY, predictions, average=None)
                    f1_score_macro= f1_score(testY, predictions, average='macro')
                    f1_score_micro= f1_score(testY, predictions, average='micro')
                    f1_score_weighted= f1_score(testY, predictions, average='weighted')
                    print('el fscore : '+ str(f1_score_macro))
                    resultado={'k':k,'d':d,'w': w,'f-score':f1_score_macro}
                    if(resultado['f-score']>mResultado['f-score']):
                        mResultado=resultado
                        print('Se ha actualizado el mejor f-score ' + str(mResultado['f-score']))
                    # Check if the file exists

                    if os.path.isfile('resultadosExamenMulticlassSinPreprocesado.csv'):
                        # Append classification report to the file
                        df = pd.read_csv('resultadosExamenMulticlassSinPreprocesado.csv')
                        cr = classification_report(testY,predictions)
                        data = {'k': [k], 'd': [d], 'Weights': [w],'precision None': [precision],'recall None': [recall],'f-score_macro':[f1_score_macro],'f1_score_micro':[f1_score_micro],'f1_score_weighted':[f1_score_weighted]}
                        new_df = pd.DataFrame(data)
                        df = pd.concat([df, new_df], ignore_index=True)
                        df.to_csv('resultadosExamenMulticlassSinPreprocesado.csv', index=False)
                    else:
                    # Create a new file with classification report
                        
                        cm=confusion_matrix(testY, predictions, labels=[1,0])
                        data = {'k': [k], 'd': [d], 'Weights': [w],'precision None': [precision],'recall None': [recall],'f-score_macro':[f1_score_macro],'f1_score_micro':[f1_score_micro],'f1_score_weighted':[f1_score_weighted]}
                        df = pd.DataFrame(data)
                        df.to_csv('resultadosExamenMulticlassSinPreprocesado.csv', index=False)
                    #salvar modelo
    mk=mResultado['k']
    md=mResultado['d']
    mw=mResultado['w']
    print(f'el mejor modelo es  k {mk} y la mejor d es {md} y la mejor w es {mw}')
    clf = KNeighborsClassifier(n_neighbors=mResultado['k'],
                                        weights=mResultado['w'],
                                        algorithm='auto',
                                        leaf_size=30,
                                        p=mResultado['d'])

                    # Explica lo que se hace en este paso


                    #Balancea el resultado se asignará un peso mayor a las clases menos representadas en el conjunto de datos.
    clf.class_weight = "balanced"

    # Explica lo que se hace en este paso
    #el clasificador se ajusta (fit) a los datos de entrenamiento (trainX, trainY), 
    # lo que significa que se ajustará a los patrones en los datos y aprenderá a clasificar nuevos datos.
    clf.fit(trainX, trainY)
    pickle.dump(clf,open ('modeloExamenMulticlass.sav','wb'))
    print('guardado correctamente empleando Pickle')
    print("bukatu da")
