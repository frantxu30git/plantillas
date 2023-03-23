	# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import getopt
import sys
import numpy as np
import pandas as pd
import sklearn as sk
import imblearn
import pickle
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


model=""
p="./"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'p:m:f:h',['path=','model=','testFile=','h'])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    print('OPTIONS   :',options)

    for opt,arg in options:
        if opt in ('-p','--path'):
            p = arg
        elif opt in ('-f', '--file'):
            f = arg
        elif opt in ('-m', '--model'):
            m = arg
        elif opt in ('-h','--help'):
            print(' -p modelAndTestFilePath \n -m modelFileName -f testFileName\n ')
            exit(1)

    
    if p == './':
        model=p+str(m)
        iFile = p+ str(f)
    else:
        model=p+"/"+str(m)
        iFile = p+"/" + str(f)
        

    #Abrir el fichero .csv con las instancias a predecir y que no contienen la clase y cargarlo en un dataframe de pandas para hacer la prediccion
    y_test=pd.DataFrame()
    testX = pd.read_csv(iFile)
    testX = testX.drop('Especie', axis=1)
    #### preproceso ####
    drop_rows_when_missing = []
    impute_when_missing = [{'feature': 'Ancho de sepalo', 'impute_with': 'MEAN'}, {'feature': 'Largo de sepalo', 'impute_with': 'MEAN'}, {'feature': 'Largo de petalo', 'impute_with': 'MEAN'}, {'feature': 'Ancho de petalo', 'impute_with': 'MEAN'}]

    for feature in drop_rows_when_missing:
        testX = testX[testx[feature].notnull()]
        print('Dropped missing records in %s' % feature)

    for feature in impute_when_missing:
	    if feature['impute_with'] == 'MEAN':
	        v = testX[feature['feature']].mean()
	    elif feature['impute_with'] == 'MEDIAN':
	        v = testX[feature['feature']].median()
	    elif feature['impute_with'] == 'CREATE_CATEGORY':
	        v = 'NULL_CATEGORY'
	    elif feature['impute_with'] == 'MODE':
	        v = testX[feature['feature']].value_counts().index[0]
	    elif feature['impute_with'] == 'CONSTANT':
	        v = feature['value']
	    testX[feature['feature']] = testX[feature['feature']].fillna(v)

    rescale_features = {'Ancho de sepalo': 'AVGSTD', 'Largo de sepalo': 'AVGSTD', 'Largo de petalo': 'AVGSTD', 'Ancho de petalo': 'AVGSTD'}

    for (feature_name, rescale_method) in rescale_features.items():
        if rescale_method == 'MINMAX':
            _min = testX[feature_name].min()
            _max = testX[feature_name].max()
            scale = _max - _min
            shift = _min
        else:
            shift = testX[feature_name].mean()
            scale = testX[feature_name].std()
        if scale == 0.:
            del testX[feature_name]
            print('Feature %s was dropped because it has no variance' % feature_name)
        else:
            print('Rescaled %s' % feature_name)
            testX[feature_name] = (testX[feature_name] - shift).astype(np.float64) / scale
    ######
    print('hola')
    print(testX.head(5))
    clf = pickle.load(open(model, 'rb'))
    
    predictions = clf.predict(testX)
    probas = clf.predict_proba(testX)
    y_test['preds'] = predictions
    predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
    results_test = testX.join(predictions, how='left')
    print('\n')
    print(results_test)
    print("FSCORE",f1_score(y_test,predictions,average="macro"))
    results_test.to_csv("fichero.csv")
    
