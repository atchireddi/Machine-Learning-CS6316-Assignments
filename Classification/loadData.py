import nltk
from nltk.stem import snowball
from nltk import FreqDist
import os

vocabulary = {'love','wonderful','best', 'great', 'superb', 'still', 'beautiful', 'bad', 'worst',
'stupid', 'waste', 'boring', '?', '!','loving','loved','loves'}
# SBStemmer = snowball.SnowballStemmer("english", ignore_stopwords=True)
# vocabulary=[v for v in vocabulary]
# vocabulary.sort()

def transfer(fileDj,vocabulary):
    fo=open(fileDj,"r")
    content=fo.read()
    tokens=nltk.word_tokenize(content)
    # st=[SBStemmer.stem(t) for t in tokens]
    st=tokens
    fo.close()

    fdist=FreqDist(st)
    BOWDj = []
    for key in vocabulary:
        if key in fdist.keys():
            BOWDj.append(fdist.get(key))
        else:
            BOWDj.append(0)
    return BOWDj



def loadData(Path):
    Xtrain=[]
    Xtest=[]
    ytrain=[]
    ytest=[]
    def load_files_in_a_folder(folder):
        if not folder[-1]=="/":
            folder=folder+"/"
        fs = os.listdir(folder)    # list all the files and sub folders under the Path
        for f in fs:
            if not f.startswith("."):  # ignore files or folders start with period
                if not os.path.isdir(folder+f):  # if this is not a folder, that is, this is a file
                    x=transfer(folder+f,vocabulary=vocabulary)
                    y=os.path.dirname(folder+f).split("/")[-1]
                    test_or_train=os.path.dirname(folder+f).split("/")[-2]  # to see this is test or train, hard code
                    if test_or_train=="training_set":
                        Xtrain.append(x)
                        ytrain.append(y)
                    else:
                        Xtest.append(x)
                        ytest.append(y)
                else:
                    subfolder=folder+f+"/"
                    load_files_in_a_folder(subfolder)  # else this is a folder

    load_files_in_a_folder(Path)
    return Xtrain, ytrain, Xtest, ytest