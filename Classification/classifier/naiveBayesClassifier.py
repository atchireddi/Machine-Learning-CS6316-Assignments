from sklearn.naive_bayes import MultinomialNB
import numpy as np

class naiveBayesClassifier:

	def __init__(self):
		self.vocabulary = {'love','wonderful','best', 'great', 'superb', 'still', 'beautiful', 'bad', 'worst',
'stupid', 'waste', 'boring', '?', '!','loving','loved','loves'}

	def naiveBayesMulFeature_train(self, Xtrain, ytrain):
	    cpos=np.zeros(len(self.vocabulary)) # calculate the count for each word in each group
	    cneg=np.zeros(len(self.vocabulary))
	    for i in np.arange(len(ytrain)):
	        if ytrain[i]=='pos':
	            cpos=cpos+np.array(Xtrain[i])
	        else:
	            cneg=cneg+np.array(Xtrain[i])

	    a=1.0                                         # Laplace smoothing parameter
	    thetaPos = (cpos+a)*1.0/(sum(cpos)+a*len(cpos))  # Laplace smoothing
	    thetaNeg = (cneg+a)*1.0/(sum(cneg)+a*len(cneg))
	    return thetaPos, thetaNeg


	def naiveBayesMulFeature_test(self, Xtest, ytest,thetaPos, thetaNeg):
	    yPredict = []
	    ntest = len(ytest)
	    for i in np.arange(ntest):
	        p_pos = np.log(0.5)+np.inner(np.log(thetaPos),np.array(Xtest[i]))
	        p_neg = np.log(0.5)+np.inner(np.log(thetaNeg),np.array(Xtest[i]))
	        if p_pos > p_neg:
	            pred = "pos"
	        else:
	            pred = "neg"
	        yPredict.append(pred)
	    Accuracy = sum(np.array(yPredict)==np.array(ytest))*1.0/ntest
	    return yPredict, Accuracy

	def run(self, Xtrain, ytrain, Xtest, ytest):
		thetaPos, thetaNeg = self.naiveBayesMulFeature_train(Xtrain, ytrain)
		yPredict, Accuracy = self.naiveBayesMulFeature_test(Xtest, ytest,thetaPos, thetaNeg)
		print ("The accurary of naiveBayes classifier is %f") %Accuracy
