import numpy as np
from os import listdir
from email import message_from_string
from BeautifulSoup import BeautifulSoup as BS
from re import split
import sys 		
from sklearn.linear_model import Perceptron		
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import *
#############################################################################
#Misc INIT
dirList=[1,2,3,4,5,6,7,8,9,10]
dirList.remove(int(sys.argv[1]))
count_vectorizer = CountVectorizer()
tf_transformer = TfidfTransformer(sublinear_tf=True)
classifier = Perceptron()
#############################################################################
All_files=[]
for i in dirList:
	for j in listdir('data/part'+str(i)):
		All_files.append('data/part'+str(i)+'/'+j)

mails=[]
for i in All_files:
	if i[11:14]=='spm' or i[12:15]=='spm':
		mails.append(
			dict(mail=open(i,'r').read().strip(), category='spam')
			)
	else:	
		mails.append(
			dict(mail=open(i,'r').read().strip(), category='nspam')
			)

for n in range(len(mails)):
  html = mails[n]['mail']
  text = ' '.join(BS(html).findAll(text=True))
  mails[n]['text'] = text

train_counts = tf_transformer.fit_transform(count_vectorizer.fit_transform([i['text'] for i in mails]))
train_labels = [(i['category']=='nspam') for i in mails]

##############################################################################
test_files=[]
for i in [int(sys.argv[1])]:
	for j in listdir('data/part'+str(i)):
		test_files.append('data/part'+str(i)+'/'+j)

test_mails=[]
for i in test_files:
	if i[11:14]=='spm' or i[12:15]=='spm':
		test_mails.append(
			dict(mail=open(i,'r').read().strip(), category='spam')
			)
	else:
		test_mails.append(
			dict(mail=open(i,'r').read().strip(), category='nspam')
			)

for n in range(len(test_mails)):
  html = test_mails[n]['mail']
  text = ' '.join(BS(html).findAll(text=True))
  test_mails[n]['text'] = text

test_counts		 = tf_transformer.transform(count_vectorizer.transform([i['text'] for i in test_mails]))
test_labels      = [(i['category']=='nspam') for i in test_mails ]

###############################################################################

classifier.fit(train_counts,train_labels)

################################################################################

c=0
c_s=0
c_ns=0
c_ws=0
c_wns=0

predicted_labels=[]
for i in range(len(test_mails)):
	t_value = classifier.predict(test_counts[i].toarray())
	predicted_labels.append(int(t_value))
	if test_labels[i]==0: 
		c_s=c_s+1
	else:
		c_ns=c_ns+1
	po=0
	if test_labels[i]!=t_value:
		c=c+1
		if t_value==1:
			c_ws=c_ws+1
		else:
			c_wns=c_wns+1

################################################################################


print "*"*60
print "Running on part",sys.argv[1]
print "Total Mails:",len(test_mails)
print "Total Spam:", c_s
print "Total nSpam:", c_ns
print "Total Misclassified:"+str(c)
print "Misclassified Spam:"+str(c_ws)
print "Misclassified nSpam:"+str(c_wns)
print "\nConfusion Matrix"
print confusion_matrix(test_labels,predicted_labels, labels = [0,1])
print "\nClassification Report\n",classification_report(test_labels,predicted_labels,target_names=['spam','nSpam'])
print "*"*60