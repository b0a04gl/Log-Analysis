import classify
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from collections import defaultdict
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz
df=classify.sample5
cat=classify.cat;
category_list=classify.category_list
from sklearn import metrics
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=4, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')#UNI & BI -GRAMS
features = tfidf.fit_transform(df.event).toarray()  #obtaining features
labels = df.category_id
print(features.shape)

print(type(features))
category_to_id=classify.category_to_id
N = 2
#correlating event with features
for category, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  # print(feature_names)
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(category))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

# training
#naive bayer's classifier

X_train, X_test, y_train, y_test = train_test_split(df['event'], df['category'], random_state = 2,test_size=0.3)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_test_dtm = count_vect.transform(X_test)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train) #overfitting

print(X_test.shape)
print(type(X_test))
print(X_test_dtm.shape)
# #testing
# teststring=["Failed to get featured snaps: lookup api.snapcraft.io on server misbehaving","autorefresh.go:376: cannot prepare auto-refresh change: cannot refresh snap-declaration for get https://api.snapcraft.io/api/v1/snaps/assertions/snap-declaration/16/9btclmjz31r0ultmbj5nnge0xm1azfmp?max-format=3: dial tcp: lookup api.snapcraft.io on server misbehaving"]
# teststring.append("Host name conflict, retrying with audist-Veriton-M200-H81-41")
# teststring.append("Window manager warning: Buggy client sent a message with a timestamp of for")
# teststring.append("")
ser=df.event[:30]
print(ser)
X_test_dtm2=count_vect.transform(ser)

print(ser.shape)
print(type(ser))
print(X_test_dtm2.shape)
y_pred_class=clf.predict(X_test_dtm2)
print(y_pred_class)
print(metrics.accuracy_score(y_test, y_pred_class))
####################################
fig = plt.figure(figsize=(5,2))
cat[category_list[3]].groupby('process').event.count().plot.bar(ylim=0)
plt.show()

grouped_category=cat[category_list[3]].groupby('process')
print(grouped_category)
# for row in grouped_category:
#     print(row)

# process_df = cat[category_list[3]][['process']].drop_duplicates()
# # print(process_df.values)
# process_list=process_df.values
newlist=[]
newlist=cat[category_list[3]].process.unique()
print(newlist)
# print(process_list[0][0])
sub_cat={}
if(len(newlist)>=1):
  sub_cat[newlist[0]]=grouped_category.get_group(newlist[0])
if (len(newlist) >= 2):
  sub_cat[newlist[1]]=grouped_category.get_group(newlist[1])
if(len(newlist)>=3):
  sub_cat[newlist[2]]=grouped_category.get_group(newlist[2])
newlist=newlist[:3]
def find_subcat_name(pname):
  text=""
  for row in sub_cat[pname].itertuples():
    print(row.event.lower())
    text += str(row.event.lower())

  re.sub(r'[^\w]', ' ', text)

  for c in string.punctuation:
    text = text.replace(c, "")
                    #removing stop words
  stop_words = set(stopwords.words('english'))

  word_tokens = word_tokenize(text)

  filtered_sentence = [w for w in word_tokens if not w in stop_words]

  filtered_sentence = []

  for w in word_tokens:
    if w not in stop_words:
      filtered_sentence.append(w)

  d = defaultdict(int)
  for w in filtered_sentence:
    d[w] += 1
  count=0
  tempcat={}
  maxMatch=0
  category_list2=[]
  maxCat=""
  for w in sorted(d, key=d.get, reverse=True):
    print(w,d[w])
    print(fuzz.token_set_ratio(pname, w))
    if(fuzz.token_set_ratio(pname, w)>maxMatch) and (w not in category_list2):
      maxMatch=fuzz.token_set_ratio(pname, w)
      maxCat=w
      category_list2.append(maxCat)
  print(maxCat,maxMatch)
  sub_cat[pname] = sub_cat[pname].assign(sub_category=maxCat)
for p in newlist:
  print(p)
  find_subcat_name(p)
  for row in sub_cat[p].itertuples():
    print(row)


for i in range(len(newlist)):
    print(tabulate(sub_cat[newlist[i]],headers="keys",tablefmt="psql"))



