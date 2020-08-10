import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import glob
import os
from io import StringIO
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
from sklearn.feature_selection import chi2
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
class log_analysis:


    def __init__(self):
        print("Hey,You!")

                                                                    ###---parsing---###

    def parse(self,file):
        f = open(file, encoding="utf8")
        lines = f.readlines()
        records = {}
        t = []
        p = []
        e = []
        c = []
        cid = []
        subcat = []
        for line in lines:
            # print(line)
            timestamp = ""
            process = ""
            event = ""
            for w in line.split()[0:3]:
                timestamp += str(w) + " "
            t.append(timestamp)
            for w in line.split()[3:5]:
                if (w[0].isalpha() == True):
                    process += str(w) + " "
            p.append(process)
            for w in line.split()[5:]:
                if (w[0].isalpha() == True):
                    event += str(w) + " "
            e.append(event)
            c.append(None)
            cid.append(None)
            subcat.append(None)
        records = {"timestamp": t, "process": p, "event": e, "category": c, "category_id": cid, "sub_category": subcat}
        temp_df = pd.DataFrame(records)
        # vocubalating unique events in log records
        sample5 = temp_df[
            ['timestamp', 'process', 'event', 'category', 'category_id', 'sub_category']].drop_duplicates().sort_values(
            'timestamp')  # removing duplicates
        return sample5

                                                       ###---Initial classification---###

    def init_classification(self,sample5):
        i = 0
        indexlist = list(sample5.index.values)
        resultindex = []
        vocab = ['host name conflict', 'error', 'warning', 'server misbehaving', 'device state change',
                 'gdbus error']  # general anomaly keywords
        # filtering records of anomaly
        for row in sample5.itertuples():
            words = row.event.lower()
            # print(words)
            index = indexlist[i]
            for v in vocab:
                if (words.find(v) != -1):
                    sample5.at[index, 'category'] = 'anomaly'
                    break
            i += 1

        sample5 = sample5[sample5.category.notnull()]
        return sample5


                                                    ###---removing stop words---###


    def remove_stop_words(self,sample5):
        text = ""
        for row in sample5.itertuples():
            text += str(row.event.lower())

        re.sub(r'[^\w]', ' ', text)
        re.sub(r'[\d]+ ', ' ', text)
        for c in string.punctuation:
            text = text.replace(c, "")
        stop_words = set(stopwords.words('english'))

        word_tokens = word_tokenize(text)

        filtered_sentence = [w for w in word_tokens if not w in stop_words]

        filtered_sentence = []

        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w)
        return filtered_sentence

    def hasNumbers(self,inputString):
        return any(char.isdigit() for char in inputString)

    def find_subcat_name(self,sub_cat,pname,category_list2):
            filtered_sentence=self.remove_stop_words(sub_cat[pname])
            d = defaultdict(int)
            for w in filtered_sentence:
                d[w] += 1
            count = 0
            tempcat = {}
            maxMatch = 0
            maxCat = ""
            for w in sorted(d, key=d.get, reverse=True):

                print(w, d[w])
                print(fuzz.token_set_ratio(pname, w))
                if (fuzz.token_set_ratio(pname, w) > maxMatch) and (w not in category_list2) and (self.hasNumbers(w)==False):
                    maxMatch = fuzz.token_set_ratio(pname, w)
                    maxCat = w
                    category_list2.append(maxCat)
            print(maxCat, maxMatch)
            sub_cat[pname] = sub_cat[pname].assign(sub_category=maxCat)
            return category_list2

                                                    ###--categorizing---###

    def categorize(self,filtered_sentence,sample5,cat,category_list,i,level):
        vocab = ['host name conflict', 'error', 'warning', 'server misbehaving', 'device state change',
                 'gdbus error']  # general anomaly keywords

        if (level==1):
            category_list = []
            count = 0
            d = defaultdict(int)
            for w in filtered_sentence:
                d[w] += 1
            for w in sorted(d, key=d.get, reverse=True):
                # print(w, d[w])
                for v in vocab:
                    if (v.find(w) != -1) and (v not in category_list):
                        category_list.append(v)
                        count += 1
                        break
                if (count == 5):
                    break
            category_list.sort()
            # print(category_list)

            j = 0
            indexlist = list(sample5.index.values)

            for row in sample5.itertuples():
                # print(type(row.event))
                word = row.event.lower()

                index = indexlist[j]
                if (word.find(category_list[0]) != -1):
                    sample5.at[index, 'category'] = category_list[0]
                if (word.find(category_list[1]) != -1):
                    sample5.at[index, 'category'] = category_list[1]
                if (word.find(category_list[2]) != -1):
                    sample5.at[index, 'category'] = category_list[2]
                if (word.find(category_list[3]) != -1):
                    sample5.at[index, 'category'] = category_list[3]
                if (word.find(category_list[4]) != -1):
                    sample5.at[index, 'category'] = category_list[4]
                j += 1
            sample5 = sample5.drop_duplicates(subset="event").sort_values('timestamp')  # removing duplicates

            sample5['category_id'] = sample5['category'].factorize()[0]
            return sample5,category_list;
        if (level == 2):
            fig = plt.figure(figsize=(5, 2))
            cat[category_list[i]].groupby('process').event.count().plot.bar(ylim=0)
            plt.show()
            grouped_category = cat[category_list[i]].groupby('process')
            print(grouped_category.first())
            newlist = []
            newlist = cat[category_list[i]].process.unique()
            print(newlist)
            sub_cat = {}
            if (len(newlist) >= 1):
                sub_cat[newlist[0]] = grouped_category.get_group(newlist[0])
            if (len(newlist) >= 2):
                sub_cat[newlist[1]] = grouped_category.get_group(newlist[1])
            if (len(newlist) >= 3):
                sub_cat[newlist[2]] = grouped_category.get_group(newlist[2])
            newlist = newlist[:3]
            category_list2=[]
            for p in newlist:
                print(p)
                category_list2=self.find_subcat_name(sub_cat,p,category_list2)
            for i in range(len(newlist)):
                print(tabulate(sub_cat[newlist[i]], headers="keys", tablefmt="psql"))
            return None

                                                        ###---plotting categories---###

    def plot_category(self,sample5,category_list):

        fig = plt.figure(figsize=(5, 2))
        sample5.groupby('category').event.count().plot.bar(ylim=0)
        plt.show()
        grouped_category = sample5.groupby('category')
        cat = {}

        cat[category_list[0]] = grouped_category.get_group(category_list[0])
        cat[category_list[1]] = grouped_category.get_group(category_list[1])
        cat[category_list[2]] = grouped_category.get_group(category_list[2])
        cat[category_list[3]] = grouped_category.get_group(category_list[3])
        cat[category_list[4]] = grouped_category.get_group(category_list[4])
        for i in range(len(category_list)):
            self.printDf(cat[category_list[i]])
        return cat

                                                        ###---Training---###

    def train(self,df, category_to_id ):
        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=4, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                                stop_words='english')  # UNI & BI -GRAMS
        features = tfidf.fit_transform(df.event).toarray()  # obtaining features
        labels = df.category_id
        print(features.shape)

        print(type(features))

        N = 2
        # correlating event with features
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
        # naive bayer's classifier
        X_train, X_test, y_train, y_test = train_test_split(df['event'], df['category'], random_state=1)
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer()
        X_test_dtm = count_vect.transform(X_test)
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        clf = MultinomialNB().fit(X_train_tfidf, y_train)  # overfitting
        y_pred_class=clf.predict(X_test_dtm)
        print("\t\t\tTEST RESULTS FOR SPLITTED TEST DATASET\t\t\t")
        print("Prediction result\t:\t{}".format(y_pred_class))
        y_pred_prob = clf.predict_proba(X_test_dtm)[:, 1]
        print("\t\t\tprobablities of prediction\t\t\t")
        print(y_pred_prob)
        print("Accuracy score\t:\t{}".format(metrics.accuracy_score(y_test,y_pred_class)))
        print("\t\t\tConfusion matrix\t\t\t")
        print(metrics.confusion_matrix(y_test, y_pred_class))
        return clf,count_vect,y_test

                                                    ###---Testing---###

    def test(self,clf,count_vect,y_test,teststring):
        y_pred_class=clf.predict(count_vect.transform(teststring))
        print(y_pred_class)

        ###---printing Dataframe---###

    def printDf(self,df):
        print("\t\tDataFrame size \t:\t{}".format(df.size))
        print("\t\tDataFrame shape \t:\t{}".format(df.shape))
        print(tabulate(df,headers='keys',tablefmt='psql'))



l1=log_analysis()

sample5=l1.parse("D:\Log Analysis\Dataset\log7.txt")

sample5=l1.init_classification(sample5)

filtered_sentence=l1.remove_stop_words(sample5)

sample5,category_list=l1.categorize(filtered_sentence,sample5,{},[],0,1)

category_id_df = sample5[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)

# l1.printDf(sample5)

cat=l1.plot_category(sample5,category_list)

for i in range(len(category_list)):
    l1.categorize(filtered_sentence,sample5,cat,category_list,i,2)

clf,count_vect,y_test=l1.train(sample5,category_to_id)

l1.test(clf,count_vect,y_test,["Failed to get featured snaps: lookup api.snapcraft.io on server misbehaving"])
