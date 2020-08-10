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
import parser

file="D:\Log Analysis\Dataset\log5.txt"

f=open(file,encoding="utf8")
lines=f.readlines()
records={}
t=[]
p=[]
e=[]
c=[]
cid=[]
subcat=[]
for line in lines:
        #print(line)
        timestamp=""
        process=""
        event=""
        for w in line.split()[0:3]:
             timestamp+=str(w)+" "
        t.append(timestamp)
        for w in line.split()[3:5]:
            if (w[0].isalpha() == True):
               process+=str(w)+" "
        p.append(process)
        for w in line.split()[5:]:
            if (w[0].isalpha() == True):
                event+=str(w)+" "
        e.append(event)
        c.append(None)
        cid.append(None)
        subcat.append(None)
records={"timestamp":t,"process":p,"event":e,"category":c,"category_id":cid,"sub_category":subcat}
temp_df=pd.DataFrame(records)
                                                     # vocubalating unique events in log records
sample5 = temp_df[['timestamp','process' ,'event','category','category_id','sub_category']].drop_duplicates().sort_values('timestamp')  # removing duplicates

                              ##########################################
                                #categorization
                            ##########################################
i=0
indexlist=list(sample5.index.values)
resultindex=[]
vocab = ['host name conflict','error','warning','server misbehaving','device state change','gdbus error']  #general anomaly keywords

                    #filtering records of anomaly
for row in sample5.itertuples():
    words=row.event.lower()
    # print(words)
    index = indexlist[i]
    for v in vocab:
        if(words.find(v)!=-1):
            sample5.at[index, 'category'] = 'anomaly'
            break
    i+=1

sample5= sample5[sample5.category.notnull()]
# print(sample5)

############################################
text=""
for row in sample5.itertuples():
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

# print(word_tokens)
# print(filtered_sentence)

                #finding frequency
category_list=[]
count=0
d = defaultdict(int)
for w in filtered_sentence:
  d[w] += 1
for w in sorted(d, key=d.get, reverse=True):
 # print(w, d[w])
 for v in vocab:
     if (v.find(w) != -1) and (v not in category_list):

         category_list.append(v)
         count+=1
         break
 if (count==5):
     break
category_list.sort()
# print(category_list)

j=0
indexlist=list(sample5.index.values)

for row in sample5.itertuples():
   # print(type(row.event))
   word=row.event.lower()

   index=indexlist[j]
   if (word.find(category_list[0])!=-1):
       sample5.at[index, 'category'] = category_list[0]
   if (word.find(category_list[1]) != -1):
       sample5.at[index, 'category'] = category_list[1]
   if (word.find(category_list[2]) != -1):
       sample5.at[index, 'category'] = category_list[2]
   if (word.find(category_list[3]) != -1):
       sample5.at[index, 'category'] = category_list[3]
   if (word.find(category_list[4]) != -1):
       sample5.at[index, 'category'] = category_list[4]
   j+=1
sample5 = sample5.drop_duplicates(subset="event").sort_values('timestamp')  # removing duplicates

sample5['category_id'] = sample5['category'].factorize()[0]
category_id_df = sample5[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
# print(sample5)
# print(category_to_id)
# for row in sample5.itertuples():
#     print(row)


#   # plotting graph
fig = plt.figure(figsize=(5,2))
sample5.groupby('category').event.count().plot.bar(ylim=0)
# plt.show()
grouped_category=sample5.groupby('category')
cat={}
# for row in grouped_category:
#     print(row)
cat[category_list[0]]=grouped_category.get_group(category_list[0])
cat[category_list[1]]=grouped_category.get_group(category_list[1])
cat[category_list[2]]=grouped_category.get_group(category_list[2])
cat[category_list[3]]=grouped_category.get_group(category_list[3])
cat[category_list[4]]=grouped_category.get_group(category_list[4])
# print(tabulate(cat[category_list[0]],headers="keys",tablefmt="psql"))
# print(tabulate(cat[category_list[1]],headers="keys",tablefmt="psql"))
# print(tabulate(cat[category_list[2]],headers="keys",tablefmt="psql"))
# print(tabulate(cat[category_list[3]],headers="keys",tablefmt="psql"))
# print(tabulate(cat[category_list[4]],headers="keys",tablefmt="psql"))

####################################################################################################################################################################################j=0
month={}
month["Jan"]=1
month["Feb"]=2
month["Mar"]=3
month["Apr"]=4
month["May"]=5
month["Jun"]=6
month["Jul"]=7
month["Aug"]=8
month["Sep"]=9
month["Oct"]=10
month["Nov"]=11
month["Dec"]=12


from datetime import datetime
from datetime import timedelta
k=1
test=sample5
indexlist = list(test.index.values)
print(test.shape)
test=test.assign(timeperiod=None)
orgTime = test.at[indexlist[0], 'timestamp']
orgTime = orgTime.split()
time = orgTime[2].split(':')
orgTime = orgTime[0:2]
count=1
for i in time:
    orgTime.append(i)
date_obj1 = datetime(year=2019, month=month[orgTime[0]], day=int(orgTime[1]), hour=int(orgTime[2]),
                    minute=int(orgTime[3]), second=int(orgTime[3]))
newdiff=0
for row in test.itertuples():
    if(k<len(indexlist)):
        index=indexlist[k]
        orgTime = test.at[index, 'timestamp']
        orgTime = orgTime.split()
        time = orgTime[2].split(':')
        orgTime = orgTime[0:2]
        for i in time:
            orgTime.append(i)
        date_obj2=datetime(year=2019,month=month[orgTime[0]],day=int(orgTime[1]),hour=int(orgTime[2]),minute=int(orgTime[3]),second=int(orgTime[4]))

        diff=date_obj2-date_obj1
        day_diff=diff.days
        hour_diff=int(diff/ timedelta(hours=1))

        print(hour_diff)
        if(newdiff<hour_diff):
            newdiff=hour_diff
            count+=1
        test.at[index, 'timeperiod'] = count



    k+=1
# #   # plotting graph
fig = plt.figure(figsize=(5,2))
test.groupby('timeperiod').event.count().plot.bar(ylim=0)
plt.show()
grouped_category=test.groupby('timeperiod')
timeunit={}
print(grouped_category.first())
timeunit[0]=grouped_category.get_group(1)
timeunit[1]=grouped_category.get_group(2)
timeunit[2]=grouped_category.get_group(3)
timeunit[3]=grouped_category.get_group(4)
timeunit[4]=grouped_category.get_group(5)
timeunit[5]=grouped_category.get_group(6)

fig = plt.figure(figsize=(5,2))
timeunit[0].groupby('category').event.count().plot.bar(ylim=0)

# plt.show()

fig = plt.figure(figsize=(5,2))
timeunit[1].groupby('category').event.count().plot.bar(ylim=0)
# plt.show()

fig = plt.figure(figsize=(5,2))
timeunit[2].groupby('category').event.count().plot.bar(ylim=0)
# plt.show()

fig = plt.figure(figsize=(5,2))
timeunit[3].groupby('category').event.count().plot.bar(ylim=0)
# plt.show()

fig = plt.figure(figsize=(5,2))
timeunit[4].groupby('category').event.count().plot.bar(ylim=0)
# plt.show()

fig = plt.figure(figsize=(5,2))
timeunit[5].groupby('category').event.count().plot.bar(ylim=0)
# plt.show()
count=[]
devstatec=errorc=hostc=sermisc=warnc=0
for i in range(6):
    count.append(timeunit[i].shape[0])
    # print(count)





print(max(count))


