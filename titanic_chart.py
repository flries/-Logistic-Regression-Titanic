''' 1. Importing Libraries and Packages '''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib as mpl

train_data = pd.read_csv("input/train.csv")
test_data = pd.read_csv("input/test.csv")

# Survived vs. Pclass
df_s = train_data[train_data.Survived==1].groupby('Pclass').size()
df_d = train_data[train_data.Survived==0].groupby('Pclass').size()

x_idx = [1,2,3]

fig = plt.figure(figsize=(6,6))

b1 = plt.bar(x_idx, df_d, width=0.5, color='#CD5C5C')
b2 = plt.bar(x_idx, df_s, width=0.5, bottom=df_d, color='#98BF64')

plt.title('Survivors vs. Pclass')
plt.xticks(x_idx, ('1st Class', '2nd Class', '3rd Class'))
plt.xlabel('Pclass')
plt.legend((b1[0], b2[1]), ('Not survived', 'Survived'))
#plt.show()


# Survived vs. Gender
df = train_data[['Survived','Sex']].groupby(['Sex', 'Survived']).Sex.count().unstack()

fig, ax = plt.subplots(figsize=(6,6))
cmap = mpl.colors.ListedColormap(['#CD5C5C', '#98BF64'])

df.plot(kind='bar', stacked=True, title='Survived vs. Gender', ax=ax, colormap=cmap, width=0.5)
#plt.show()

# Survived vs Embarked vs Pclass
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

emb_surv = train_data[['Survived','Embarked']]
emb_surv = emb_surv.groupby(['Embarked','Survived']).Embarked.count().unstack()
emb_surv.plot(kind='bar', stacked=True, ax=ax[0], colormap=cmap, width=0.5, title='Survived vs Embarked')

cmap = mpl.colors.ListedColormap(['#FFDEAD', '#4682B4', '#778899'])
emb_pclass = train_data[['Pclass','Embarked']]
emb_pclass = emb_pclass.groupby(['Embarked','Pclass']).Pclass.count().unstack()
emb_pclass.plot(kind='bar', stacked=True, ax=ax[1], colormap=cmap , title='Embarked vs Pclass')
plt.show()

# Histogram of Age
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=(12,6))
fig.subplots_adjust(wspace=.05)

df_survived = train_data[(train_data.Survived==1) & (train_data.Age > 0)].Age;
df_deads = train_data[(train_data.Survived==0) & (train_data.Age > 0)].Age;

ax[0].hist(df_survived, bins=10, color = 'gray', range=(0,100))
ax[0].set_title('Histogram of Age (survived)')
ax[1].hist(df_deads, bins=10, color = 'gray', range=(0,100))
ax[1].set_title('Histogram of Age (not survived)')

plt.show()
