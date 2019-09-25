from prepareData import *
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


#Want to display the count of benign and malignant tumors
ClassNames = ['Benign', 'Malignant'] #For visual purposes change, we label Benign and Malignant tumors

df2 = prepareData.dataframe
count = len(df2) #take length of df2
df2['Class'] = pd.cut(df2['Class'], bins=2, labels=ClassNames) #Encode 2 -> Benign, and 4 -> malignant

#plot the graph
classes = df2['Class']
ax = sns.countplot(x=classes, hue='Class', data=df2)
ax.set_title('Frequency of Benign and Malignant Tumors')
ax2 = ax.twinx() #twin axes

ax2.yaxis.tick_left()
ax.yaxis.tick_right()
ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')

ax2.set_ylabel('Frequency [%]') #Label for left side

#Adding the numerical frequency for each box plot
for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/count), (x.mean(), y),
            ha='center', va='bottom') #Set the alignment of the text

#Use a LinearLocator to ensure the correct number of ticks for count
ax.yaxis.set_major_locator(ticker.LinearLocator(10))

#Frequency range from 0 to 100
ax2.set_ylim(0, 100)
ax.set_ylim(0, count)

# And use a MultipleLocator to ensure a tick spacing of 10
ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

print(prepareData.dataframe.head())
plt.show()
