# import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
df = pd.read_csv('tennis_stats.csv')
print('-------------------')
print(df.describe())
print('-------------------')
print(df.head())
print('-------------------')
print("Number of rows: " + str(len(df)))
print('-------------------')
print("Number of columns: " + str(len(df.columns)))
print('-------------------')
print("Types of columns: " + str(df.dtypes))

# perform exploratory analysis here:

# Distribution of the Fist Serve
# histogram
plt.hist(df['FirstServe'], bins=20)
plt.xlabel('First Serve %')
plt.ylabel('Frequency')
plt.title('Distribution of First Serve %')
plt.show()
# boxplot
plt.boxplot(df['FirstServe'])
plt.xlabel('First Serve %')
plt.title('Distribution of First Serve %')
plt.show()












## perform single feature linear regressions here:






















## perform two feature linear regressions here:






















## perform multiple feature linear regressions here:
