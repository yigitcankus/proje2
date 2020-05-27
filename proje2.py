import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from statsmodels.tools.eval_measures import mse, rmse
from sklearn.linear_model import Ridge
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import linear_model
from scipy.stats import bartlett
from scipy.stats import levene
from statsmodels.tsa.stattools import acf
from scipy.stats import jarque_bera
from scipy.stats import normaltest
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("final_dataa.csv")

print(df.head(10))

df.drop(df.columns[[0, 2, 3, 15, 17, 18]], axis=1, inplace=True)
# deleting unnecessary columns which are "unnamed column", info,zaddress,zipcode, zipid, zestimate (an estime price for the houses from zillow.com)


df['zindexvalue'] = df['zindexvalue'].str.replace(',', '')
df["zindexvalue"]=df["zindexvalue"].astype(np.int64)
# I changed the index value to int. It was obj
df.info()

df.plot(kind="scatter",x="longitude",y="latitude")
plt.title("Evlerin konumuna göre bulundukları yer.")
plt.show()
#I visualized the data with longitude and latitude


corr_matrix = df.corr()
plt.title("Değişkenlerimizin birbiryle korelasyonları")
sns.heatmap(corr_matrix, annot= True,cmap='viridis')
plt.show()
#heatmap of the correlated values

print(corr_matrix["lastsoldprice"].sort_values(ascending=False))
#correlation coefficents for last sold price(target)

#since the most correlated value is finishedsqft I made a new feature called price_per_sqft
df['price_per_sqft'] = df['lastsoldprice']/df['finishedsqft']
corr_matrix = df.corr()
print(corr_matrix["lastsoldprice"].sort_values(ascending=False))
#but this new feature didnt make a big impact

freq = df.groupby('neighborhood').size()
mean = df.groupby('neighborhood').mean()['price_per_sqft']
cluster = pd.concat([freq, mean], axis=1)
cluster['neighborhood'] = cluster.index
cluster.columns = ['freq', 'price_per_sqft','neighborhood']
print(cluster.describe())
#minik bir veri kümesi oluşturduk. Hangi neighborhoodda kaç ev olduğunu ve bunların price per sqft(square footage) fiyatlarını bulduk

cluster1 = cluster[cluster.price_per_sqft < 756]
print("\n",len(cluster1.index)," neighborhoods are low price which are", cluster1.index)

cluster_temp = cluster[cluster.price_per_sqft >= 756]
cluster2 = cluster_temp[cluster_temp.freq <123]
print("\n", len(cluster2.index), "neighborhoods are high price but has low frequency which are ", cluster2.index)

cluster3 = cluster_temp[cluster_temp.freq >=123]
print("\n", len(cluster3.index), "neighborhoods are high price and has high frequency which are ", cluster3.index)


def get_group(x):
    if x in cluster1.index:
        return 'low_price'
    elif x in cluster2.index:
        return 'high_price_low_freq'
    else:
        return 'high_price_high_freq'
df['group'] = df.neighborhood.apply(get_group)
# I categorized the new feature cluster into 3.



n = pd.get_dummies(df.group)
df = pd.concat([df, n], axis=1)
m = pd.get_dummies(df.usecode)
df = pd.concat([df, m], axis=1)
drops = ['group', 'usecode']
df.drop(drops, inplace=True, axis=1)
# I got the dummies of the groups and usecode columns


def is_new(row):
    if row["yearbuilt"] > 2005:
        return 1
    else:
        return 0

df["is_new"] = df.apply(is_new, axis=1)
# I created a new feature called is new. IF house built year > 2005 it is new and showed with 1.

df["rooms+bathroom"] = df["bathrooms"]+ df["totalrooms"]
# A new feature with sumation of rooms and bathrooms

#########################################################################################################
# columns names='address', 'bathrooms', 'bedrooms', 'finishedsqft', 'lastsolddate',
#        'lastsoldprice', 'latitude', 'longitude', 'neighborhood', 'totalrooms',
#        'yearbuilt', 'zindexvalue', 'price_per_sqft', 'high_price_high_freq',
#        'high_price_low_freq', 'low_price', 'Apartment', 'Condominium',
#        'Cooperative', 'Duplex', 'Miscellaneous', 'Mobile', 'MultiFamily2To4',
#        'MultiFamily5Plus', 'SingleFamily', 'Townhouse', 'is_new',
#        'rooms+bathroom'
print(df.columns)
print(df.info())
X = df[["bathrooms","bedrooms","zindexvalue","finishedsqft","yearbuilt","price_per_sqft","is_new","high_price_low_freq","Duplex","MultiFamily2To4"]]
Y = df["lastsoldprice"]

X = sm.add_constant(X)
results = sm.OLS(Y, X).fit()
print(results.summary())
# Adj R-squared is 0.521

# Lineer regression varsayımları:
# Varsayım 2: Hata terimi ortalamada sıfır olmalıdır

Y = df['lastsoldprice']
X = df[["bathrooms","bedrooms","zindexvalue","finishedsqft","totalrooms","yearbuilt","price_per_sqft","is_new","high_price_low_freq","Duplex","MultiFamily2To4"]]
lrm = linear_model.LinearRegression()
lrm.fit(X, Y)

tahmin = lrm.predict(X)
hatalar = Y - tahmin

print("\nModelin ortalama hatası : {:.15f}".format(np.mean(hatalar)))
# evet hata terimi 0'dır.


#Varsayım 3: homoscedasticity
bart_stats = bartlett(tahmin, hatalar)
lev_stats = levene(tahmin, hatalar)

print("Bartlett test değeri : {0:3g} ve p değeri : {1:.21f}".format(bart_stats[0], bart_stats[1]))
print("Levene test değeri   : {0:3g} ve p değeri : {1:.21f}".format(lev_stats[0], lev_stats[1]))
# P değerleri 0.05ten küçük dolayısıyla null hipotezimizi reddeder. Haatlarımız heteroscedastic


# #Varsayım 4: düşük çoklu doğrusallık/low multicollinearity
corr_df = df[["bathrooms","bedrooms","zindexvalue","finishedsqft","totalrooms","yearbuilt","price_per_sqft","is_new","high_price_low_freq","Duplex","MultiFamily2To4"]]
print(corr_df.corr())
plt.figure(figsize=(20,10))
plt.title("Yeni değişkenlerimizin birbiriyle korelasyonları")
sns.heatmap(df.corr(),annot=True, cmap="bwr")
plt.show()
#Değerlerimiz -1 ile 1 arasında olduğu için mükemmel çoklu doğrusallıktalar.

#Varsayım 5: hata terimleri birbiriyle ilişkisiz olmalıdır
plt.figure(figsize=(9,6))
plt.plot(hatalar)
plt.title("hatalar")
plt.show()

acf_data = acf(hatalar)

plt.figure(figsize=(9,6))
plt.title("ACF hatalar")
plt.plot(acf_data[1:])
plt.show()
#Hatalar arasındaki otokorelasyon çok küçüktür.

jb_stats = jarque_bera(hatalar)
norm_stats = normaltest(hatalar)

print("Jarque-Bera test değeri : {0} ve p değeri : {1}".format(jb_stats[0], jb_stats[1]))
print("Normal test değeri      : {0}  ve p değeri : {1:.30f}".format(norm_stats[0], norm_stats[1]))
# Testlerimizin sonucu 0.05ten küçük çıktığı için hatalarımız normal dağılmamıştır.


########
# Modelimizin bir de eğitim kümesindeki performansına bakalım.

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print('Linear Regression R squared": %.4f' % regressor.score(X_test, y_test))
#Önceden 0.52 olan R squared değerimiz eğitim kümesi ile 0.53 oldu


#####
#Yeni bir model oluşturup önceki modelle karşılastıralım.

X = df[["bathrooms","bedrooms","zindexvalue","finishedsqft","yearbuilt"]]
Y = df["lastsoldprice"]

X = sm.add_constant(X)
results = sm.OLS(Y, X).fit()
print(results.summary())

#Ilk modelimizin AIC ve BIC değeri ikinci modelimizden küçük olduğu için onun daha iyi olduğunu söylebiliriz.
#Ikıncı modelimizin F-testi değeri daha yüksek. F testi yönüyle bakarsak modelimizi daha iyi anlatıyor. (???)
#Ilk modelimizin Adj R sqaured değeri daha yüksek.


### Ilk değerimizin değerlendirme metrikleri.
Y = df['lastsoldprice']
X = df[["bathrooms","bedrooms","zindexvalue","finishedsqft","totalrooms","yearbuilt","price_per_sqft","is_new","high_price_low_freq","Duplex","MultiFamily2To4"]]

X = sm.add_constant(X)
results = sm.OLS(Y, X).fit()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
X_test = sm.add_constant(X_test)

y_preds = results.predict(X_test)

plt.figure(dpi = 100)
plt.scatter(y_test, y_preds)
plt.plot(y_test, y_test, color="red")
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin edilen Değerler")
plt.title("Ilk model Gerçek ve tahmin edilen değerler")
plt.show()

print("ILK Model Ortalama Mutlak Hata (MSE)        : {}".format(mean_absolute_error(y_test, y_preds)))
print("ILK Model Ortalama Kare Hata (MSE)          : {}".format(mse(y_test, y_preds)))
print("ILK Model Kök Ortalama Kare Hata (RMSE)     : {}".format(rmse(y_test, y_preds)))
print("ILK Model Ortalama Mutlak Yüzde Hata (MAPE) : {}".format(np.mean(np.abs((y_test - y_preds) / y_test)) * 100))


#Ikıncı Modelimizin değerlendirme metrikleri

X = df[["bathrooms","bedrooms","zindexvalue","finishedsqft","yearbuilt"]]
Y = df["lastsoldprice"]

X = sm.add_constant(X)
results = sm.OLS(Y, X).fit()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
X_test = sm.add_constant(X_test)

y_preds = results.predict(X_test)

plt.figure(dpi = 100)
plt.scatter(y_test, y_preds)
plt.plot(y_test, y_test, color="red")
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin edilen Değerler")
plt.title("Ikinci model Gerçek ve tahmin edilen değerler")
plt.show()

print("İkinci Model Ortalama Mutlak Hata (MSE)        : {}".format(mean_absolute_error(y_test, y_preds)))
print("İkinci Model Ortalama Kare Hata (MSE)          : {}".format(mse(y_test, y_preds)))
print("İkinci Model Kök Ortalama Kare Hata (RMSE)     : {}".format(rmse(y_test, y_preds)))
print("İkinci Model Ortalama Mutlak Yüzde Hata (MAPE) : {}".format(np.mean(np.abs((y_test - y_preds) / y_test)) * 100))

#Ilk modelimizin bütün hata değerlerinde daha küçük çıktığı için daha iyidir.

#######################################################################################################################
# Ilk modelimiz için Ridge regresyonu

Y = df['lastsoldprice']
X = df[["bathrooms","bedrooms","zindexvalue","finishedsqft","totalrooms","yearbuilt","price_per_sqft","is_new","high_price_low_freq","Duplex","MultiFamily2To4"]]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

ridgeregr = Ridge(alpha=10**37)
ridgeregr.fit(X_train, y_train)
y_egitim_tahmini = ridgeregr.predict(X_train)
y_test_tahmini = ridgeregr.predict(X_test)
print("Eğitim kümesi R-Kare değeri       : {}".format(ridgeregr.score(X_train, y_train)))
print("-----Test kümesi istatistikleri---")
print("Test kümesi R-Kare değeri         : {}".format(ridgeregr.score(X_test, y_test)))
print("Ortalama Mutlak Hata (MAE)        : {}".format(mean_absolute_error(y_test, y_test_tahmini)))
print("Ortalama Kare Hata (MSE)          : {}".format(mse(y_test, y_test_tahmini)))
print("Kök Ortalama Kare Hata (RMSE)     : {}".format(rmse(y_test, y_test_tahmini)))
print("Ortalama Mutlak Yüzde Hata (MAPE) : {}".format(np.mean(np.abs((y_test - y_test_tahmini) / y_test)) * 100))

#Ridge regresyonu modelimizi anlatamadı. R-squared değeri negatif oldu ve diğer hatalar da arttı.



















