#####################################################
# AB Testi ile BiddingYöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################

#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi veaveragebidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchasemetriğine odaklanılmalıdır.




#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleriab_testing.xlsxexcel’ininayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna AverageBiddinguygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç



#####################################################
# Proje Görevleri
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direkt 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.





#region Görev 1:  Veriyi Hazırlama ve Analiz Etme

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
import scipy.stats as st
from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import f_oneway
from scipy.stats import kruskal
from statsmodels.stats.proportion import proportions_ztest



# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.

df_control = pd.read_excel("ab_testing.xlsx", sheet_name="Control Group")

df_test = pd.read_excel("ab_testing.xlsx", sheet_name="Test Group")

# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.


df_control.head()
df_test.head()

df_control.describe().T
df_test.describe().T

df_control.shape
df_test.shape

df_control.isnull().sum()
df_test.isnull().sum()



# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.

df = pd.concat([df_control, df_test])
df.shape
df.head()


#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.
#H0 Maximum ve Average Bidding türlerinin satış ortalamalarında fark yoktur.
#H1 ... vardır


# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz
df_control["Purchase"].mean()
df_test["Purchase"].mean()

df_control["Purchase"].mean() / df_test["Purchase"].mean()
(df_control["Purchase"] / df_test["Purchase"]).mean()

#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################


# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.

# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz.

test_stat, pvalue = shapiro(df_control["Purchase"])
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat,
                                           pvalue))

test_stat, pvalue = shapiro(df_test["Purchase"])
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat,
                                           pvalue))

#Normallik varsayımı iki değişken için de sağlanmakta.

#H0: Homojendir.
#H1: Değildir.

test_stat, pvalue = levene(df_control["Purchase"],
                           df_test["Purchase"])
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat,
                                           pvalue))

#Homojenlik iki değişken için de sağlanmakta.


# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz

#Varsayımlar sağlantığı için parametrik test kullanılır.

test_stat, pvalue = ttest_ind(df_control["Purchase"],
                              df_test["Purchase"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#p-value = 0.3493, H0 Reddedilemez.
#m1 ve m2 arasında fark yoktur.

# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.

## Varsayım kontrollerinden aldığımız çıktı doğrultusunda parametrik test uyguladık ve iki değişken arasında istatistiksel olarak anlamlı bir fark olmadığını gördük.




##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.
# Normallik Varsayımı için Shapiro, Varyans Homojenliği varsayımı için Levene, bu varsayımlardan aldığımız dönüt doğrultusunda bağımsız iki örneklem testi (ttest_id) kullandım.



# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.

# Yeni teklif türünün istatistiksel olarak anlamlı bir fark oluşturmadığı, bu sebeple halihazırda kullanımda olan teklif türünün kullanımının daha az maliyetli ve mantıklı olduğu sonucuna vardık.
