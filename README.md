# veriye_ilk_adim
# 1.Adım:Kütüphane ve Veri Yükleme

  import numpy as np
  
  import pandas as pd
  
  import matplotlib.pyplot as plt

  import seaborn as sns
  
  from sklearn.ensemble import RandomForestClassifier
  
İlgili kütüphaneleri (NumPy, Pandas, Matplotlib, Seaborn ve Scikit-learn) içe aktarıyoruz. Bu kütüphaneler veri manipülasyonu, görselleştirme ve makine öğrenimi için gerekli araçları sağlar.

# 2.Adım:Veriyi Okuma ve İnceleme

  df = pd.read_csv('glass.csv')  - CSV dosyasını oku
  
  df.head()  - Verinin ilk beş satırını göster
  
  df.shape  - Veri setinin boyutunu kontrol et
  
  df['Type'].value_counts()  - 'Type' sütunundaki sınıf değerlerinin sayısını görüntüler

![image](https://github.com/DilekDogan/veriye_ilk_adim/assets/79989171/73a30714-2dab-4469-a973-7f98f9d8daa9)


glass.csv dosyasını Pandas kullanarak veri çerçevesine yüklüyoruz. df.head() ile veri setinin ilk beş satırını inceleyebiliriz. .shape ile veri setinin boyutunu (satır ve sütun sayısı) görüntüleyebiliriz. df['Type'].value_counts() ise 'Type' sütunundaki farklı değerlerin sayısını gösterir, veri setindeki sınıf dağılımını anlamamıza yardımcı olur.

# 3. Adım:Veri Hazırlığı
   
 X = df.drop('Type', axis=1)  - Bağımsız değişkenleri al (Type sütununu çıkar)
 
 y = df['Type']  - Bağımlı değişken (hedef değişken) olarak Type sütununu seç
 
Bağımlı ve bağımsız değişkenleri seçiyoruz. 'Type' sütunu bağımlı değişken olarak belirlenirken, diğer sütunlar bağımsız değişkenleri oluşturuyor.

![image](https://github.com/DilekDogan/veriye_ilk_adim/assets/79989171/03213509-7aaa-4b99-869b-7e0fdd819015)

# 4.Adım:Veriyi Eğitim ve Test Setlerine Ayırma

  from sklearn.model_selection import train_test_split
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

  -Veriyi train ve test setlerine ayır; test seti %30, random_state=9 sabit bir rastgele durum kullan

train_test_split() fonksiyonu, veriyi belirli bir oranda (burada %70 train, %30 test) train ve test setlerine böler. Bu, modelin train verisi üzerinde öğrenmesi ve daha sonra gerçek dünya verileri üzerinde test edilmesi için gereklidir. random_state parametresi aynı rastgele bölmenin her seferinde aynı olmasını sağlar, böylece tekrarlanabilirlik sağlanır.

# 5.Adım:Model Oluşturma ve Eğitme
  model = RandomForestClassifier(random_state=9)  - Random Forest sınıflandırma modelini oluştur
  
  model.fit(X_train, y_train)  - Modeli eğit
  
![image](https://github.com/DilekDogan/veriye_ilk_adim/assets/79989171/5a57dba5-d445-4e9c-85fb-65c7eb15b318)

RandomForestClassifier, rastgele orman algoritmasını kullanarak bir sınıflandırma modeli oluşturur. model.fit() ile bu model, eğitim veri seti (X_train ve y_train) üzerinde eğitilir. Model, veri setindeki desenleri tanıyarak öğrenir.

# 6.Adım:Tahmin Yapma ve Performans Değerlendirmesi

cam_pred = model.predict(X_test)  - Test seti üzerinde tahmin yap

from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

accuracy_score(y_test, cam_pred)  - Doğruluk (accuracy) skoru

  
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test) - Confusion Matrix'i görüntüleme

![image](https://github.com/DilekDogan/veriye_ilk_adim/assets/79989171/92ee52d3-d195-40ce-9f4d-e61bafcdc43f)


print(classification_report(y_test, cam_pred)) - Sınıflandırma raporunu görüntüleme (precision, recall, f1-score gibi metrikler)

![image](https://github.com/DilekDogan/veriye_ilk_adim/assets/79989171/431d6901-896a-4c20-9051-7b08ba1d4571)


Oluşturulan model, test veri seti (X_test) üzerinde model.predict() ile tahmin yapar. Ardından accuracy_score() ile modelin doğruluğunu, ConfusionMatrixDisplay ile karışıklık matrisini ve classification_report() ile sınıflandırma performansını değerlendiririz. Bu adımlar, modelin gerçek veriler üzerindeki performansını anlamamıza yardımcı olur.
