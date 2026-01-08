import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


plt.ion() 


print(" Veriseti Yükleniyor...")
try:
    data = pd.read_csv('insurance.csv')
    print("Veri başarıyla yüklendi!")
except FileNotFoundError:
    print("HATA: 'insurance.csv' dosyası bulunamadı! Lütfen aynı klasöre koy.")
    exit()





data['sex'] = data['sex'].map({'female': 0, 'male': 1})
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})


data = pd.get_dummies(data, columns=['region'], drop_first=True)



plt.figure(figsize=(10, 6))
sns.scatterplot(x='bmi', y='charges', hue='smoker', data=data, palette='coolwarm')
plt.title('Vücut Kitle İndeksi (BMI) ve Sigaranın Masrafa Etkisi')
plt.xlabel('BMI')
plt.ylabel('Masraf ($)')

plt.draw()  
plt.pause(1) 

# eğitim
print("\n Model Eğitiliyor...")
X = data.drop('charges', axis=1)
y = data['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# test
print(" Test Sonuçları Hesaplanıyor...")
tahminler = model.predict(X_test)
mae = mean_absolute_error(y_test, tahminler)
r2 = r2_score(y_test, tahminler)
print(f"   -> Model Başarısı (R2): %{r2*100:.1f}")
print(f"   -> Ortalama Sapma: {mae:.2f} $")


print("\n  Müşteri Senaryosu Oluşturuluyor...")


r_yas = np.random.randint(18, 66)
r_cinsiyet = np.random.choice([0, 1]) 
r_bmi = round(np.random.uniform(16, 45), 2)
r_cocuk = np.random.randint(0, 6)
r_sigara = np.random.choice([0, 1], p=[0.7, 0.3]) # %30 ihtimalle sigara içiyor


bolgeler = ['northeast', 'northwest', 'southeast', 'southwest']
secilen_bolge = np.random.choice(bolgeler)


r_nw = 1 if secilen_bolge == 'northwest' else 0
r_se = 1 if secilen_bolge == 'southeast' else 0
r_sw = 1 if secilen_bolge == 'southwest' else 0


yeni_kisi = pd.DataFrame({
    'age': [r_yas],
    'sex': [r_cinsiyet],
    'bmi': [r_bmi],
    'children': [r_cocuk],
    'smoker': [r_sigara],
    'region_northwest': [r_nw],
    'region_southeast': [r_se],
    'region_southwest': [r_sw]
})

#tahmin
fiyat_tahmini = model.predict(yeni_kisi)[0]


print("\n" + "="*50)
print("     S İ G O R T A   T E K L İ F   F İ Ş İ")
print("="*50)
print(f"| {'ÖZELLİK':<20} | {'DEĞER':<23} |")
print("-" * 50)
print(f"| {'Yaş':<20} | {r_yas:<23} |")
print(f"| {'Cinsiyet':<20} | {'Erkek' if r_cinsiyet==1 else 'Kadın':<23} |")
print(f"| {'Vücut Kitle İnd.':<20} | {r_bmi:<23} |")
print(f"| {'Çocuk Sayısı':<20} | {r_cocuk:<23} |")
print(f"| {'Sigara Kullanımı':<20} | {'EVET' if r_sigara==1 else 'Hayır':<23} |")
print(f"| {'Bölge':<20} | {secilen_bolge.capitalize():<23} |")
print("-" * 50)
print(f"| {'TAHMİNİ FİYAT':<20} | {fiyat_tahmini:,.2f} $ {'':<11} |")
print("="*50)

print("\n(Grafik açık. Kapatmak için Enter'a basabilirsin...)")

input()
