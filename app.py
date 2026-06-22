import os
import sys

# --- BULUT SUNUCUSU İÇİN OTOMATİK KÜTÜPHANE YÜKLEME SİHİRBAZI ---
try:
    import pandas as pd
    import numpy as np
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import r2_score
except ModuleNotFoundError:
    os.system(f'"{sys.executable}" -m pip install pandas scikit-learn numpy')
    import pandas as pd
    import numpy as np
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import r2_score

import streamlit as st

st.set_page_config(page_title="YSA Kayaç Modelleme", layout="wide")

st.title("🌋 Kayaç Özellikleri ve Dalga Yayılımı Tahminleme Aracı")
st.write("Dokuz Eylül Üniversitesi - Maden Mühendisliği Bölümü Bitirme Projesi")

# --- POLİNOMSAL MATEMATİKSEL MODELLERİNİZ ---
def matematiksel_model_veri_uretiyor(yogunluk, porozite):
    p_hizi_yogunluk = (11086 * (yogunluk**2)) - (60952 * yogunluk) + 86489
    s_hizi_yogunluk = (9087.6 * (yogunluk**2)) - (50269 * yogunluk) + 71034
    tebd_yogunluk = (-180.87 * (yogunluk**2)) + (1137.8 * yogunluk) - 1715.9
    
    p_hizi_porozite = (22.538 * (porozite**2)) - (392.19 * porozite) + 4322.8
    s_hizi_porozite = (15.617 * (porozite**2)) - (278.81 * porozite) + 2716.3
    tebd_porozite = (0.8738 * (porozite**2)) - (12.505 * porozite) + 76.576
    dinamik_modul_porozite = (527.47 * (porozite**2)) - (9181 * porozite) + 53622

    p_hizi = (p_hizi_yogunluk + p_hizi_porozite) / 2
    s_hizi = (s_hizi_yogunluk + s_hizi_porozite) / 2
    tebd = (tebd_yogunluk + tebd_porozite) / 2
    dinamik_modul = dinamik_modul_porozite
    
    statik_elastisite = dinamik_modul * 0.15 
    dinamik_poisson = 0.24
    
    return [p_hizi, s_hizi, statik_elastisite, tebd, dinamik_poisson, dinamik_modul]

dosya_adi = "verriler.txt"

if not os.path.exists(dosya_adi):
    st.error(f"Veri dosyası ({dosya_adi}) bulunamadı! Lütfen bu Python koduyla aynı klasörde olduğundan emin olun.")
else:
    df_lab = pd.read_csv(dosya_adi)
    df_lab.columns = [col.strip().replace("'", "").replace('"', '') for col in df_lab.columns]
    
    # Laboratuvar verilerinin sınırlarını net olarak hafızaya alıyoruz
    min_yeg, max_yeg = df_lab['Yogunluk'].min(), df_lab['Yogunluk'].max()
    min_por, max_por = df_lab['Porozite'].min(), df_lab['Porozite'].max()

    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
        st.session_state.scaler_X = None
        st.session_state.scaler_y = None

    islem = st.sidebar.radio("İşlem Adımı Seçin:", ["1. Veri Havuzu ve YSA Eğitimi", "2. Eğitilen Ağ ile Tahmin Yap"])

    if islem == "1. Veri Havuzu ve YSA Eğitimi":
        st.header("🤖 Verileri Birleştirme ve Yapay Sinir Ağını Eğitme")
        st.write("`verriler.txt` dosyasından okunan güncel laboratuvar verileriniz:")
        st.dataframe(df_lab)
        
        sentetik_adet = st.number_input("Formüller kullanılarak kaç adet ek sentetik veri üretilip ağa beslensin?", min_value=0, max_value=500, value=100)
        
        if st.button("Verileri Harmanla ve Machine Learning Başlat"):
            X_list = list(df_lab[['Yogunluk', 'Porozite']].values)
            y_list = list(df_lab[['P_Hizi', 'S_Hizi', 'Statik_Elastisite', 'TEBD', 'Dinamik_Poisson', 'Dinamik_Elastisite']].values)
            
            np.random.seed(42)
            for _ in range(sentetik_adet):
                rastgele_yogunluk = np.random.uniform(min_yeg, max_yeg)
                rastgele_porozite = np.random.uniform(min_por, max_por)
                
                hesaplanan_cikti = matematiksel_model_veri_uretiyor(rastgele_yogunluk, rastgele_porozite)
                X_list.append([rastgele_yogunluk, rastgele_porozite])
                y_list.append(hesaplanan_cikti)
                
            X_all = np.array(X_list)
            y_all = np.array(y_list)
            
            scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
            X_scaled = scaler_X.fit_transform(X_all)
            y_scaled = scaler_y.fit_transform(y_all)
            
            model = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='tanh', solver='lbfgs', max_iter=5000, random_state=42)
            model.fit(X_scaled, y_scaled)
            
            st.session_state.trained_model = model
            st.session_state.scaler_X = scaler_X
            st.session_state.scaler_y = scaler_y
            
            st.success(f"🎉 Başarılı! En yüksek kararlılıkta YSA modeli eğitildi.")
            
            y_pred = scaler_y.inverse_transform(model.predict(X_scaled))
            param_isimleri = ['P Hızı (m/s)', 'S Hızı (m/s)', 'Statik Elastisite (MPa)', 'TEBD (MPa)', 'Dinamik Poisson', 'Dinamik Elastisite (MPa)']
            
            cols = st.columns(6)
            for i, ad in enumerate(param_isimleri):
                r2 = r2_score(y_all[:, i], y_pred[:, i])
                cols[i].metric(label=ad, value=f"R²: {r2:.4f}")

    elif islem == "2. Eğitilen Ağ ile Tahmin Yap":
        st.header("🔮 Girdilere Göre Geri Kalan Tüm Verileri Tahmin Etme")
        
        if st.session_state.trained_model is None:
            st.warning("⚠️ Lütfen önce birinci aşamaya gidip yapay zekayı eğitin!")
        else:
            col_in1, col_in2 = st.columns(2)
            with col_in1:
                g_yeg = st.number_input("Yoğunluk Değeri girin (g/cm³):", value=2.72, format="%.4f")
            with col_in2:
                g_por = st.number_input("Porozite Değeri girin (%):", value=11.50, format="%.2f")
                
            if st.button("Yapay Zekadan Çıktıları Al"):
                # 🔥 AKILLI SINIR KORUMASI: Girdi laboratuvar dışındaysa, değerleri saçmalatmadan en yakın sınıra sabitler
                klip_yeg = np.clip(g_yeg, min_yeg, max_yeg)
                klip_por = np.clip(g_por, min_por, max_por)
                
                if g_yeg != klip_yeg or g_por != klip_por:
                    st.toast(f"ℹ️ Girdiler güvenli laboratuvar aralığına ({min_yeg}-{max_yeg} g/cm³, %{min_por}-%{max_por}) sabitlenerek tahmin yapıldı.")
                
                girdi = np.array([[klip_yeg, klip_por]])
                girdi_s = st.session_state.scaler_X.transform(girdi)
                tahmin_s = st.session_state.trained_model.predict(girdi_s)
                tahmin = st.session_state.scaler_y.inverse_transform(tahmin_s)[0]
                
                out1, out2, out3 = st.columns(3)
                out1.info(f"**P Hızı:** {tahmin[0]:.2f} m/s")
                out2.info(f"**S Hızı:** {tahmin[1]:.2f} m/s")
                out3.info(f"**Statik Elastisite Modülü:** {tahmin[2]:.2f} MPa")
                
                out4, out5, out6 = st.columns(3)
                out4.success(f"**TEBD (UCS):** {tahmin[3]:.2f} MPa")
                out5.success(f"**Dinamik Poisson:** {tahmin[4]:.4f}")
                out6.success(f"**Dinamik Elastisite Modülü:** {tahmin[5]:.2f} MPa")