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
    
    min_yeg, max_yeg = float(df_lab['Yogunluk'].min()), float(df_lab['Yogunluk'].max())
    min_por, max_por = float(df_lab['Porozite'].min()), float(df_lab['Porozite'].max())
    min_tebd, max_tebd = float(df_lab['TEBD'].min()), float(df_lab['TEBD'].max())

    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
        st.session_state.scaler_X = None
        st.session_state.scaler_y = None

    islem = st.sidebar.radio("İşlem Adımı Seçin:", ["1. Veri Havuzu ve YSA Eğitimi", "2. Eğitilen Ağ ile Tahmin Yap"])

    if islem == "1. Veri Havuzu ve YSA Eğitimi":
        st.header("🤖 Verileri Birleştirme (3 Girdi) ve Yapay Sinir Ağını Eğitme")
        st.write("`verriler.txt` dosyasından okunan güncel veri havuzunuz:")
        st.dataframe(df_lab)
        
        sentetik_adet = st.number_input("Formüller kullanılarak kaç adet ek sentetik veri üretilip ağa beslensin?", min_value=0, max_value=500, value=100)
        
        if st.button("Verileri Harmanla ve Machine Learning Başlat"):
            X_list = list(df_lab[['Yogunluk', 'Porozite', 'TEBD']].values)
            # YSA sadece P_Hizi ve S_Hizi parametrelerini tahmin etmek üzere odaklanıyor
            y_list = list(df_lab[['P_Hizi', 'S_Hizi']].values)
            
            np.random.seed(42)
            for _ in range(sentetik_adet):
                rastgele_yogunluk = np.random.uniform(min_yeg, max_yeg)
                rastgele_porozite = np.random.uniform(min_por, max_por)
                hesaplanan_cikti = matematiksel_model_veri_uretiyor(rastgele_yogunluk, rastgele_porozite)
                
                X_list.append([rastgele_yogunluk, rastgele_porozite, hesaplanan_cikti[3]])
                y_list.append([hesaplanan_cikti[0], hesaplanan_cikti[1]])
                
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
            
            st.success(f"🎉 Başarılı! Hız odaklı YSA modeli eğitildi.")
            
            y_pred = scaler_y.inverse_transform(model.predict(X_scaled))
            param_isimleri = ['P Hızı (m/s)', 'S Hızı (m/s)']
            
            cols = st.columns(2)
            for i, ad in enumerate(param_isimleri):
                r2 = r2_score(y_all[:, i], y_pred[:, i])
                cols[i].metric(label=ad, value=f"R²: {r2:.4f}")

    elif islem == "2. Eğitilen Ağ ile Tahmin Yap":
        st.header("🔮 3 Girdiye Göre Kalan Tüm Mekanik Özellikleri Tahmin Etme")
        
        if st.session_state.trained_model is None:
            st.warning("⚠️ Lütfen önce birinci aşamaya gidip yapay zekayı eğitin!")
        else:
            col_in1, col_in2, col_in3 = st.columns(3)
            with col_in1:
                g_yeg = st.number_input("Yoğunluk Değeri girin (g/cm³):", value=2.72, format="%.4f")
            with col_in2:
                g_por = st.number_input("Porozite Değeri girin (%):", value=11.50, format="%.2f")
            with col_in3:
                g_tebd = st.number_input("TEBD / UCS Değeri girin (MPa):", value=45.00, format="%.2f")
                
            if st.button("Yapay Zekadan Çıktıları Al"):
                girdi = np.array([[g_yeg, g_por, g_tebd]])
                girdi_s = st.session_state.scaler_X.transform(girdi)
                tahmin_s = st.session_state.trained_model.predict(girdi_s)
                tahmin = st.session_state.scaler_y.inverse_transform(tahmin_s)[0]
                
                v_p = max(500.0, tahmin[0])
                v_s = max(300.0, tahmin[1])
                rho = g_yeg * 1000.0 # g/cm3 -> kg/m3 dönüşümü
                
                # --- FİZİKSEL FORMÜLLERLE MODÜL HESAPLAMALARI ---
                # Dinamik Poisson Oranı formülü
                if (v_p**2 - 2 * v_s**2) != 0 and (2 * (v_p**2 - v_s**2)) != 0:
                    d_poisson = (v_p**2 - 2 * v_s**2) / (2 * (v_p**2 - v_s**2))
                else:
                    d_poisson = 0.25
                d_poisson = np.clip(d_poisson, 0.05, 0.45)
                
                # Dinamik Elastisite Modülü formülü (Pa cinsinden hesaplayıp MPa'ya çeviriyoruz)
                d_elastisite = (rho * (v_s**2) * (3 * (v_p**2) - 4 * (v_s**2))) / (v_p**2 - v_s**2) / 1000000.0
                if d_elastisite <= 0 or np.isnan(d_elastisite):
                    d_elastisite = (rho * v_p**2 * 1000000.0) / 1000000.0 * 0.1 # Fallback
                
                # Statik Elastisite Modülü bağıntısı
                s_elastisite = d_elastisite * 0.15
                
                out1, out2 = st.columns(2)
                out1.info(f"**P Hızı:** {v_p:.2f} m/s")
                out2.info(f"**S Hızı:** {v_s:.2f} m/s")
                
                out3, out4, out5 = st.columns(3)
                out3.success(f"**Statik Elastisite Modülü:** {s_elastisite:.2f} MPa")
                out4.success(f"**Dinamik Poisson:** {d_poisson:.4f}")
                out5.success(f"**Dinamik Elastisite Modülü:** {d_elastisite:.2f} MPa")