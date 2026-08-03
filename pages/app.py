import streamlit as st
import pandas as pd
import io
import numpy as np
import joblib
import json
import os
import re
import plotly.express as px
import plotly.graph_objects as go
import shap
import warnings
warnings.filterwarnings('ignore')

# ── KONFIGURASI HALAMAN ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi LoS RSU Aulia",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.sidebar .sidebar-content {
    background-color: var(--secondary-background-color);
}
.page-header {
    background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
    color: white;
    padding: 24px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}
.page-header h2 { margin: 0; font-size: 1.75rem; font-weight: 700; letter-spacing: -0.025em; }
.page-header p  { margin: 8px 0 0; font-size: 1rem; opacity: 0.9; }
.card {
    background: var(--background-color);
    border: 1px solid var(--secondary-background-color);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
}
.card-title {
    font-size: 0.875rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-color);
    opacity: 0.8;
    margin-bottom: 12px;
}
.badge-pendek  { background:rgba(34, 197, 94, 0.1); color:#22c55e; border:1px solid rgba(34, 197, 94, 0.3); border-radius:12px; padding:16px 24px; font-weight:700; font-size:1.25rem; display:inline-block;}
.badge-sedang  { background:rgba(245, 158, 11, 0.1); color:#f59e0b; border:1px solid rgba(245, 158, 11, 0.3); border-radius:12px; padding:16px 24px; font-weight:700; font-size:1.25rem; display:inline-block;}
.badge-panjang { background:rgba(239, 68, 68, 0.1); color:#ef4444; border:1px solid rgba(239, 68, 68, 0.3); border-radius:12px; padding:16px 24px; font-weight:700; font-size:1.25rem; display:inline-block;}
.metric-box {
    background: var(--secondary-background-color);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    border: 1px solid var(--secondary-background-color);
    border-top: 4px solid #3b82f6;
}
.metric-box .metric-val { font-size: 2rem; font-weight: 700; color: var(--text-color); }
.metric-box .metric-lbl { font-size: 0.875rem; color: var(--text-color); opacity: 0.8; margin-top: 4px; font-weight: 500;}
.metric-box.alert { border-top-color: #ef4444; background: rgba(239, 68, 68, 0.05); border-color: rgba(239, 68, 68, 0.2);}
.metric-box.alert .metric-val { color: #ef4444; }
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 12px 24px;
    transition: all 0.2s;
}
.stButton > button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ── LOAD ASSETS ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_assets():
    model    = joblib.load(os.path.join(BASE, 'dashboard', 'model.pkl'))
    encoders = joblib.load(os.path.join(BASE, 'dashboard', 'encoders.pkl'))
    fitur    = joblib.load(os.path.join(BASE, 'dashboard', 'fitur.pkl'))
    nama     = joblib.load(os.path.join(BASE, 'dashboard', 'nama_model.pkl'))
    return model, encoders, fitur, nama

@st.cache_resource
def load_shap_explainer(_model):
    return shap.TreeExplainer(_model)

model, encoders, FITUR, nama_model = load_assets()
shap_explainer = load_shap_explainer(model)

LABEL       = {0: 'Pendek (≤ 2 days)', 1: 'Sedang (3-5 days)', 2: 'Panjang (> 5 days)'}
LABEL_SHORT = ['Pendek', 'Sedang', 'Panjang']
BADGE_CSS   = {0: 'badge-pendek', 1: 'badge-sedang', 2: 'badge-panjang'}
WARNA_BAR   = ['#22c55e', '#f59e0b', '#ef4444']

def parse_usia(usia_str):
    # Menangani token 'Hr' (hari) untuk usia neonatal -- konsisten dengan
    # 1_cleaning.py, supaya prediksi (individu maupun massal) sinkron dengan
    # format yang dipelajari model saat training.
    if pd.isna(usia_str): return 0.0
    s = str(usia_str).strip()
    tahun = re.search(r'(\d+)\s*Th', s, re.IGNORECASE)
    bulan = re.search(r'(\d+)\s*Bl', s, re.IGNORECASE)
    hari  = re.search(r'(\d+)\s*Hr', s, re.IGNORECASE)
    t = int(tahun.group(1)) if tahun else 0
    b = int(bulan.group(1)) / 12 if bulan else 0
    h = int(hari.group(1)) / 365 if hari else 0
    if not tahun and not bulan and not hari:
        try: return float(s)
        except: return 0.0
    return round(t + b + h, 4)

def encode_input(data: dict) -> pd.DataFrame:
    row = {}
    for col in FITUR:
        val = data.get(col)
        if col == 'usia':
            row[col] = parse_usia(val) if isinstance(val, str) else float(val or 0)
        elif col in encoders:
            val_str = str(val)
            if val_str in set(encoders[col].classes_):
                row[col] = encoders[col].transform([val_str])[0]
            else:
                row[col] = -1
        else:
            row[col] = val
    return pd.DataFrame([row])

def encode_dataframe(df_input: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(index=df_input.index)
    mask_unknown_total = pd.Series(False, index=df_input.index)
    for col in FITUR:
        if col == 'usia':
            X[col] = df_input[col].apply(lambda v: parse_usia(v) if isinstance(v, str) else float(v or 0))
        elif col in encoders:
            val_str = df_input[col].astype(str)
            known = set(encoders[col].classes_)
            mask_unknown = ~val_str.isin(known)
            mask_unknown_total = mask_unknown_total | mask_unknown
            encoded = np.full(len(val_str), -1, dtype=int)
            if (~mask_unknown).any():
                encoded[~mask_unknown] = encoders[col].transform(val_str[~mask_unknown])
            X[col] = encoded
        else:
            X[col] = df_input[col]
    return X, mask_unknown_total

# ── SIDEBAR NAVIGATION ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Sistem Prediksi LoS")
    st.markdown(f"**Model Aktif:** `{nama_model}`")
    st.markdown("---")
    menu = st.radio("Navigasi", ["Prediksi Individu", "Prediksi Massal", "Kapasitas & Tren", "Perbandingan Model"])
    st.markdown("---")
    if st.button("🔄 Muat Ulang Model & Data", use_container_width=True,
                 help="Gunakan setelah melatih ulang model (model.pkl/data terbaru) — cache Streamlit tidak otomatis mendeteksi perubahan file."):
        st.cache_resource.clear()
        st.rerun()

ICD_MAP = {
    'A': 'Bab A (Penyakit Infeksi & Parasit)', 'B': 'Bab B (Penyakit Infeksi & Parasit)',
    'C': 'Bab C (Tumor/Neoplasma)', 'D': 'Bab D (Neoplasma & Darah)',
    'E': 'Bab E (Endokrin, Nutrisi & Metabolik)', 'F': 'Bab F (Gangguan Mental & Perilaku)',
    'G': 'Bab G (Sistem Saraf)', 'H': 'Bab H (Mata & Telinga)',
    'I': 'Bab I (Sistem Sirkulasi / Jantung)', 'J': 'Bab J (Sistem Pernapasan)',
    'K': 'Bab K (Sistem Pencernaan)', 'L': 'Bab L (Kulit & Jaringan Subkutan)',
    'M': 'Bab M (Sistem Otot & Tulang)', 'N': 'Bab N (Sistem Kemih & Kelamin)',
    'O': 'Bab O (Kehamilan & Persalinan)', 'P': 'Bab P (Kondisi Perinatal)',
    'Q': 'Bab Q (Kelainan Bawaan)', 'R': 'Bab R (Gejala & Tanda Klinis)',
    'S': 'Bab S (Cedera & Keracunan)', 'T': 'Bab T (Cedera & Keracunan)',
    'U': 'Bab U (Kode Khusus / COVID-19)', 'V': 'Bab V (Sebab Eksternal)',
    'W': 'Bab W (Sebab Eksternal)', 'X': 'Bab X (Sebab Eksternal)',
    'Y': 'Bab Y (Sebab Eksternal)', 'Z': 'Bab Z (Status Kesehatan & Pelayanan)'
}

# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN 1: PREDIKSI INDIVIDU
# ══════════════════════════════════════════════════════════════════════════════
if menu == "Prediksi Individu":
    st.markdown("""
    <div class="page-header">
        <h2>Prediksi Lama Rawat Inap</h2>
        <p>Masukkan data klinis dan demografi pasien untuk memprediksi probabilitas kategori Length of Stay (LoS).</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Data Demografi</div>", unsafe_allow_html=True)
        st.caption("Usia (isi salah satu atau kombinasi — mengikuti format data asli 'Th/Bl/Hr')")
        c_th, c_bl, c_hr = st.columns(3)
        with c_th:
            usia_th = st.number_input("Tahun", min_value=0, max_value=120, value=0, step=1)
        with c_bl:
            usia_bl = st.number_input("Bulan", min_value=0, max_value=11, value=0, step=1)
        with c_hr:
            usia_hr = st.number_input("Hari", min_value=0, max_value=30, value=0, step=1)
        usia = round(usia_th + usia_bl / 12 + usia_hr / 365, 4)
        jenis_kelamin = st.selectbox("Jenis Kelamin", encoders['jenis_kelamin'].classes_.tolist())
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Data Klinis</div>", unsafe_allow_html=True)
        ruang_perawatan = st.selectbox("Ruang Perawatan", encoders['ruang_perawatan'].classes_.tolist())
        jenis_perawatan = st.selectbox("Jenis Perawatan", encoders['jenis_perawatan'].classes_.tolist())
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Admisi & Diagnosis</div>", unsafe_allow_html=True)
        cara_datang = st.selectbox("Cara Datang", encoders['cara_datang'].classes_.tolist())
        hari_masuk = st.selectbox("Hari Masuk", encoders['hari_masuk'].classes_.tolist())
        bab_diagnosis = st.selectbox("Bab ICD-10", encoders['bab_diagnosis'].classes_.tolist(), format_func=lambda x: ICD_MAP.get(x, x), help="Huruf pertama kode ICD-10")
        st.markdown('</div>', unsafe_allow_html=True)
        
    if st.button("Proses Prediksi", use_container_width=True):
        input_data = {
            'usia': usia, 'jenis_kelamin': jenis_kelamin,
            'bab_diagnosis': bab_diagnosis, 'ruang_perawatan': ruang_perawatan,
            'jenis_perawatan': jenis_perawatan, 'cara_datang': cara_datang, 'hari_masuk': hari_masuk
        }
        X_input = encode_input(input_data)
        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        
        st.markdown("<hr style='margin: 32px 0;'>", unsafe_allow_html=True)
        
        if usia < 0.1:
            st.warning(
                "ℹ️ **Catatan**: pasien neonatus (usia <±1 bulan) jumlahnya kecil secara proporsional "
                "dalam data pelatihan (±3,9% dari total data, meski kini terwakili penuh sesuai populasi "
                "aslinya). Selain itu, recall model untuk kelas 'Panjang' secara umum masih terbatas "
                "(lihat halaman *Perbandingan Model*). Gunakan prediksi ini sebagai alat bantu pendukung, "
                "bukan dasar tunggal keputusan klinis."
            )

        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown("### Hasil Prediksi")
            st.markdown(f"<div class='{BADGE_CSS[pred]}'>{LABEL[pred]}</div>", unsafe_allow_html=True)
            st.caption(
                "Model memiliki F1-macro ±0,55 dan recall kelas 'Panjang' yang masih terbatas — "
                "gunakan sebagai alat bantu pendukung, bukan pengganti penilaian klinis. "
                "Lihat halaman *Perbandingan Model* untuk detail performa."
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Probabilitas Kelas")
            df_prob = pd.DataFrame({'Kategori': LABEL_SHORT, 'Probabilitas': proba * 100})
            fig_prob = px.bar(df_prob, x='Probabilitas', y='Kategori', orientation='h', color='Kategori',
                              color_discrete_sequence=WARNA_BAR, text_auto='.1f')
            fig_prob.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), xaxis_title="%", yaxis_title="", showlegend=False, 
                                   plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=True, gridcolor='#e2e8f0', range=[0, 100]))
            st.plotly_chart(fig_prob, use_container_width=True)
            
        with r2:
            st.markdown("### Analisis Fitur Penggerak (SHAP)")
            st.info("Visualisasi ini menggunakan **SHAP Values** untuk menjelaskan alasan di balik prediksi. "
                    "Bar mengarah ke **kanan (merah)** meningkatkan kemungkinan pasien dirawat lama pada kelas prediksi, "
                    "sedangkan **kiri (biru)** menurunkannya.")
            
            exp = shap_explainer(X_input)
            label_tampil = []
            for i, col in enumerate(FITUR):
                val_encoded = X_input.iloc[0][col]
                if col in encoders and val_encoded != -1:
                    label_tampil.append(encoders[col].inverse_transform([int(val_encoded)])[0])
                else:
                    label_tampil.append(str(val_encoded))
            
            # Kita fokus pada kelas yang diprediksi
            shap_vals = exp.values[0, :, pred]
            df_shap = pd.DataFrame({'Fitur': FITUR, 'Nilai Asli': label_tampil, 'Kontribusi': shap_vals})
            df_shap['Teks'] = df_shap['Fitur'] + " = " + df_shap['Nilai Asli'].astype(str)
            df_shap = df_shap.sort_values(by='Kontribusi', key=abs, ascending=True)
            
            fig_shap = px.bar(df_shap, x='Kontribusi', y='Teks', orientation='h', 
                              color=df_shap['Kontribusi'] > 0, 
                              color_discrete_map={True: '#ef4444', False: '#3b82f6'})
            fig_shap.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), showlegend=False, yaxis_title="", 
                                   plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=True, gridcolor='#e2e8f0'))
            st.plotly_chart(fig_shap, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN 2: PREDIKSI MASSAL
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Prediksi Massal":
    st.markdown("""
    <div class="page-header">
        <h2>Prediksi Lama Rawat Inap — Massal</h2>
        <p>Unggah file CSV atau Excel (.xlsx) untuk memprediksi banyak pasien sekaligus dalam hitungan detik.</p>
    </div>
    """, unsafe_allow_html=True)
    
    file = st.file_uploader("Unggah file", type=['csv', 'xlsx'])

    with st.expander("📄 Belum punya format data? Unduh template di sini"):
        st.markdown(
            "Kolom wajib: `usia` (format `'Th'`/`'Bl'`/`'Hr'`, mis. `31 Th 4 Bl 13 Hr`, atau angka tahun desimal), "
            "`jenis_kelamin`, `bab_diagnosis`, `ruang_perawatan`, `jenis_perawatan`, `cara_datang`, `hari_masuk`."
        )
        df_template = pd.DataFrame([{
            'usia': '31 Th 4 Bl 13 Hr',
            'jenis_kelamin': encoders['jenis_kelamin'].classes_[0],
            'bab_diagnosis': encoders['bab_diagnosis'].classes_[0],
            'ruang_perawatan': encoders['ruang_perawatan'].classes_[0],
            'jenis_perawatan': encoders['jenis_perawatan'].classes_[0],
            'cara_datang': encoders['cara_datang'].classes_[0],
            'hari_masuk': encoders['hari_masuk'].classes_[0],
        }])
        def to_excel(df):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')
            return output.getvalue()
            
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Unduh Template (.csv)", data=df_template.to_csv(index=False).encode('utf-8'),
                                file_name="template_prediksi_massal.csv", mime="text/csv", use_container_width=True)
        with col2:
            st.download_button("Unduh Template (.xlsx)", data=to_excel(df_template),
                                file_name="template_prediksi_massal.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

    if file:
        if file.name.endswith('.xlsx'):
            df_upload = pd.read_excel(file)
        else:
            df_upload = pd.read_csv(file)
        st.markdown("### Preview Data Input")
        st.dataframe(df_upload.head(), use_container_width=True)
        if st.button("Jalankan Prediksi", type="primary", use_container_width=True):
            X_all, mask_unknown = encode_dataframe(df_upload)
            preds  = model.predict(X_all)
            probas = model.predict_proba(X_all)
            
            hasil = pd.DataFrame({
                'Prediksi': [LABEL[p] for p in preds],
                'Prob Pendek': np.round(probas[:, 0] * 100, 1),
                'Prob Sedang': np.round(probas[:, 1] * 100, 1),
                'Prob Panjang': np.round(probas[:, 2] * 100, 1)
            })
            df_hasil = pd.concat([df_upload.reset_index(drop=True), hasil], axis=1)
            st.success(f"Prediksi selesai untuk {len(df_upload)} baris pasien!")
            if mask_unknown.any():
                st.warning(
                    f"⚠️ {mask_unknown.sum()} dari {len(df_upload)} baris memiliki nilai kategori "
                    f"(mis. ruang perawatan, cara datang) yang tidak dikenali oleh model karena tidak "
                    f"pernah muncul di data pelatihan. Prediksi untuk baris tersebut kurang dapat diandalkan."
                )
                df_hasil['Kategori_Tidak_Dikenal'] = mask_unknown.values
            st.dataframe(df_hasil, use_container_width=True)
            
            st.markdown("### Ringkasan Prediksi")
            dist = hasil['Prediksi'].value_counts().reindex([LABEL[i] for i in [0,1,2]]).fillna(0)
            fig = px.bar(x=LABEL_SHORT, y=dist.values, color=LABEL_SHORT, color_discrete_sequence=WARNA_BAR, text_auto=True)
            fig.update_layout(title="", height=300, showlegend=False, xaxis_title="Kategori", yaxis_title="Jumlah Pasien", plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#e2e8f0'))
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("Unduh File Hasil (.csv)", data=df_hasil.to_csv(index=False).encode('utf-8'), file_name="hasil_prediksi_massal.csv", mime="text/csv", use_container_width=True)
            with col2:
                st.download_button("Unduh File Hasil (.xlsx)", data=to_excel(df_hasil), file_name="hasil_prediksi_massal.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN 3: KAPASITAS & TREN
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Kapasitas & Tren":
    st.markdown("""
    <div class="page-header">
        <h2>Kapasitas & Tren Ruang Rawat</h2>
        <p>Analisis tren pasien dan prediksi beban ruang perawatan berdasarkan data riwayat rumah sakit.</p>
    </div>
    """, unsafe_allow_html=True)
    
    path_clean = os.path.join(BASE, 'data', 'data_clean.csv')
    if os.path.exists(path_clean):
        df_kap = pd.read_csv(path_clean, parse_dates=['tgl_masuk'])
        df_kap['periode'] = df_kap['tgl_masuk'].dt.to_period('M').astype(str)
        
        BULAN_INDO = {'01': 'Januari', '02': 'Februari', '03': 'Maret', '04': 'April', '05': 'Mei', '06': 'Juni', '07': 'Juli', '08': 'Agustus', '09': 'September', '10': 'Oktober', '11': 'November', '12': 'Desember'}
        periode_pilih = st.selectbox("Pilih Periode Laporan", sorted(df_kap['periode'].unique()), index=len(df_kap['periode'].unique())-1, format_func=lambda x: f"{BULAN_INDO.get(x.split('-')[1], '')} {x.split('-')[0]}" if '-' in x else x)
        df_periode = df_kap[df_kap['periode'] == periode_pilih]
        
        st.markdown("<br>", unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        with k1: st.markdown(f"<div class='metric-box'><div class='metric-val'>{len(df_periode)}</div><div class='metric-lbl'>Total Kunjungan Masuk</div></div>", unsafe_allow_html=True)
        with k2: st.markdown(f"<div class='metric-box'><div class='metric-val'>{df_periode['los'].mean():.1f} Hari</div><div class='metric-lbl'>Rata-rata Length of Stay</div></div>", unsafe_allow_html=True)
        with k3: 
            n_panjang = (df_periode['los_kategori'] == 2).sum()
            st.markdown(f"<div class='metric-box {'alert' if n_panjang/len(df_periode)>0.1 else ''}'><div class='metric-val'>{n_panjang}</div><div class='metric-lbl'>Pasien LoS >5 Hari (Panjang)</div></div>", unsafe_allow_html=True)
        
        st.markdown("<br><div class='card'>", unsafe_allow_html=True)
        st.markdown("### Beban Kategori per Ruang Perawatan")
        ringkas_ruang = df_periode.groupby('ruang_perawatan')['los_kategori'].value_counts().unstack().fillna(0).reindex(columns=[0,1,2])
        ringkas_ruang.columns = LABEL_SHORT
        fig = px.bar(ringkas_ruang, barmode='stack', color_discrete_sequence=WARNA_BAR)
        fig.update_layout(height=450, xaxis_title="Ruang Perawatan", yaxis_title="Jumlah Pasien", legend_title="Kategori LoS", plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(showgrid=True, gridcolor='#e2e8f0'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Data operasional tidak ditemukan. Pastikan pipeline data (1_cleaning.py) telah berjalan.")

# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN 4: PERBANDINGAN MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Perbandingan Model":
    st.markdown("""
    <div class="page-header">
        <h2>Evaluasi & Perbandingan Model</h2>
        <p>Analisis mendalam mengenai kinerja arsitektur Random Forest vs XGBoost dalam studi kasus imbalanced data ini.</p>
    </div>
    """, unsafe_allow_html=True)
    
    metrik_path = os.path.join(BASE, 'data', 'metrik.json')
    cm_path     = os.path.join(BASE, 'data', 'cm.json')
    
    if os.path.exists(metrik_path):
        with open(metrik_path) as f: metrik_all = json.load(f)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Komparasi Metrik Keseluruhan (Test Set)")
        df_cmp = pd.DataFrame({k: {mk: mv for mk, mv in v.items() if mk != 'per_kelas'} for k, v in metrik_all.items()}).T.round(4)
        
        fig = go.Figure()
        for metric in df_cmp.columns:
            fig.add_trace(go.Bar(name=metric, x=df_cmp.index, y=df_cmp[metric], text=df_cmp[metric], textposition='auto'))
        fig.update_layout(barmode='group', height=400, plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(showgrid=True, gridcolor='#e2e8f0'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Recall Kelas 'Panjang'")
        st.info("Tujuan utama digunakannya teknik SMOTENC adalah untuk meningkatkan kemampuan model mendeteksi kelas 'Panjang' (pasien yang akan dirawat lebih dari 5 hari) yang jumlahnya sangat sedikit. Skor Recall di bawah ini menunjukkan persentase kelas Panjang yang berhasil ditangkap.")
        c1, c2 = st.columns(2)
        for i, (nama, m) in enumerate(metrik_all.items()):
            col = c1 if i==0 else c2
            rec = m['per_kelas']['Panjang']['recall']
            col.markdown(f"<div class='metric-box'><div class='metric-lbl'>Recall ({nama})</div><div class='metric-val'>{rec*100:.1f}%</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Matriks Konfusi (Confusion Matrix)")
        if os.path.exists(cm_path):
            with open(cm_path) as f: cm_all = json.load(f)
            c1, c2 = st.columns(2)
            for i, nama in enumerate(['XGBoost', 'Random Forest']):
                col = c1 if i==0 else c2
                fig_cm = px.imshow(cm_all[nama], text_auto=True, color_continuous_scale='Blues',
                                   x=LABEL_SHORT, y=LABEL_SHORT)
                fig_cm.update_layout(title=f"CM {nama}", xaxis_title="Prediksi", yaxis_title="Kenyataan Aktual", height=400)
                col.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
                
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Interpretasi Fitur Menggunakan SHAP")
        shap_json_path = os.path.join(BASE, 'data', 'shap_summary.json')
        if os.path.exists(shap_json_path):
            with open(shap_json_path) as f: shap_json = json.load(f)
            c1, c2 = st.columns(2)
            for i, nama in enumerate(['XGBoost', 'Random Forest']):
                col = c1 if i==0 else c2
                df_sh = pd.DataFrame(list(shap_json[nama].items()), columns=['Fitur', 'Kepentingan']).sort_values('Kepentingan')
                fig_sh = px.bar(df_sh, x='Kepentingan', y='Fitur', orientation='h', color_discrete_sequence=['#3b82f6'])
                fig_sh.update_layout(title=f"SHAP Global - {nama}", height=400, xaxis_title="Rata-rata |SHAP Value|", plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=True, gridcolor='#e2e8f0'))
                col.plotly_chart(fig_sh, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Data metrik belum tersedia. Silakan jalankan `3_evaluation.py` terlebih dahulu.")