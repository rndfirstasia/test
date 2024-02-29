import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import plost
from streamlit_gsheets import GSheetsConnection

#GLOBAL
#Layout
st.set_page_config(page_title="Data Analytic", page_icon="icon.png")
#Hide warning
st.set_option('deprecation.showPyplotGlobalUse', False)

#df
#df = pd.read_excel('data.xlsx')

# File upload
#uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])

#if uploaded_file is not None:
    #df = pd.read_excel(uploaded_file)
    # Now you can use the DataFrame 'df' as needed
    #st.write(df)

#gsheet
#conn = st.connection("gsheets", type=GSheetsConnection)
#df = conn.read(worksheet="Sheet1")

url = "https://docs.google.com/spreadsheets/d/1iHmbhpqlSssw6WCMCvbe6PeMED5pdgLee4CUqhCi318/edit?usp=sharing"

conn = st.connection("gsheets", type=GSheetsConnection)

df = conn.read(spreadsheet=url, worksheet="569635382")
st.dataframe(df)

#Sidebar
#st.sidebar.title('Search & Filter')
#st.sidebar.subheader('Filter:')
#filter tanggal
df['TGL TEST'] = pd.to_datetime(df['TGL TEST'])
start_date = st.sidebar.date_input('Tanggal Mulai', value=pd.to_datetime('today') - pd.Timedelta(days=30))
end_date = st.sidebar.date_input('Tanggal Akhir', value=pd.to_datetime('today'))
df = df[(df['TGL TEST'].dt.date >= start_date) & (df['TGL TEST'].dt.date <= end_date)]
bins= [0,24,40,56,75, np.inf]
labels_umur = ['Gen Z', 'Milenials', 'Gen X', 'Baby Boomer', 'Silent']
df['generasi'] = pd.cut(df['UMUR'], bins=bins, labels=labels_umur, right=False)
#Filter generasi
selected_generations = st.sidebar.multiselect('Pilih Generasi', labels_umur, default=labels_umur)
df = df[df['generasi'].isin(selected_generations)]
#Filter Managerial & Non
labels_mng = ['Managerial','Non Managerial']
selected_managerial = st.sidebar.multiselect('Pilih Managerial/Non', labels_mng, default=labels_mng)
df = df[df['MANAGERIAL'].isin(selected_managerial)]
#Filter Job Family
labels_fmly = ['Financial','Information Technology','Operation','People','Sales Marketing']
selected_jobfam = st.sidebar.multiselect('Pilih Job Family', labels_fmly, default=labels_fmly)
df = df[df['JOB FAMILY'].isin(selected_jobfam)]

#st.sidebar.divider()
#st.sidebar.subheader('Search:')
# Search
pilihan_klien_unik = pd.Series(df['KLIEN']).drop_duplicates().tolist()
pilihan_nama_unik = pd.Series(df['NAMA']).drop_duplicates().tolist()

selected_kliens = st.sidebar.multiselect('Pilih Klien', pilihan_klien_unik)

if selected_kliens:
    df = df[df['KLIEN'].isin(selected_kliens)]
else:
    st.sidebar.write('Tidak ada klien yang dipilih')

selected_namas = st.sidebar.multiselect('Pilih Nama peserta', pilihan_nama_unik)

if selected_namas:
    df = df[df['NAMA'].isin(selected_namas)]
else:
    st.sidebar.write('Tidak ada nama yang dipilih')

#Dataframe Tertampil
st.title('Data Tarikan')
st.markdown('Data yang dimunculkan dapat dirubah sesuai dengan filter yang diinginkan')
st.markdown('Default tanggal adalah satu bulan dan filter generasi disesuaikan dengan umur tahun sekarang (bukan tahun dia tes).')
st.markdown('Filter Generasi dibagi menjadi')
kategori_umur = {
  "Generasi": ["Gen Z", "Milenials", "Gen X", "Baby Boomer", "Silent"],
  "Rentang Tahun Lahir": ["Mid-1990s hingga Early-2010s", "Early-1980s hingga Mid-1990s", "Early-1960s hingga Late-1970s", "Mid-1940s hingga Early-1960s", "Early-1920s hingga Mid-1940s"],
  "Kategori Umur (per 2024)": ["14 - 29 tahun", "29 - 44 tahun", "45 - 64 tahun", "64 - 80 tahun", "80+ tahun"]
}
kategori_umur_df = pd.DataFrame(kategori_umur)
st.table(kategori_umur_df)
df

def get_median(series):
    return np.median(series)

def get_mode(series):
    series = np.asarray(series.dropna()) if hasattr(series, 'dropna') else np.asarray(series)
    if series.size == 0:
        return None

    values, counts = np.unique(series, return_counts=True)
    if counts.size == 0:
        return None
      
    index = np.argmax(counts)
    return values[index]

def get_mean(series):
    return np.mean(series)

def get_std(series):
    return np.std(series, ddof=1)

def get_variance(series):
    return np.var(series, ddof=1)

def get_range(series):
    return np.ptp(series)

def get_quartiles(series):
    series_clean = np.asarray(series.dropna()) if hasattr(series, 'dropna') else np.asarray(series)
    
    if series_clean.size == 0:
        return None, None, None
    
    # Calculate percentiles
    quartiles = np.percentile(series_clean, [25, 50, 75])
    return quartiles 

def get_skewness(series):
    n = len(series)
    mean = np.mean(series)
    std = np.std(series, ddof=1)
    skewness = (np.sum((series - mean) ** 3) / n) / (std ** 3)
    return skewness

def get_kurtosis(series):
    n = len(series)
    mean = np.mean(series)
    std = np.std(series, ddof=1)
    kurtosis = (np.sum((series - mean) ** 4) / n) / (std ** 4) - 3
    return kurtosis

descriptive = {
    'Median' : [get_median(df['IQ']), get_median(df['S']), get_median(df['T']), get_median(df['A']), get_median(df['G']), get_median(df['E'])],
    'Mode' : [get_mode(df['IQ']), get_mode(df['S']), get_mode(df['T']), get_mode(df['A']), get_mode(df['G']), get_mode(df['E'])],
    'Mean' : [get_mean(df['IQ']), get_mean(df['S']), get_mean(df['T']), get_mean(df['A']), get_mean(df['G']), get_mean(df['E'])],
    'Std' : [get_std(df['IQ']), get_std(df['S']), get_std(df['T']), get_std(df['A']), get_std(df['G']), get_std(df['E'])],
    'Var' : [get_variance(df['IQ']), get_variance(df['S']), get_variance(df['T']), get_variance(df['A']), get_variance(df['G']), get_variance(df['E'])],
    'Range' : [get_range(df['IQ']), get_range(df['S']), get_range(df['T']), get_range(df['A']), get_range(df['G']), get_range(df['E'])],
    'Percentile 25' : [get_quartiles(df['IQ'])[0], get_quartiles(df['S'])[0], get_quartiles(df['T'])[0], get_quartiles(df['A'])[0], get_quartiles(df['G'])[0], get_quartiles(df['E'])[0]],
    'Percentile 50' : [get_quartiles(df['IQ'])[1], get_quartiles(df['S'])[1], get_quartiles(df['T'])[1], get_quartiles(df['A'])[1], get_quartiles(df['G'])[1], get_quartiles(df['E'])[1]],
    'Percentile 75' : [get_quartiles(df['IQ'])[2], get_quartiles(df['S'])[2], get_quartiles(df['T'])[2], get_quartiles(df['A'])[2], get_quartiles(df['G'])[2], get_quartiles(df['E'])[2]],
    'Skew' : [get_skewness(df['IQ']), get_skewness(df['S']), get_skewness(df['T']), get_skewness(df['A']), get_skewness(df['G']), get_skewness(df['E'])],
    'Kurt' : [get_kurtosis(df['IQ']), get_kurtosis(df['S']), get_kurtosis(df['T']), get_kurtosis(df['A']), get_kurtosis(df['G']), get_kurtosis(df['E'])]
}

desc_df = pd.DataFrame(descriptive, index=['IQ', 'S', 'T', 'A', 'G', 'E'])

st.title('Descriptive Stats')
st.write(len(df))

st.table(desc_df)

#Jenis Kelamin
st.subheader('Jenis Kelamin')

distribusi = df['J/K'].value_counts().rename(index={'L': 'Laki-laki', 'P': 'Perempuan'})

def func(pct, allvalues):
    absolute = int(pct/100.*np.sum(allvalues))
    return "{:.1f}%\n({:d})".format(pct, absolute)

plt.figure(figsize=(8, 6))
plt.pie(distribusi, labels=distribusi.index, colors=sns.color_palette('pastel'), autopct=lambda pct: func(pct, distribusi), startangle=140)

#centre_circle = plt.Circle((0,0),0.70,fc='white')
#fig = plt.gcf()
#fig.gca().add_artist(centre_circle)

plt.title('Distribusi Jenis Kelamin')
plt.tight_layout()
st.pyplot(plt)

#Kesimpulan
map_kesimpulan = {'Dapat Disarankan': 4, 'Dapat Potensial': 4, 'Dapat Sesuai': 4,
               'Cukup Disarankan': 3, 'Cukup Potensial': 3, 'Cukup Sesuai': 3,
               'Kurang Disarankan': 2, 'Kurang Potensial': 2, 'Kurang Sesuai': 2,
               'Tidak Disarankan': 1, 'Tidak Potensial': 1, 'Tidak Sesuai': 1}

df['KESIMPULAN'] = df['KESIMPULAN'].replace(map_kesimpulan)
check = df['KESIMPULAN']

count_df = df['KESIMPULAN'].value_counts().sort_index()
percent_df = df['KESIMPULAN'].value_counts(normalize=True).sort_index() * 100

plt.figure(figsize=(10, 7))
plt.pie(count_df, labels=['Tidak Rekomendasi (1)', 'Kurang Rekomendasi (2)', 'Cukup Rekomendasi (3)', 'Rekomendasi (4)'],autopct='%1.1f%%', startangle=140)

plt.title('Distribusi Kesimpulan')
st.pyplot(plt)

#IQ
st.title('Graph Distribusi IQ')
st.write("""
## Perbandingan Histogram IQ dan IQ Ori

Dengan membandingkan kedua histogram, kita dapat mengamati perbedaan antara skor yang telah dikalibrasi oleh Assessor dan skor asli, memberikan wawasan tentang seberapa signifikan dampak penyesuaian yang dilakukan oleh Assessor terhadap distribusi skor IQ.

1. **IQ (Adjusted Intelligence Quotient)**:
Histogram ini akan menampilkan frekuensi skor IQ setelah dilakukan penyesuaian oleh seorang assessor. Penyesuaian yang dilakukan menyesuaikan dengan buku kasus. Hasilnya adalah skor yang dianggap lebih mewakili kemampuan intelektual individu dalam kondisi penilaian yang telah disempurnakan.

2. **IQ Ori (Original Intelligence Quotient)**:
Histogram ini menampilkan frekuensi dari skor IQ asli tanpa penyesuaian, yang berasal langsung dari hasil tes peserta. Skor ini mencerminkan performa individu berdasarkan kriteria dan standar tes yang telah ditetapkan, tanpa memasukkan koreksi atau penilaian tambahan dari seorang assessor.
""")

def iq_category(iq):
    if iq > 130:
        return 'Very Superior'
    elif iq >= 120 and iq <= 129:
        return 'Superior'
    elif iq >= 110 and iq <= 119:
        return 'High Average'
    elif iq >= 90 and iq <= 109:
        return 'Average'
    elif iq >= 80 and iq <= 89:
        return 'Low Average'
    elif iq >= 70 and iq <= 79:
        return 'Borderline'
    else:
        return 'Extremely Low'

df['IQCategory'] = df['IQOri'].apply(iq_category)
all_iq_categories = ['Extremely Low', 'Borderline', 'Low Average', 'Average', 'High Average', 'Superior', 'Very Superior']
category_iq_counts = df['IQCategory'].value_counts().reindex(all_iq_categories, fill_value=0)

count_iq = pd.DataFrame({'IQ LEVEL': all_iq_categories, 'Count': category_iq_counts})

fig, ax = plt.subplots()
ax.bar(count_iq['IQ LEVEL'], count_iq['Count'])
ax.set_xlabel('IQ Category')
ax.set_ylabel('Count')
ax.set_title('IQ Distribution by Wechsler Scale')
plt.xticks(rotation=45, ha='right')

st.pyplot(fig)

st.text_area(label="Tambahkan Analisa:", key="iq_text_area")

#STAGE
st.title('Graph Distribusi STAGE')
st.write("""
Analisis histogram kepribadian STAGE memperlihatkan bagaimana skor individu berdistribusi di lima domain utama:

1. **Stability**: Berhubungan dengan kontrol emosi dan ketahanan stres.
2. **Tenacity**: Mengukur tingkat kesadaran, organisasi, dan bertanggung jawab.
3. **Adaptability**: Menunjukkan kreativitas, keingintahuan, dan penerimaan terhadap pengalaman baru.
4. **Genuineness**: Refleksi dari keramahan, empati, dan perilaku kooperatif.
5. **Extraversion**: Menilai energi sosial dan kecenderungan untuk berinteraksi dengan orang lain.

Dengan membandingkan histogram-histogram ini, kita dapat menggambarkan variasi dalam lima aspek kunci STAGE dalam database.
""")

df_stage_profil = df[['S_Trait', 'T_Trait', 'A_Trait', 'G_Trait', 'E_Trait']]

freq_dfs = []
total_counts = df_stage_profil.count() 
for trait in df_stage_profil.columns:
    freq_series = df_stage_profil[trait].value_counts(normalize=True) * 100
    raw_counts = df_stage_profil[trait].value_counts()  
    freq_df = freq_series.reset_index()
    freq_df['Count'] = freq_df['index'].apply(lambda x: raw_counts[x])  
    freq_df.columns = ['Value', 'Percentage', 'Count']
    freq_df['Trait'] = trait
    freq_dfs.append(freq_df)

final_df = pd.concat(freq_dfs)

traits = df_stage_profil.columns.tolist()
values = final_df['Value'].unique()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

fig, ax = plt.subplots(figsize=(10, 8))

starts = {trait: 0 for trait in traits}

for index, value in enumerate(values):
    percentages = []
    raw_counts = []
    for trait in traits:
        df_subset = final_df[(final_df['Trait'] == trait) & (final_df['Value'] == value)]
        if df_subset.empty:
            percentages.append(0)
            raw_counts.append(0)
        else:
            percentages.append(df_subset['Percentage'].values[0])
            raw_counts.append(df_subset['Count'].values[0])

    color = colors[index % len(colors)]
    bars = ax.barh(traits, percentages, left=[starts[trait] for trait in traits], color=color, label=value if index < len(colors) else "")

    for i, trait in enumerate(traits):
        starts[trait] += percentages[i]
        if percentages[i] > 0:  
            ax.annotate(f"{percentages[i]:.2f}%\n({raw_counts[i]})",
                        xy=(starts[trait] - percentages[i] / 2, i),
                        xytext=(0, 0),  
                        textcoords="offset points",
                        ha='center', va='center',
                        color='black', fontsize=8)

ax.set_xlabel('Percentage')
ax.set_title('Simplified Traits Distribution with Annotations')

plt.tight_layout()
st.pyplot(plt)

#Agility
st.title('Agility Index')
st.write("""
Agility (ketangkasan) adalah kemampuan untuk memahami keadaan dengan cepat dan beradaptasi atau menyesuaikan diri secara efektif di dalamnya. Semakin tinggi Agility Index seseorang, menunjukkan kemudahan bagi dirinya untuk mengatasi kondisi yang dinamis di pekerjaan dan perusahaan.
""")

df_agility = df[["People Agility", "Mental Agility", "Self-Awareness", "Result Agility", "Change Agility"]]

def calculate_counts(column):
    low_count = (column == 0).sum()
    medium_count = (column == 1).sum()
    high_count = (column == 2).sum()
    return low_count, medium_count, high_count

count_agility = []

for column_name in df_agility:
    counts = calculate_counts(df_agility[column_name])
    count_agility.append([column_name] + list(counts))

count_agility_df = pd.DataFrame(count_agility, columns=['Agility Type', 'Low', 'Medium', 'High'])

melted_df = count_agility_df.melt(id_vars=["Agility Type"], var_name="Agility Level", value_name="Count")

plt.figure(figsize=(10, 6)) 
sns.barplot(x='Agility Type', y='Count', hue='Agility Level', data=melted_df)

plt.title('Agility Comparison')  
plt.xticks(rotation=45)  
plt.tight_layout()  
st.pyplot(plt)

# Agility Level
count_agility_level = df['AGILITY LEVEL'].value_counts()
count_agility_level_df = pd.DataFrame(count_agility_level).reset_index()
count_agility_level_df.columns = ['AGILITY LEVEL', 'Count']

order = ['L', 'M', 'H']

st.subheader('Agility Level')

plt.figure(figsize=(10, 6))
sns.barplot(data=count_agility_level_df, x='AGILITY LEVEL', y='Count', order=order)
plt.title('Agility Level')
plt.xlabel('AGILITY LEVEL')
plt.ylabel('Count')
st.pyplot(plt)

#Stage
st.title('STAGE')
st.write("""
Berikut adalah distribusi dari STAGE:
""")
plt.figure(figsize=(10, 8))
sns.kdeplot(df['S'], label='Stability')
sns.kdeplot(df['T'], label='Tenacity')
sns.kdeplot(df['A'], label='Adaptability')
sns.kdeplot(df['G'], label='Genuineness')
sns.kdeplot(df['E'], label='Extraversion')
plt.title('Distribusi Data STAGE')
plt.xlabel('T Score')
plt.ylabel('Frekuensi')
plt.legend()
st.pyplot(plt)

#Stage Facet - Stability
st.title('Facet Stability')
st.write("""
Berikut adalah distribusi dari facet stability:
""")

plt.figure(figsize=(10, 8))
sns.kdeplot(df['S1'], label='Worry')
sns.kdeplot(df['S2'], label='Calmness')
sns.kdeplot(df['S3'], label='Optimism')
sns.kdeplot(df['S4'], label='Recovery')
plt.title('Distribusi Data Facet S')
plt.xlabel('T Score')
plt.ylabel('Frekuensi')
plt.legend()
plt.show()
st.pyplot(plt)

#Stage Facet - Tenacity
st.title('Facet Tenacity')
st.write("""
Berikut adalah distribusi dari facet tenacity:
""")

plt.figure(figsize=(10, 8))
sns.kdeplot(df['T1'], label='Excellence')
sns.kdeplot(df['T2'], label='Systematic')
sns.kdeplot(df['T3'], label='Achievement Drive')
sns.kdeplot(df['T4'], label='Attentiveness')
sns.kdeplot(df['T5'], label='Deliberation')
plt.title('Distribusi Data Facet T')
plt.xlabel('T Score')
plt.ylabel('Frekuensi')
plt.legend()
plt.show()
st.pyplot(plt)

#Stage Facet - Adaptability
st.title('Facet Adaptability')
st.write("""
Berikut adalah distribusi dari facet adaptability:
""")

plt.figure(figsize=(10, 8))
sns.kdeplot(df['A1'], label='Altruism')
sns.kdeplot(df['A2'], label='Compliance')
sns.kdeplot(df['A3'], label='Modesty')
sns.kdeplot(df['A4'], label='Assertiveness')
plt.title('Distribusi Data Facet T')
plt.xlabel('T Score')
plt.ylabel('Frekuensi')
plt.legend()
st.pyplot(plt)

#Stage Facet - Genuineness
st.title('Facet Genuineness')
st.write("""
Berikut adalah distribusi dari facet genuineness:
""")

plt.figure(figsize=(10, 8))
sns.kdeplot(df['G1'], label='Innovation')
sns.kdeplot(df['G2'], label='Complexity')
sns.kdeplot(df['G3'], label='Flexibility')
sns.kdeplot(df['G4'], label='Wideness')
plt.title('Distribusi Data Facet G')
plt.xlabel('T Score')
plt.ylabel('Frekuensi')
plt.legend()
st.pyplot(plt)

#Stage Facet - Extraversion
st.title('Facet Extraversion')
st.write("""
Berikut adalah distribusi dari facet extraversion:
""")

plt.figure(figsize=(10, 8))
sns.kdeplot(df['E1'], label='Friendliness')
sns.kdeplot(df['E2'], label='Gregariousness')
sns.kdeplot(df['E3'], label='Energy')
sns.kdeplot(df['E4'], label='Leading Incharge')
sns.kdeplot(df['E5'], label='Trust')
sns.kdeplot(df['E6'], label='Courtesy')
plt.title('Distribusi Data Facet E')
plt.xlabel('T Score')
plt.ylabel('Frekuensi')
plt.legend()
st.pyplot(plt)

#Kompetensi
st.title('Kompetensi-kompetensi')
st.markdown('Catatan: niatnya sih nanti mau dibandingin kaya kalau lagi pilih pemain di winning, tapi lihat dulu, jadi sementara seperti ini')

df_komp = df[['Action Orientation','Obedience','Comfort with Ambiguity','Ambition','Analytical Thinking','Leadership','Business Acumen','Competitiveness','Creativity','Customer Service Orientation','Decision-Making Skills','Delegation','Development of Personel','Diplomacy','Tolerance with Diversity','Entrepreneurship','Conflict Management','Flexibility','Persistance','Visionary','Assessing','Independence','Informing Others','Innovation','Accuracy','Listening','Managing Through System','Facilitating','Talent Developing','Objectivity','Optimism','Organization','Comfort with repetitif work','Performance Focus','Planning','Political Savvy','Quality Orientation','Range of Perspective and Interests','Dependability','Responsibility','Risk Taking','Carefullness','Sales Orientation','Self Confidence','Self Control','Self Development','Teamwork and Cooperation','Technical Learning','Work/Life Balance']]

def create_radar_chart(ax, angles, values, labels, category_labels, plt):
    values += values[:1]
    angles += angles[:1]  
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)
    ax.set_xticks(angles[:-1])  
    ax.set_xticklabels(category_labels)
    
n_kompetensi = len(df_komp.columns)  
n_cols = 3  
n_rows = n_kompetensi // n_cols + (n_kompetensi % n_cols > 0)  

fig, axes = plt.subplots(figsize=(n_cols * 4, n_rows * 3), nrows=n_rows, ncols=n_cols, subplot_kw=dict(polar=True))
fig.subplots_adjust(hspace=0.5, wspace=0.5)  

categories = ['VH', 'H', 'M', 'L', 'VL']
labels = np.array(categories)

for i, (ax, column) in enumerate(zip(axes.flatten(), df_komp.columns)):
    
    frekuensi = df_komp[column].value_counts().reindex(categories, fill_value=0)
    values = frekuensi.values.flatten().tolist()

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

    create_radar_chart(ax, angles, values, labels, categories, plt)
    ax.set_title(column, size=11, color='red', y=1.1)

for i in range(n_kompetensi, n_rows * n_cols):
    fig.delaxes(axes.flatten()[i])

st.pyplot(plt)
