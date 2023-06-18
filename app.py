import streamlit as st
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pandas as pd
import plotly.express as px
import sqlite3
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import base64
import time


# Connect to a SQLite database file
conn = sqlite3.connect('example.db')

# Create a table
conn.execute('''
    CREATE TABLE IF NOT EXISTS comments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        comment TEXT,
        sentiment TEXT
    )
''')

# Load model
model = joblib.load('svm_model.pkl')

# Load vectorizer
vectorizer = joblib.load('vectorizer.pkl')

# Inisialisasi stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Preprocessing teks menggunakan Sastrawi
def preprocess_text(text):
    text = stemmer.stem(text) # stemming
    text = text.lower() # lowercase
    text = re.sub(r'\d+', '', text) # menghapus angka
    text = re.sub(r'[^\w\s]', '', text) # menghapus tanda baca
    text = re.sub(r'\s+', ' ', text) # menghapus spasi berlebihan
    text = text.strip() # menghapus spasi di awal dan akhir kalimat
    return text

# Fungsi untuk membuat prediksi sentimen
def predict_sentiment(comment):
    comment = preprocess_text(comment)
    comment_vectorized = vectorizer.transform([comment])
    sentiment = model.predict(comment_vectorized)[0]
    return sentiment

def predict_from_csv(comments, true_sentiments):
    # Membuat kolom sentimen prediksi
    predicted_sentiments = []
    
    with st.spinner("Sedang melakukan prediksi..."):
        progress_bar = st.progress(0)  # Widget progress untuk menampilkan persentase
        
        # Simulasi waktu prediksi
        for i, comment in enumerate(comments):
            sentiment = predict_sentiment(comment)
            predicted_sentiments.append(sentiment)
            
            # Menghitung persentase dan memperbarui widget progress
            progress = (i + 1) / len(comments)
            progress_bar.progress(progress)
            
            # Simulasi waktu delay antar iterasi
            time.sleep(0.1)
        
        # Menghitung nilai akurasi
        accuracy = accuracy_score(true_sentiments, predicted_sentiments)
        accuracy_percent = accuracy * 100
    
    return accuracy_percent, predicted_sentiments




def download_csv(df):
    # Mengonversi DataFrame menjadi file CSV
    csv = df.to_csv(index=False)
    
    # Membuat tautan unduh
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download CSV File</a>'
    
    return href



# Fungsi untuk memvisualisasikan data
@st.cache_data
def visualize_data(df):
    fig = px.pie(df, names='sentiment', title='Sentiment Distribution')

    return fig

# Fungsi untuk membuat wordcloud
def generate_wordcloud(df):
    words = ' '.join(df['comment'].tolist())
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(words)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis('off')
    return fig


html_code = '''
<div style="background-color: #5000ca; height: 150px; width: 100%; display: flex; justify-content: center; align-items: center;">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320"><path fill="#fff" fill-opacity="1" d="M0,224L80,218.7C160,213,320,203,480,202.7C640,203,800,213,960,208C1120,203,1280,181,1360,170.7L1440,160L1440,320L1360,320C1280,320,1120,320,960,320C800,320,640,320,480,320C320,320,160,320,80,320L0,320Z"></path></svg>
</div>
'''


# Halaman utama Streamlit
def main():
    st.set_page_config(page_title="Website Sentimen Komentar Instagram")

    

    # Membuat sidebar
    menu = ['Prediksi Sentimen', 'Visualisasi Data']
    choice = st.sidebar.selectbox('Pilih Menu', menu)
    
    
    # Mengatur tampilan sidebar dengan CSS
    st.markdown(
    """
    <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            background-color: #6760c0;
        }
    </style>
    """,
    unsafe_allow_html=True,
    )

    image_path = 'image/main.png'
    image = open(image_path, 'rb').read()
    image_encoded = base64.b64encode(image).decode('utf-8')

    main_content = f"""
    <style>
        .css-uf99v8.e1g8pov65 {{
            position: relative;
        }}
        
        .custom-image {{
            position: absolute;
            bottom: 0;
            right: 0;
            max-width: 200px; /* Adjust the maximum width as needed */
            height: auto;
        }}
    </style>

    <div class="css-uf99v8 e1g8pov65">
        <img class="custom-image" src="data:image/png;base64,{image_encoded}" alt="Image">
    </div>
    """

    st.markdown(main_content, unsafe_allow_html=True)



    if choice == 'Prediksi Sentimen':
        st.title('Prediksi Sentimen Komentar Instagram')
        
        # Menggunakan st.radio untuk memilih metode prediksi
        prediction_method = st.radio("Metode Prediksi:", ("Prediksi Komentar", "Unggah File CSV"))
        
        if prediction_method == "Prediksi Komentar":
            comment = st.text_area('Masukkan komentar:')
            if st.button("Submit"):
                if comment:
                    sentiment = predict_sentiment(comment)
                    if sentiment == 'positive':
                        st.success('Sentimen: {} :smile:'.format(sentiment))
                    else:
                        st.error('Sentimen: {} :disappointed:'.format(sentiment))
                    conn.execute('INSERT INTO comments (comment, sentiment) VALUES (?, ?)', (comment, sentiment))
                    conn.commit()
        elif prediction_method == "Unggah File CSV":
            csv_file = st.file_uploader("Unggah file CSV", type=['csv'])
            if csv_file is not None:
                # Membaca file CSV
                df = pd.read_csv(csv_file)
                
                # Mengambil daftar nama kolom
                column_names = df.columns.tolist()
                
                                # Menggunakan st.selectbox untuk memilih kolom komentar
                comment_column = st.selectbox("Pilih Kolom Komentar", column_names)
                
                # Menggunakan st.selectbox untuk memilih kolom sentimen
                sentiment_column = st.selectbox("Pilih Kolom Sentimen", column_names)
                
                # Mengambil kolom komentar
                comments = df[comment_column].astype(str)
                
                # Mengambil kolom sentimen yang benar
                true_sentiments = df[sentiment_column].astype(str)
                
                # Menghitung jumlah data yang diunggah
                num_data_uploaded = len(df)
                
                # Menggunakan st.checkbox untuk memilih semua data
                select_all_data = st.checkbox("Pilih Semua Data")
                
                # Menampilkan input jumlah data jika tidak memilih semua data
                if not select_all_data:
                    num_data_to_use = st.number_input("Jumlah Data yang Digunakan", min_value=1, max_value=num_data_uploaded, value=num_data_uploaded)
                else:
                    num_data_to_use = num_data_uploaded
                
                if st.button("Submit"):
                    # Mengambil sejumlah data yang dipilih
                    comments = comments[:num_data_to_use]
                    true_sentiments = true_sentiments[:num_data_to_use]
                    
                    accuracy, predicted_sentiments = predict_from_csv(comments, true_sentiments)
                    st.info("Akurasi: {}%".format(accuracy))
                    
                    # Menampilkan diagram pie untuk sentimen menggunakan Plotly
                    sentiment_counts = pd.Series(predicted_sentiments).value_counts()
                    fig = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index, 
                                title="Distribusi Sentimen")
                    st.plotly_chart(fig)



    elif choice == 'Visualisasi Data':
            st.title('Visualisasi Sentimen Komentar Instagram')
            # Load dataset
            df = pd.read_sql("SELECT * FROM comments", con=conn)
           
            
            st.checkbox("Use container width", value=False, key="use_container_width")
            st.data_editor(df, use_container_width=st.session_state.use_container_width)
            
            # Tombol untuk mengekspor DataFrame menjadi file CSV
            if st.button("Export to CSV"):
                csv_link = download_csv(df)
                st.markdown(csv_link, unsafe_allow_html=True)

            # Visualisasi data
            fig = visualize_data(df)
            st.plotly_chart(fig)
            
            
            # Menampilkan wordcloud
            st.title('Wordcloud Komentar Instagram')
            fig_wordcloud = generate_wordcloud(df)
            st.pyplot(fig_wordcloud)
 


if __name__ == '__main__':
    main()
