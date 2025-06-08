import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow.keras.layers import Embedding, Dense, Concatenate, Input, Dropout, Flatten
from tensorflow.keras.models import Model
import os

# Page configuration
st.set_page_config(page_title="UMKM-Investor Recommendation System", layout="wide")

# Load datasets
@st.cache_data
def load_data():
    umkm_df = pd.read_csv('data/umkm_data.csv')
    investor_df = pd.read_csv('data/investor_data.csv')
    user_data = pd.read_csv('data/users_data.csv')
    
    # Merge data
    umkm_df = umkm_df.merge(user_data[['user_id', 'lokasi_usaha', 'pertumbuhan_pendapatan']], on='user_id', how='left')
    investor_df = investor_df.merge(user_data[['user_id', 'lokasi_usaha']], on='user_id', how='left')
    
    return umkm_df, investor_df

umkm_df, investor_df = load_data()

# Preprocessing functions
def extract_provinsi(lokasi):
    if pd.isna(lokasi):
        return None
    try:
        provinsi = lokasi.split(',')[-1].strip()
        return provinsi.lower()
    except:
        return None

def clean_text(text):
    if pd.isna(text):
        return None
    return text.strip().lower()

# Apply preprocessing
umkm_df['provinsi'] = umkm_df['lokasi_usaha'].apply(extract_provinsi)
for col in ['kategori', 'model_bisnis', 'skala', 'jangkauan', 'provinsi']:
    umkm_df[col] = umkm_df[col].apply(clean_text)
for col in ['kategori', 'model_bisnis', 'skala', 'jangkauan', 'lokasi_usaha']:
    investor_df[col] = investor_df[col].apply(clean_text)

# Create vocabularies
kategori_vocab = list(pd.unique(umkm_df['kategori'].dropna()))
model_bisnis_vocab = list(pd.unique(umkm_df['model_bisnis'].dropna()))
skala_vocab = list(pd.unique(umkm_df['skala'].dropna()))
jangkauan_vocab = list(pd.unique(umkm_df['jangkauan'].dropna()))
lokasi_vocab = list(pd.unique(umkm_df['provinsi'].dropna()))

# Create StringLookup layers
kategori_lookup = tf.keras.layers.StringLookup(vocabulary=kategori_vocab, mask_token=None)
model_bisnis_lookup = tf.keras.layers.StringLookup(vocabulary=model_bisnis_vocab, mask_token=None)
skala_lookup = tf.keras.layers.StringLookup(vocabulary=skala_vocab, mask_token=None)
jangkauan_lookup = tf.keras.layers.StringLookup(vocabulary=jangkauan_vocab, mask_token=None)
lokasi_lookup = tf.keras.layers.StringLookup(vocabulary=lokasi_vocab, mask_token=None)

# Encode features
def encode_single_label(text_series, lookup_layer):
    return lookup_layer(text_series.fillna('').astype(str)).numpy()

umkm_encoded = {
    'kategori': encode_single_label(umkm_df['kategori'], kategori_lookup),
    'model_bisnis': encode_single_label(umkm_df['model_bisnis'], model_bisnis_lookup),
    'skala': encode_single_label(umkm_df['skala'], skala_lookup),
    'jangkauan': encode_single_label(umkm_df['jangkauan'], jangkauan_lookup),
    'lokasi_usaha': encode_single_label(umkm_df['provinsi'], lokasi_lookup),
}

investor_encoded = {
    'kategori': encode_single_label(investor_df['kategori'], kategori_lookup),
    'model_bisnis': encode_single_label(investor_df['model_bisnis'], model_bisnis_lookup),
    'skala': encode_single_label(investor_df['skala'], skala_lookup),
    'jangkauan': encode_single_label(investor_df['jangkauan'], jangkauan_lookup),
    'lokasi_usaha': encode_single_label(investor_df['lokasi_usaha'], lokasi_lookup),
}

# Define encoder models
def create_umkm_encoder():
    kategori_input = Input(shape=(), dtype=tf.int32, name='kategori')
    model_bisnis_input = Input(shape=(), dtype=tf.int32, name='model_bisnis')
    skala_input = Input(shape=(), dtype=tf.int32, name='skala')
    jangkauan_input = Input(shape=(), dtype=tf.int32, name='jangkauan')
    lokasi_input = Input(shape=(), dtype=tf.int32, name='provinsi')

    kategori_emb = Flatten()(Embedding(len(kategori_vocab) + 1, 16)(kategori_input))
    model_bisnis_emb = Flatten()(Embedding(len(model_bisnis_vocab) + 1, 16)(model_bisnis_input))
    skala_emb = Flatten()(Embedding(len(skala_vocab) + 1, 16)(skala_input))
    jangkauan_emb = Flatten()(Embedding(len(jangkauan_vocab) + 1, 16)(jangkauan_input))
    lokasi_emb = Flatten()(Embedding(len(lokasi_vocab) + 1, 16)(lokasi_input))

    features = Concatenate()([
        kategori_emb, model_bisnis_emb, skala_emb, jangkauan_emb, lokasi_emb
    ])

    x = Dense(128, activation='relu')(features)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(32, activation='relu', name='umkm_embedding')(x)

    return Model(inputs=[
        kategori_input, model_bisnis_input, skala_input, jangkauan_input, lokasi_input
    ], outputs=output, name='umkm_encoder')

def create_investor_encoder():
    kategori_input = Input(shape=(), dtype=tf.int32, name='kategori')
    model_bisnis_input = Input(shape=(), dtype=tf.int32, name='model_bisnis')
    skala_input = Input(shape=(), dtype=tf.int32, name='skala')
    jangkauan_input = Input(shape=(), dtype=tf.int32, name='jangkauan')
    lokasi_input = Input(shape=(), dtype=tf.int32, name='lokasi_usaha')

    kategori_emb = Flatten()(Embedding(len(kategori_vocab) + 1, 16)(kategori_input))
    model_bisnis_emb = Flatten()(Embedding(len(model_bisnis_vocab) + 1, 16)(model_bisnis_input))
    skala_emb = Flatten()(Embedding(len(skala_vocab) + 1, 16)(skala_input))
    jangkauan_emb = Flatten()(Embedding(len(jangkauan_vocab) + 1, 16)(jangkauan_input))
    lokasi_emb = Flatten()(Embedding(len(lokasi_vocab) + 1, 16)(lokasi_input))

    features = Concatenate()([
        kategori_emb, model_bisnis_emb, skala_emb, jangkauan_emb, lokasi_emb
    ])

    x = Dense(128, activation='relu')(features)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(32, activation='relu', name='investor_embedding')(x)

    return Model(inputs=[
        kategori_input, model_bisnis_input, skala_input, jangkauan_input, lokasi_input
    ], outputs=output, name='investor_encoder')

# Define recommendation model
class UMKMRecommendationModel(tfrs.Model):
    def __init__(self, umkm_encoder, investor_encoder, umkm_ds):
        super().__init__()
        self.umkm_encoder = umkm_encoder
        self.investor_encoder = investor_encoder
        def map_fn(x):
            umkm_features = [
                x['kategori'],
                x['model_bisnis'],
                x['skala'],
                x['jangkauan'],
                x['lokasi_usaha'],
            ]
            umkm_embedding = self.umkm_encoder(umkm_features)
            return umkm_embedding
        cached_umkm = umkm_ds.batch(1000).cache()
        self.umkm_candidates = cached_umkm.map(map_fn)
        self.retrieval_task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=self.umkm_candidates
            )
        )
    
    def call(self, features):
        investor_emb = self.investor_encoder([
            features['kategori'],
            features['model_bisnis'],
            features['skala'],
            features['jangkauan'],
            features['lokasi_usaha']
        ])
        umkm_emb = self.umkm_encoder([
            features['kategori'],
            features['model_bisnis'],
            features['skala'],
            features['jangkauan'],
            features['lokasi_usaha']
        ])
        return {
            'investor_embedding': investor_emb,
            'umkm_embedding': umkm_emb
        }
    
    def compute_loss(self, features, training=False):
        model_output = self(features)
        return self.retrieval_task(
            query_embeddings=model_output['investor_embedding'],
            candidate_embeddings=model_output['umkm_embedding']
        )

# Create datasets
def create_dataset():
    if len(umkm_df) == 0 or len(investor_df) == 0:
        raise ValueError("Empty dataset detected. Check umkm_df and investor_df.")
    umkm_dataset = tf.data.Dataset.from_tensor_slices({
        'umkm_id': umkm_df['umkm_id'].astype(str).values,
        'kategori': umkm_encoded['kategori'],
        'model_bisnis': umkm_encoded['model_bisnis'],
        'skala': umkm_encoded['skala'],
        'jangkauan': umkm_encoded['jangkauan'],
        'lokasi_usaha': umkm_encoded['lokasi_usaha'],
        'pertumbuhan_pendapatan': umkm_df['pertumbuhan_pendapatan'],
    })
    investor_dataset = tf.data.Dataset.from_tensor_slices({
        'investor_id': investor_df['investor_id'].astype(str).values,
        'kategori': investor_encoded['kategori'],
        'model_bisnis': investor_encoded['model_bisnis'],
        'skala': investor_encoded['skala'],
        'jangkauan': investor_encoded['jangkauan'],
        'lokasi_usaha': investor_encoded['lokasi_usaha'],
    })
    return umkm_dataset, investor_dataset

umkm_ds, investor_ds = create_dataset()

# Load model with weights
@st.cache_resource
def load_model():
    umkm_encoder = create_umkm_encoder()
    investor_encoder = create_investor_encoder()
    model = UMKMRecommendationModel(umkm_encoder, investor_encoder, umkm_ds)
    checkpoint_path = "checkpoints/model_best.weights.h5"
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        st.success(f"Loaded best weights from {checkpoint_path}")
    else:
        st.error(f"No model weights found at {checkpoint_path}!")
        return None
    return model

# Recommendation function
def get_recommendations(investor_id, model, top_k=5):
    try:
        investor_idx = investor_df.index[investor_df['investor_id'] == investor_id].tolist()
        if not investor_idx:
            st.error(f"Invalid investor_id: {investor_id}. Not found in investor_df.")
            return []
        investor_idx = investor_idx[0]
    except Exception as e:
        st.error(f"Error finding investor_id {investor_id}: {e}")
        return []

    investor_features = {
        'kategori': tf.constant([investor_encoded['kategori'][investor_idx]], dtype=tf.int32),
        'model_bisnis': tf.constant([investor_encoded['model_bisnis'][investor_idx]], dtype=tf.int32),
        'skala': tf.constant([investor_encoded['skala'][investor_idx]], dtype=tf.int32),
        'jangkauan': tf.constant([investor_encoded['jangkauan'][investor_idx]], dtype=tf.int32),
        'lokasi_usaha': tf.constant([investor_encoded['lokasi_usaha'][investor_idx]], dtype=tf.int32),
    }

    investor_emb = model.investor_encoder([
        investor_features['kategori'],
        investor_features['model_bisnis'],
        investor_features['skala'],
        investor_features['jangkauan'],
        investor_features['lokasi_usaha']
    ])

    similarities = []
    for i in range(len(umkm_df)):
        umkm_features = [
            tf.constant([umkm_encoded['kategori'][i]], dtype=tf.int32),
            tf.constant([umkm_encoded['model_bisnis'][i]], dtype=tf.int32),
            tf.constant([umkm_encoded['skala'][i]], dtype=tf.int32),
            tf.constant([umkm_encoded['jangkauan'][i]], dtype=tf.int32),
            tf.constant([umkm_encoded['lokasi_usaha'][i]], dtype=tf.int32),
        ]
        umkm_emb = model.umkm_encoder(umkm_features)
        similarity = -tf.keras.losses.cosine_similarity(investor_emb, umkm_emb).numpy()[0]
        similarities.append((i, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = similarities[:top_k]
    return [(umkm_df['umkm_id'].iloc[umkm_id], score) for umkm_id, score in top_recommendations]

# Streamlit UI
st.title("UMKM-Investor Recommendation System")
st.markdown("Select an investor ID to get UMKM recommendations based on their preferences.")

# Load model
model = load_model()
if model is None:
    st.stop()

# Investor selection
investor_ids = investor_df['investor_id'].tolist()
selected_investor = st.selectbox("Select Investor ID", investor_ids)

# Number of recommendations
top_k = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

# Get and display recommendations
if st.button("Get Recommendations"):
    recommendations = get_recommendations(selected_investor, model, top_k)
    
    if recommendations:
        st.subheader(f"Recommendations for Investor {selected_investor}")
        
        # Display investor preferences
        investor_idx = investor_df.index[investor_df['investor_id'] == selected_investor].tolist()[0]
        st.write("**Investor Preferences:**")
        st.write(f"- Location: {investor_df.iloc[investor_idx]['lokasi_usaha']}")
        st.write(f"- Category: {investor_df.iloc[investor_idx]['kategori']}")
        st.write(f"- Business Model: {investor_df.iloc[investor_idx]['model_bisnis']}")
        st.write(f"- Scale: {investor_df.iloc[investor_idx]['skala']}")
        st.write(f"- Market Reach: {investor_df.iloc[investor_idx]['jangkauan']}")
        
        # Display recommendations
        st.subheader(f"Top {top_k} Recommended UMKMs")
        for rank, (umkm_id, score) in enumerate(recommendations, 1):
            umkm_idx = umkm_df.index[umkm_df['umkm_id'] == umkm_id].tolist()[0]
            umkm_row = umkm_df.iloc[umkm_idx]
            with st.expander(f"Rank {rank}: UMKM {umkm_id} (Score: {score:.3f})"):
                st.write(f"**Category**: {umkm_row['kategori']}")
                st.write(f"**Location**: {umkm_row['lokasi_usaha']}")
                st.write(f"**Business Model**: {umkm_row['model_bisnis']}")
                st.write(f"**Scale**: {umkm_row['skala']}")
                st.write(f"**Market Reach**: {umkm_row['jangkauan']}")
                st.write(f"**Revenue Growth**: {umkm_row['pertumbuhan_pendapatan']:.2f}%")
    else:
        st.warning("No recommendations available. Please check the investor ID or model.")

# Instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Select an investor ID from the dropdown.
2. Choose the number of recommendations using the slider.
3. Click 'Get Recommendations' to view the top UMKM matches.
4. Expand each recommendation to see detailed UMKM information.
""")

# Load model
model = load_model()

# User selects investor ID
investor_ids = investor_df['investor_id'].dropna().unique().tolist()
selected_investor_id = st.selectbox("Select Investor ID", investor_ids)

# Number of recommendations
top_k = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

# Button to trigger recommendations
if st.button("Get Recommendations"):
    if model is not None:
        with st.spinner("Computing recommendations..."):
            recommendations = get_recommendations(selected_investor_id, model, top_k=top_k)

            if recommendations:
                st.success("Recommendations retrieved successfully!")
                st.subheader("Top Recommended UMKMs:")
                for idx, (umkm_id, score) in enumerate(recommendations, start=1):
                    umkm_info = umkm_df[umkm_df['umkm_id'] == umkm_id].iloc[0]
                    st.markdown(f"**{idx}. UMKM ID: {umkm_id}**  \n"
                                f"Kategori: {umkm_info['kategori']}  \n"
                                f"Model Bisnis: {umkm_info['model_bisnis']}  \n"
                                f"Skala: {umkm_info['skala']}  \n"
                                f"Jangkauan: {umkm_info['jangkauan']}  \n"
                                f"Provinsi: {umkm_info['provinsi']}  \n"
                                f"Pertumbuhan Pendapatan: {umkm_info['pertumbuhan_pendapatan']}  \n"
                                f"Skor Kecocokan: {score:.4f}")
            else:
                st.warning("No recommendations could be generated.")
    else:
        st.error("Model is not loaded. Please check checkpoint path or model definition.")
