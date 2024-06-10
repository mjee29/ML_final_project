import streamlit as st
import pandas as pd
import gensim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 데이터 로드 및 전처리
def load_data():
    data = pd.read_csv(r'C:\Users\mjkim\ML_final_project\최종데이터.csv', encoding="latin1")
    data_cleaned = data.drop(columns=['Unnamed: 0'])
    data_cleaned['Notes'] = data_cleaned['Notes'].apply(eval)
    data_cleaned['category'] = data_cleaned['category'].str.split(',')

    # category에 가중치 부여
    category_weight = 2.0
    data_cleaned['features'] = data_cleaned.apply(
        lambda row: row['Notes'] + [cat * int(category_weight) for cat in row['category']], axis=1
    )
    data_cleaned['features_str'] = data_cleaned['features'].apply(lambda x: ' '.join(x))

    return data_cleaned

# Word2Vec 모델 훈련 및 문서 벡터 계산
def train_word2vec_model(data):
    notes = data['Notes'].tolist()
    model = gensim.models.Word2Vec(notes, vector_size=100, window=5, min_count=1, workers=4)
    doc_vectors = np.array([
        compute_doc_vector_word2vec(data['Notes'][i] + data['category'][i], model)
        for i in range(len(data))
    ])
    return model, doc_vectors

# 문서 벡터 계산 함수 (TF-IDF 가중치 없이)
def compute_doc_vector_word2vec(words, model):
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# 추천 함수 (Word2Vec 사용)
def recommend_perfumes_word2vec_only(perfume_name, data, similarity_matrix, top_n=5):
    try:
        idx = data.index[data['Name'] == perfume_name].tolist()[0]
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        perfume_indices = [i[0] for i in sim_scores]
        recommended_perfumes = data['Name'].iloc[perfume_indices]
        scores = [sim[1] for sim in sim_scores]
        recommendations = list(zip(recommended_perfumes, scores))
        return recommendations
    except IndexError:
        return []

def run_recommendation():
    st.header("Recommend by Perfume Name")
    perfume_name = st.text_input("Enter a perfume name:").strip()

    if st.button("Recommend by Name"):
        data_cleaned = load_data()
        model, doc_vectors = train_word2vec_model(data_cleaned)
        word2vec_similarity_matrix = cosine_similarity(doc_vectors)

        if perfume_name.lower() in data_cleaned['Name'].str.lower().values:
            recommendations_word2vec_only = recommend_perfumes_word2vec_only(perfume_name, data_cleaned, word2vec_similarity_matrix)
            if recommendations_word2vec_only:
                recommended_df = pd.DataFrame(recommendations_word2vec_only, columns=["Name", "Similarity"])
                st.write("\nTop 5 recommended perfumes:")
                st.dataframe(recommended_df)
            else:
                st.write("No similar perfumes found.")
        else:
            st.write("Perfume not found in the database.")

if __name__ == "__main__":
    run_recommendation()
