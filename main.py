import streamlit as st

st.title("Perfume Recommendation System")

# 사이드바에 기능 선택 메뉴 추가
st.sidebar.header("Choose Recommendation Type")
option = st.sidebar.radio("Select a recommendation type:", ["Recommend by Name", "Recommend by Category"])

# 선택된 옵션에 따라 다른 파일의 함수 호출
if option == "Recommend by Name":
    import recommend_name
    recommend_name.run_recommendation()
elif option == "Recommend by Category":
    import recommend_category
    recommend_category.run_recommendation()
