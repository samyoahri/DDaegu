import os
import streamlit as st
from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    LLMPredictor,
    load_index_from_storage,
)
from langchain.chat_models import ChatOpenAI
from PIL import Image
import base64
from pyparsing import empty
index_name = "./saved_index"
documents_folder = "./documents"


@st.cache_resource
def initialize_index(index_name, documents_folder):
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    )
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    if os.path.exists(index_name):
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name),
            service_context=service_context,
        )
    else:
        documents = SimpleDirectoryReader(documents_folder).load_data()
        index = GPTVectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
        index.storage_context.persist(persist_dir=index_name)

    return index


@st.cache_data(max_entries=200, persist=True)
def query_index(_index, query_text):
    if _index is None:
        return "Please initialize the index!"
    response = _index.as_query_engine().query(query_text)
    return str(response)



# GIF 파일 경로
gif_path = "./logo1_unscreen.gif"

file_ = open(gif_path, "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

# 이미지를 원형으로 자르기 위한 CSS 스타일
css_style = """
<style>
    img {
        border-radius: 100%;

    }
</style>
"""

con1,con2 = st.columns([1, 4])
with con1:
    # st.markdown으로 이미지 표시
    st.markdown(
        f'{css_style}<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )            

with con2:
    st.title("DDaegu")


st.header("때구 챗봇에 오신 것을 환영합니다")
st.write(
    "때구에게 맛집을 추천받아보세요"
)




index = None
# api_key = st.text_input("Enter your OpenAI API key here:", type="password")
api_key_file_path = "./api_key.txt"
with open(api_key_file_path, "r") as file:
    api_key = file.read().strip()
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    index = initialize_index(index_name, documents_folder)


if index is None:
    st.warning("Please enter your api key first.")

text = st.text_input("질문:", value="대구 북구 맛집 추천해줘")

if st.button("질문하기") and text is not None:
    response = query_index(index, text)
    st.markdown(response)

    llm_col, embed_col = st.columns(2)
    with llm_col:
        st.markdown(
            f"LLM Tokens Used: {index.service_context.llm_predictor._last_token_usage}"
        )

    with embed_col:
        st.markdown(
            f"Embedding Tokens Used: {index.service_context.embed_model._last_token_usage}"
        )
