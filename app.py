# app.py
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="상품설명 도우미", layout="wide")
st.title("📑 상품설명 도우미 (영업점용)")

# 1) OpenAI API 키: (환경변수에 없으면 입력받아 세션에 설정)
env_key = os.environ.get("OPENAI_API_KEY")
if not env_key:
    st.info("OpenAI API 키가 필요합니다. (환경변수 OPENAI_API_KEY 또는 아래 입력)")
    key_input = st.text_input("OpenAI API Key (sk-...)", type="password")
    if key_input:
        os.environ["OPENAI_API_KEY"] = key_input
else:
    st.write("🔒 OpenAI API Key detected in environment.")

# 모델 선택 (gpt-4o 접근권한 없다면 'gpt-3.5-turbo'로 바꿔 사용)
model_name = st.selectbox("모델 선택 (테스트용)", ["gpt-4o", "gpt-3.5-turbo"])

# PDF 업로드 (여러 개 가능)
uploaded_files = st.file_uploader("상품설명서 PDF 업로드 (여러 개 가능)", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    st.info("PDF를 로컬에 저장하고 문서를 인덱싱합니다. 조금 시간이 걸릴 수 있어요.")
    os.makedirs("docs", exist_ok=True)
    all_docs = []

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for uploaded_file in uploaded_files:
        dest_path = os.path.join("docs", uploaded_file.name)
        # 업로드 파일을 로컬에 저장
        with open(dest_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # PDF 로드 및 분리
        loader = PyPDFLoader(dest_path)
        raw_docs = loader.load()
        docs = splitter.split_documents(raw_docs)

        # 문단마다 출처(metadata) 설정 (파일명 + 페이지)
        for d in docs:
            page = d.metadata.get("page")  # PyPDFLoader는 종종 'page'를 넣어줌
            filename = uploaded_file.name
            if page is not None:
                d.metadata["source"] = f"{filename} (p.{int(page)+1})"
            else:
                d.metadata["source"] = filename
        all_docs.extend(docs)

    # 임베딩 및 벡터 DB (FAISS)
    embeddings = OpenAIEmbeddings()  # OPENAI_API_KEY 환경변수 필요
    vectorstore = FAISS.from_documents(all_docs, embeddings)

    # Retriever + LLM + RetrievalQA
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    st.success("인덱싱 완료! 질의 입력하세요.")
    query = st.text_input("궁금한 점을 물어보세요 👇")

    if query:
        with st.spinner("검색·응답 생성 중..."):
            result = qa({"query": query})
            # langchain 버전에 따라 반환 키가 달라서 안전하게 처리
            answer = result.get("result") or result.get("answer") or result.get("output_text") or str(result)
            st.markdown("### 💬 AI 답변")
            st.write(answer)

            source_docs = result.get("source_documents") or []
            if source_docs:
                st.markdown("### 📖 출처 (문단 일부 + 파일명/페이지)")
                for i, doc in enumerate(source_docs, 1):
                    src = doc.metadata.get("source", "unknown")
                    snippet = doc.page_content.strip().replace("\n", " ")[:300]
                    st.write(f"{i}. ({src}) {snippet}...")
            else:
                st.write("출처 문단을 찾지 못했습니다.")

