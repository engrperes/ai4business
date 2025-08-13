import os
import uuid
import tempfile
import pandas as pd
import pytesseract
from PIL import Image
import docx
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from pypdf.errors import PdfReadError
from openai.error import AuthenticationError, InvalidRequestError

st.set_page_config(page_title="Q&A com IA", page_icon="🔎", layout="wide")
st.subheader("Q&A com IA - PLN usando LangChain")

file_input = st.file_uploader("Upload de arquivo", type=['pdf', 'txt', 'csv', 'docx', 'jpeg', 'png', 'jpg'])
openaikey_input = st.text_input("OpenAI API Key (opcional se já estiver em Secrets)", type='password')
prompt = st.text_area("Digite sua pergunta", height=160)
run_button = st.button("Rodar!")

select_k = st.slider("Número de trechos relevantes (k)", min_value=1, max_value=5, value=2)
select_chain_type = st.radio("Tipo de cadeia (chain)", ['stuff', 'map_reduce', "refine", "map_rerank"])

def get_openai_api_key():
    # Prioriza secrets; se não houver, usa input manual
    key = st.secrets.get("OPENAI_API_KEY", None)
    if not key and openaikey_input:
        key = openaikey_input.strip()
    return key

def load_document(file_path, file_type):
    if file_type == 'application/pdf':
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif file_type == 'text/plain':
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()
    elif file_type == 'text/csv':
        df = pd.read_csv(file_path)
        return [{"page_content": df.to_string()}]
    elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text:
                full_text.append(para.text)
        return [{"page_content": "\n".join(full_text)}]
    elif file_type in ['image/jpeg', 'image/png', 'image/jpg']:
        text = pytesseract.image_to_string(Image.open(file_path))
        return [{"page_content": text}]
    else:
        st.error("Tipo de arquivo não suportado.")
        return None

def qa(file_path, file_type, query, chain_type, k, api_key):
    try:
        documents = load_document(file_path, file_type)
        if not documents:
            return None

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        os.environ["OPENAI_API_KEY"] = api_key

        embeddings = OpenAIEmbeddings()
        persist_dir = os.path.join(tempfile.gettempdir(), "chroma_" + str(uuid.uuid4()))
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_dir)

        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4", temperature=0),
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            verbose=False,
        )
        result = qa_chain({"query": query})
        return result

    except PdfReadError as e:
        st.error(f"Erro na leitura do PDF: {e}")
        return None
    except AuthenticationError as e:
        st.error(f"Erro de autenticação: {e}")
        return None
    except InvalidRequestError as e:
        st.error(f"Erro de solicitação inválida: {e}")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado: {e}")
        return None

def display_result(result):
    if result:
        st.markdown("### Resultado:")
        st.write(result.get("result", ""))
        st.markdown("### Trechos de origem relevantes:")
        for doc in result.get("source_documents", []):
            st.markdown("---")
            try:
                st.markdown(doc.page_content)
            except Exception:
                st.write(doc)

if run_button:
    api_key = get_openai_api_key()
    if not file_input:
        st.warning("Envie um arquivo.")
    elif not api_key:
        st.warning("Informe sua OpenAI API Key (via Secrets ou campo).")
    elif not prompt or not prompt.strip():
        st.warning("Digite uma pergunta.")
    else:
        with st.spinner("Executando Q&A..."):
            temp_file_path = os.path.join(tempfile.gettempdir(), file_input.name)
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_input.read())

            # Verifica a API Key com um teste simples de embeddings
            try:
                os.environ["OPENAI_API_KEY"] = api_key
                _ = OpenAIEmbeddings().embed_documents(["teste"])
            except AuthenticationError as e:
                st.error(f"OpenAI API Key inválida: {e}")
            else:
                result = qa(temp_file_path, file_input.type, prompt, select_chain_type, select_k, api_key)
                display_result(result)
