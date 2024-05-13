import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from langchain.llms import CTransformers
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Función para dividir el texto en trozos y convertirlo en documentos
def chunks_and_document(txt):
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts]
    return docs
    
# Cargar el modelo LLM
def load_llm():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = CTransformers(
        model=r"llama-2-7b-chat.ggmlv3.q2_K.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm
 
# Aplicar el modelo LLM a nuestros documentos
def chains_and_response(docs):
    llm = load_llm()
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)
    
# Título de la página
st.set_page_config(page_title='🦜🔗 Text Summarization App', layout="wide")
st.title('🦜🔗 Text Summarization App')

# Obtener preferencias del usuario (tema)
theme = st.sidebar.radio("Seleccionar tema:", ("Claro", "Oscuro"))

# Aplicar preferencias del usuario
if theme == "Claro":
    st.write("Estás usando el tema claro")
    # Configurar colores, fuentes, etc. para el tema claro
else:
    st.write("Estás usando el tema oscuro")
    # Configurar colores, fuentes, etc. para el tema oscuro

# Text input
txt_input = st.text_area('Enter your text', '', height=200)

# Widget de selección para elegir el tamaño de la fuente
font_size = st.sidebar.selectbox("Seleccionar tamaño de la fuente", [10, 12, 14, 16, 18, 20])

# Formulario para aceptar la entrada de texto del usuario para la sumarización
result = []
with st.form('summarize_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('Calculating...'):
            docs = chunks_and_document(txt_input)
            response = chains_and_response(docs)
            result.append(response)

if len(result):
    st.title('📝✅ Summarization Result')
    # Aplicar CSS para cambiar el tamaño de la fuente del texto del resumen
    st.markdown(
        f"""
        <style>
            /* Ajustar el tamaño de la fuente del resumen según la selección del usuario */
            .resumen {{
                font-size: {font_size}px;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div class='resumen'>" + response + "</div>", unsafe_allow_html=True)
