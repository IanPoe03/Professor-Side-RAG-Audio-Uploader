import streamlit as st
import openai
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.callbacks import get_openai_callback
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from openai import OpenAI
from langchain_text_splitters import CharacterTextSplitter


st.title('Professor-Side Audio Uploader')
st.write("This tool allows you to upload an audio file (currently set to .wav) which is then transcribed using Whisper and embedded so that you can make queries about the contents of the audio file")


# Initialize OpenAI client with your API key
def transcribe_audio(file):
    try:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            api_key = OPENAI_API_KEY,
            file=file
        )
        return response.text
    except Exception as e:
        return str(e)

def ingest(file_text):
    chunker = CharacterTextSplitter(chunk_size = 512, chunk_overlap = 128)
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    chunked_contents = chunker.split_text(file_text)
    vectordb = Chroma.from_texts(texts = chunked_contents, embedding = embeddings)
    return vectordb
def count_tokens(chain, query):
    with get_openai_callback() as cb:
        # Assuming `context` is accepted as a keyword argument; adjust according to your actual API's specification
        result = chain.invoke(input=query, chat_history=[msg['content'] for msg in st.session_state.messages if 'content' in msg], question = query)
        print(f'Spent {cb.total_tokens} tokens')
    #latest_response = result['result']
    return result['result']



with st.sidebar:
    OPENAI_API_KEY = st.text_input("Enter OpenAI API Key", key="api_key", type="password")
    if OPENAI_API_KEY:
        st.write("Successfully Applied OpenAI API Key")
    client = OpenAI(openai_api_key = OPENAI_API_KEY)

with st.sidebar:
       # File uploader allows user to add their own audio file
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
    
    if uploaded_file is not None:
        # Display the file details
        st.write("Filename:", uploaded_file.name)
        
        if 'text' not in st.session_state:
            # Process the file with Whisper
            if st.button("Transcribe Audio"):
                # Show a message while processing
                with st.spinner('Transcribing audio...'):
                    text = transcribe_audio(uploaded_file)
                    st.write(text)
                    st.write('transcribed')
                    vectordb = ingest(text)
                    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4-turbo", verbose=False, openai_api_key = OPENAI_API_KEY)
                    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
                    retriever = vectordb.as_retriever(search_type = "similarity", search_kwargs={"k": 3})
                    if 'conversation_buf' not in st.session_state:
                        st.session_state.conversation_buf = RetrievalQA.from_chain_type(
                            llm=llm, 
                            memory = ConversationBufferMemory(),
                            chain_type='stuff', 
                            retriever = retriever, 
                            verbose = False
                        )
                st.success("Transcription + Embedding Completed")
            # st.text_area("Transcription:", value=text, height=300)
            else:
                st.write("Click the button to transcribe.")



# Set up chat interface - conversation between an assistant and a user
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi! I can answer questions about audio files you upload"}]

# Display stored conversation
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input():
    if not OPENAI_API_KEY:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    answer = count_tokens(st.session_state.conversation_buf, prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
