"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import faiss
from langchain.llms import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import Cohere
from langchain.llms import HuggingFaceHub
from langchain.chains import VectorDBQAWithSourcesChain
import pickle
import json
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

load_dotenv()
# prompt = PromptTemplate(
#     input_variables=["page_content","source_main",'source_sub'],
#     template='Content: {page_content}\nProject: {source_main}\nSource: {source_sub}', template_format='f-string'
#     )

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = "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"
st.set_page_config(page_title="Blendle Notion QA Bot", page_icon=":robot:")
st.header("Blendle Notion QA Bot")

# READ JSON
with open("information.json", "r") as f:
    data = json.load(f)

main_folders=[None]
main_folders_og=[None]
for DATA in data:
    folder = DATA['folder']
    if folder.count(" ")>=1:
        main_folders.append(" ".join(folder.split(" ")[0:-1]))
        main_folders_og.append(folder)
    else:
        main_folders.append(folder)
        main_folders_og.append(folder)

# st.sidebar.title("Select a Project")
# proj_seletection = st.sidebar.selectbox("Choose a Project", main_folders)
proj_seletection = main_folders[1]
st.write("PROJECT: ",proj_seletection)

if proj_seletection:
    index_proj = main_folders.index(proj_seletection)
    og_main_folder = main_folders_og[index_proj]

    for DATA in data:
        folder = DATA['folder']
        if folder == og_main_folder:
            index_file_name = DATA['index']
            faiss_store_name = DATA['faiss_store']
            break

    # Load the LangChain.
    index = faiss.read_index(index_file_name.replace("indexes/",""))
    with open(faiss_store_name, "rb") as f:
        store = pickle.load(f)

    #st.write(store)
    store.index = index

    # model_id = "gpt2"
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id)
    # pipe = pipeline(
    #     "text-generation", model=model, tokenizer=tokenizer,max_length=100
    # )
    # hf = HuggingFacePipeline(pipeline=pipe)

    llm = OpenAI(temperature=0,openai_api_key=OPENAI_API_KEY)
    chain = VectorDBQAWithSourcesChain.from_llm(llm=llm, vectorstore=store)

    # st.write(chain)
# # # From here down is all the StreamLit UI.

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []


    def get_text():
        input_text = st.text_input("You: ", "Hello, how are you?", key="input")
        return input_text


    user_input = get_text()
    send_message = st.button("▶")

    if user_input and send_message:
        result = chain({"question": user_input})
        output = f"Answer: {result['answer']}\nSources: {result['sources']}"

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state["generated"]:

        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
