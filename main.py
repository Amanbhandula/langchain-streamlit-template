"""Python file to serve as the frontend"""
import streamlit as st
import json
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.llms import OpenAI

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    from langchain.prompts.prompt import PromptTemplate
    from langchain.memory import ConversationBufferMemory

    llm = OpenAI(temperature=0)

    # Define the medical chatbot prompt
    template = """
    You are an advanced healthcare assistant. Your function is to provide support for doctors by offering appropriate responses to patient inquiries based on their comprehensive medical history.
    
    Current conversation:
    {history}
    Human: {input}
    AI Assistant:"""
    PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=template
    )

    chain = ConversationChain(
        prompt=PROMPT,
        llm=llm, 
        verbose=True, 
        memory=ConversationBufferMemory(ai_prefix="AI Assistant")
    )

    return chain

chain = load_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Beta Version", page_icon=":robot:")
st.header("Farmako Medical History Chat")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

def get_medical_history():
    medical_history = st.text_area("Medical History: ", "", key="medical_history")
    return medical_history

user_input = get_text()
medical_history = get_medical_history()

if user_input and medical_history:
    combined_input = f"Medical history: {medical_history}. {user_input}"
    output = chain.run(input=combined_input)

    st.session_state.past.append(combined_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
