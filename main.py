"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import json

from langchain.chains import ConversationChain
from langchain.llms import OpenAI

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    from langchain.prompts.prompt import PromptTemplate
    from langchain.memory import ConversationBufferMemory

    llm = OpenAI(temperature=0)

    template = """Act as an advanced healthcare assistant that communicates with users based on their comprehensive medical history. This includes offering advice and recommendations on over-the-counter medications, tests, consultations with specialist doctors, or possible nutritional supplements if needed. Your responses are edited and approved by doctors before being sent to patients.

    Medical history:
    {medical_history}
    
    Human: {input}
    AI Assistant:"""
    
    PROMPT = PromptTemplate(
        input_variables=["medical_history", "input"], template=template
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
st.header("Farmako Chat with your Medical History")

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
    output = chain.run(input={"medical_history": medical_history, "input": user_input})

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
