"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.llms import OpenAI

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    from langchain.prompts.prompt import PromptTemplate
    from langchain.memory import ConversationBufferMemory

    llm = OpenAI(temperature=0)

    # Define the mental health bot prompt
    template = """Act as a mental health chatbot that communicates with users on WhatsApp, asking them relevant questions one at a time, similar to how a psychologist would. Engage in a conversation by asking appropriate questions to understand their mental state, emotions, and concerns, while providing support and empathy.

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
st.header("Farmako Mental Health Bot")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

# Create a placeholder for the input box
input_box = st.empty()

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

def get_text():
    input_text = input_box.text_input("You: ", "", key="input")
    return input_text

user_input = get_text()

if user_input:
    output = chain.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
