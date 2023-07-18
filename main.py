import streamlit as st
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    from langchain.prompts.prompt import PromptTemplate
    from langchain.memory import ConversationBufferMemory

    llm = OpenAI(temperature=0)

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

st.set_page_config(page_title="Beta Version", page_icon=":robot:")
st.header("Farmako Medical History Chat")

if 'medical_history' not in st.session_state:
    st.session_state['medical_history'] = ''

medical_history = st.text_area("Medical History: ", value=st.session_state['medical_history'], key="medical_history")

if st.button("Save Medical History", key="save_medical_history"):
    st.session_state.medical_history = medical_history
    st.success('Medical history saved!')

user_input = st.text_input("You: ", "", key="user_input")

if user_input and st.session_state['medical_history']:
    combined_input = f"Medical history: {st.session_state['medical_history']}. {user_input}"
    output = chain.run(input=combined_input)
    st.write(output)
