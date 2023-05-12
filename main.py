"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate



# Now we can override it and set it to "AI Assistant"

template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI Assistant:"""
PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=template
)
conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm, 
    verbose=True, 
    memory=ConversationBufferMemory(ai_prefix="AI Assistant")
)



def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0)
    chain = ConversationChain(llm=llm)
    return chain

chain = load_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Farmako Mental Health Bot (Beta)", page_icon=":robot:")
st.header("Farmako AI Mental Health Bot")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = chain.run(input=user_input, prompt = "Act as a mental health chatbot that communicates with users on WhatsApp, asking them relevant questions one at a time, similar to how a psychologist would. Engage in a conversation by asking appropriate questions to understand their mental state, emotions, and concerns, while providing support and empathy.")

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
