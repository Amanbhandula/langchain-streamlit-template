import random
from langchain.llms import OpenAI


def load_history_chain():
    """Logic for loading the chain you want to use should go here."""
    from langchain.prompts.prompt import PromptTemplate
    from langchain.memory import ConversationBufferMemory

    llm = OpenAI(temperature=0)

    # Define the medical history generator prompt
    template = """
    You are an AI system that generates random medical history data. Please generate a comprehensive medical history JSON for a random patient.
    """
    PROMPT = PromptTemplate(
        input_variables=[], template=template
    )

    chain = ConversationChain(
        prompt=PROMPT,
        llm=llm, 
        verbose=True, 
        memory=ConversationBufferMemory(ai_prefix="AI System")
    )

    return chain

history_chain = load_history_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Beta Version", page_icon=":robot:")
st.header("Farmako Medical History Chat")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

if "medical_history" not in st.session_state:
    st.session_state["medical_history"] = []

def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

def get_medical_history():
    if not st.session_state["medical_history"]:
        # Generate a new medical history
        output = history_chain.run(input="")
        st.session_state["medical_history"].append(json.loads(output))

    medical_history = st.session_state["medical_history"][-1]
    medical_history = json.dumps(medical_history, indent=4)  # For pretty printing
    return medical_history

user_input = get_text()
medical_history = get_medical_history()

st.text_area("Medical History: ", value=medical_history, key="medical_history")

if user_input and medical_history:
    combined_input = f"Medical history: {medical_history}. {user_input}"
    output = chain.run(input=combined_input)

    st.session_state.past.append(combined_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
