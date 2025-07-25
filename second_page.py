import streamlit as st
from agent import get_final_answer
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.tracers.context import collect_runs
from langsmith import Client
from streamlit_feedback import streamlit_feedback
import uuid

client = Client()

def second_page():
    st.header("AI504 Chatbot")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Go to Home", key="home_page"):
            st.session_state.pop("student_id", None)
            st.session_state.pop("chat_history", None)
            st.session_state.pop("dialog_identifier", None)
            st.rerun()

    with col2:
        if st.button("Refresh", key="refresh"):
            st.session_state.pop("chat_history", None)
            st.session_state.pop("dialog_identifier", None)
            st.rerun()


    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "dialog_identifier" not in st.session_state:
        st.session_state.dialog_identifier = uuid.uuid4()


    for message in st.session_state.chat_history:
        if isinstance(message,AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        else:
            with st.chat_message("Human"):
                st.write(message.content)


    def get_response(user_input):
        response, reasoning = get_final_answer(user_input, st.session_state.chat_history)
        return response, reasoning

    if user_input := st.chat_input("Type your message here..."):
        st.chat_message("Human").write(f"{user_input}")

        with collect_runs() as cb:
            with st.spinner("Thinking..."):
                response, reasoning = get_response(user_input)
                st.chat_message("AI").write(f"{response}")
                
                with st.expander("Reasoning"):
                    st.json(reasoning)

                st.session_state.chat_history.append(HumanMessage(content=user_input))
                st.session_state.chat_history.append(AIMessage(content=response))
            st.session_state.run_id = cb.traced_runs[0].id


    feedback_option = "thumbs"
    if st.session_state.get("run_id"):
        run_id = st.session_state.run_id
        feedback = streamlit_feedback(
            feedback_type = "thumbs",
            optional_text_label ="[Optional] Please provide an explanation",
            key=f"feedback_{run_id}",
        )

        score_mappings = {
            "thumbs": {"👍": 1, "👎": -1},
            "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
        }

        scores = score_mappings[feedback_option]

        if feedback:
            score = scores.get(feedback["score"])

            if score is not None:
                feedback_type_str = f"{feedback_option} {feedback['score']}"

                feedback_record = client.create_feedback(
                    run_id,
                    feedback_type_str,
                    score = score,
                    comment=feedback.get("text"),
                )
                st.session_state.feedback = {
                    "feedback_id": str(feedback_record.id),
                    "score": score,
                }
            else:
                st.warning("Invalid feedback score.")