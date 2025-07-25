
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from datetime import datetime
from syllabus import SYLLABUS
from learned_contents import PREVIOUS_LEARNED_CONTENTS

# --- Load Vector Store & Retriever ---
def get_vector_store():
    """FAISS 벡터 스토어를 로드합니다."""
    return FAISS.load_local(
        "./faiss_db/",
        embeddings=OpenAIEmbeddings(model="text-embedding-3-large"),
        allow_dangerous_deserialization=True
    )

vector_store = get_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# --- Shared QA Prompt Template ---
qa_prompt_template_str = f"Today's date is {datetime.now().strftime('%Y-%m-%d')}.\n" + """You are a helpful python programming assistant. Answer the user's question based on the chat history and provided context.

Your explanation must be within a very narrow scope of what the student has learned according to the syllabus and previous learned contents.
For example, if the student learned the format function, only explain the format function, not f-strings.
If they learned lists and for loops separately, do not use list comprehensions.
If they learned pd.DataFrame to make DataFrame using list input, you can only use list input, never use dictionary input.
Never use content or explanation that is not in the lecture materials and previous learned contents.

When responding to student questions, you must refer to the retrieved contexts if they are provided.

[Syllabus]
{syllabus}

[Previous Learned Contents]
{previous_learned_contents}

[Context]
{context}

[Chat History]
{chat_history}

[User's Question]
{input}

Provide your answer in Korean.
"""

# --- Chains ---

# 1. Planner (Router)
planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
routing_prompt_template = """You are a router & planner for a QA chatbot. You need to decide the next plan based on the history and user's new chat.

You MUST choose one of the following categories:

1. retrieval_qa: Questions that require reference to the provided lecture materials.
   - Arguments: "question", "retrieval_query"
   - "question" should be the user's question, rephrased to be self-contained based on the chat history.
   - "retrieval_query" should be a concise query for finding relevant lecture materials.
   - Example: {{"category": "retrieval_qa", "arguments": {{"question": "What is bubble sort?", "retrieval_query": "bubble sort"}}}}

2. qa: Follow-up questions where previous retrieval is sufficient (e.g., asking for a simpler explanation of the last answer).
   - Arguments: "question"
   - "question" should be a self-contained question based on the history.
   - Example: {{"category": "qa", "arguments": {{"question": "Could you explain the previous topic about bubble sort more easily?"}}}}

3. off_topic: Questions completely unrelated to programming or the lecture.
   - No arguments needed.
   - Example: {{"category": "off_topic", "arguments": {{}}}}

Analyze the chat history and the user's new chat, then return a JSON object with the appropriate category and arguments.

[Chat History]
{chat_history}

[User's New Chat]
"{input}"
"""
ROUTING_PROMPT = PromptTemplate.from_template(routing_prompt_template)
router = ROUTING_PROMPT | planner_llm | JsonOutputParser()

# 2. Executor Chains
llm_for_qa = ChatOpenAI(model="o3-2025-04-16", temperature=1)

# 2a. retrieval_qa chain
doc_prompt = ChatPromptTemplate.from_template(qa_prompt_template_str)
document_chain = create_stuff_documents_chain(llm_for_qa, doc_prompt)

retrieval_qa_runnable = RunnablePassthrough.assign(
    context=lambda x: retriever.invoke(x['topic']['arguments']['retrieval_query']),
    input=lambda x: x['topic']['arguments']['question'],
    chat_history=lambda x: x['chat_history'],
    syllabus=lambda x: SYLLABUS,
    previous_learned_contents=lambda x: PREVIOUS_LEARNED_CONTENTS
) | document_chain

# 2b. qa (general) chain
general_prompt = ChatPromptTemplate.from_template(qa_prompt_template_str)
general_chain = general_prompt | llm_for_qa | StrOutputParser()

qa_runnable = RunnablePassthrough.assign(
    input=lambda x: x['topic']['arguments']['question'],
    chat_history=lambda x: x['chat_history'],
    context=lambda x: "No context provided", # qa의 경우 context가 없음을 명시
    syllabus=lambda x: SYLLABUS,
    previous_learned_contents=lambda x: PREVIOUS_LEARNED_CONTENTS
) | general_chain


# 2c. off_topic 응답
def off_topic_responder(_):
    return {"answer": "해당 질문에 답할 수 없습니다. 수업에 관련된 질문만 해주세요."}

# 3. Branch to select the executor
branch = RunnableBranch(
    (lambda x: x['topic']['category'] == "retrieval_qa", retrieval_qa_runnable),
    (lambda x: x['topic']['category'] == "qa", qa_runnable),
    off_topic_responder
)

# 4. Full Executor Chain (Planner -> Branch)
full_executor_chain = {
    "topic": router,
    "input": lambda x: x["input"],
    "chat_history": lambda x: x["chat_history"]
} | branch


# 5. Reflector: 생성된 답변을 검증하고 수정합니다.
reflector_llm = ChatOpenAI(model="gpt-4o", temperature=0)
reflection_prompt_template = """You are a senior teaching assistant reviewing an answer from a junior TA.
Your task is to determine if the answer is appropriate based on the criteria below. If not, you must provide a revised, better answer.

[Evaluation Criteria]
1. Adherence to Lecture Scope: The answer must not go beyond the scope of the provided lecture materials and the student's learning history. It should strictly use concepts the student has already learned. For example, if the student learned only the format() function, do not use f-strings. If they learned only basic for-loops, do not use list comprehensions.
2. Clarity and Sufficiency: The explanation must be clear, easy to understand, and detailed enough to answer the question.

[Syllabus]
{syllabus}

[Previous Learned Contents]
{previous_learned_contents}

[Chat History]
{chat_history}

[Original Question]
{question}

[Junior TA's Initial Answer]
{initial_answer}

Now, review the answer.
- Does the answer meet all criteria? (Yes/No)
- If 'Yes', just repeat the initial answer.
- If 'No', provide a new, corrected answer that meets all criteria.

Your final output must be ONLY the answer itself, either the repeated initial one or your new one.
"""
REFLECTION_PROMPT = PromptTemplate.from_template(reflection_prompt_template)
reflection_chain = REFLECTION_PROMPT | reflector_llm | StrOutputParser()


# --- Main Agent Logic ---
def get_final_answer(user_input, chat_history):
    # 1. Planner를 통해 라우팅 결정
    planner_input = {
        "input": user_input,
        "chat_history": chat_history
    }
    # 실제 프롬프트 렌더링
    planner_prompt_rendered = ROUTING_PROMPT.format(**planner_input)
    planner_output = router.invoke(planner_input)

    # 2. Executor를 통해 초기 답변 생성
    executor_input = {
        "topic": planner_output,
        "input": user_input,
        "chat_history": chat_history
    }
    # executor 단계 프롬프트 추출 (카테고리에 따라 다름)
    if planner_output.get('category') == 'retrieval_qa':
        exec_prompt_rendered = qa_prompt_template_str.format(
            syllabus=SYLLABUS,
            previous_learned_contents=PREVIOUS_LEARNED_CONTENTS,
            context=str(retriever.invoke(planner_output['arguments']['retrieval_query'])),
            chat_history=chat_history,
            input=planner_output['arguments']['question']
        )
    elif planner_output.get('category') == 'qa':
        exec_prompt_rendered = qa_prompt_template_str.format(
            syllabus=SYLLABUS,
            previous_learned_contents=PREVIOUS_LEARNED_CONTENTS,
            context="No context provided",
            chat_history=chat_history,
            input=planner_output['arguments']['question']
        )
    else:
        exec_prompt_rendered = None
    initial_answer_result = branch.invoke(executor_input)
    
    if isinstance(initial_answer_result, dict) and 'answer' in initial_answer_result:
        initial_answer = initial_answer_result['answer']
    else:
        initial_answer = str(initial_answer_result)

    # 3. Reflector를 통해 답변 검증 및 최종 답변 생성
    reflector_input = {
        "syllabus": SYLLABUS,
        "previous_learned_contents": PREVIOUS_LEARNED_CONTENTS,
        "chat_history": chat_history,
        "question": user_input,
        "initial_answer": initial_answer
    }
    reflector_prompt_rendered = reflection_prompt_template.format(**reflector_input)
    final_answer = reflection_chain.invoke(reflector_input)

    reasoning = {
        "1_planner_input": planner_input,
        "1_planner_prompt": planner_prompt_rendered,
        "1_planner_output": planner_output,
        "2_executor_input": executor_input,
        "2_executor_prompt": exec_prompt_rendered,
        "2_initial_answer": initial_answer,
        "3_reflector_input": reflector_input,
        "3_reflector_prompt": reflector_prompt_rendered,
        "3_final_answer": final_answer
    }

    return final_answer, reasoning

# Example Usage:
if __name__ == '__main__':
    chat_history = []
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        response, reasoning = get_final_answer(user_input, chat_history)
        print(f"Assistant: {response}")
        print(f"Reasoning: {reasoning}")
        from langchain_core.messages import HumanMessage, AIMessage
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response)) 