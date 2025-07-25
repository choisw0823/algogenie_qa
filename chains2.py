from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_openai import ChatOpenAI

# Planner 역할을 할 LLM
planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 라우팅을 위한 프롬프트
routing_prompt_template = """
You are a router for QA chatbot. You need to decide next plan based on history and user's new chat.

1. retrieval_qa: Questions that require reference to the provided lecture materials, such as lecture content, schedule, or specific concepts.
2. general_knowledge: Questions that can be answered with general programming (Python) knowledge, without the need for lecture materials.
3. off_topic: Questions that are completely unrelated to programming or the lecture.

사용자 질문: "{input}"
카테고리: 
"""
ROUTING_PROMPT = PromptTemplate.from_template(routing_prompt_template)

# 라우터 체인 생성
# | str_output_parser() 를 붙여서 LLM의 출력을 순수 문자열로 만듭니다.
router = ROUTING_PROMPT | planner_llm | str.strip

# 각 경로에 따른 실행 체인 정의
# 1. retrieval_qa 체인 (이미 만드신 것)
retrieval_chain = get_conversational_rag(get_retreiver_chain(get_vector_store()))

# 2. general_knowledge 체인 (간단한 LLM 체인)
general_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful python programming assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])
general_chain = general_prompt | ChatOpenAI(model="gpt-4o", temperature=0.7)

# 3. off_topic 체인 (고정된 답변)
def off_topic_responder(input_dict):
    return "저는 과학고 프로그래밍 수업 조교이므로, 수업과 관련된 질문에만 답변할 수 있습니다. 다른 질문이 있으시면 말씀해주세요."

# RunnableBranch를 사용하여 전체 플래너 구성
# .with_config(run_name="...")을 사용하면 LangSmith 등에서 추적하기 용이합니다.
full_planner_chain = RunnableBranch(
    (lambda x: "retrieval_qa" in x["topic"], retrieval_chain.with_config(run_name="RetrievalQA")),
    (lambda x: "general_knowledge" in x["topic"], general_chain.with_config(run_name="GeneralKnowledge")),
    (lambda x: "off_topic" in x["topic"], off_topic_responder),
    off_topic_responder  # 기본값 (분류에 실패한 경우)
)

# 최종적으로 실행할 체인
# 사용자의 입력을 받아, 먼저 router로 주제를 분류하고, 그 결과를 full_planner_chain에 넘겨줍니다.
final_chain = {"topic": router, "input": lambda x: x["input"], "chat_history": lambda x: x["chat_history"]} | full_planner_chain

# 사용 예시
# result = final_chain.invoke({
#     "input": "버블 정렬에 대해 알려줘",
#     "chat_history": []
# })
# print(result)


# Reflector 역할을 할 LLM
reflector_llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 검증을 위한 프롬프트
reflection_prompt_template = """
당신은 시니어 조교로서, 주니어 조교가 작성한 답변을 검토합니다.
아래 평가 기준에 따라 답변을 분석하고, 수정이 필요하다면 더 나은 답변을 제안해주세요.

[평가 기준]
1. 강의 범위 준수: 답변이 아래 강의 계획표 내용을 벗어나지 않았는지 확인합니다.
   {syllabus}
2. 출처 표기: 강의 자료(context)를 사용한 경우, 'Source: ...' 형식으로 정확히 출처를 표기했는지 확인합니다.
3. 명확성 및 충분성: 학생이 이해하기 쉽고, 질문에 대해 충분히 상세하게 설명했는지 확인합니다.

[원본 질문]
{question}

[주니어 조교의 초기 답변]
{initial_answer}

[검토 결과]
위 평가 기준에 따라 답변을 검토한 결과입니다.
- 답변이 모든 기준을 충족합니까? (예/아니오)
- 만약 '아니오'라면, 어떤 점이 부족하고 어떻게 수정해야 하는지 구체적으로 설명하고, 수정된 답변 전체를 다시 작성해주세요.
"""
REFLECTION_PROMPT = PromptTemplate.from_template(reflection_prompt_template)

# 강의 계획표 (SYSTEM_PROMPT에서 가져옴)
syllabus = """
1st week, 5/25: 정렬, 선택정렬, 버블정렬, 탐색, 순차탐색, 이진탐색, 객체지향
2nd week, 6/1: 정렬 비교, 반복문, 탐색 코드
3rd week, 6/8: 알고리즘, 빅오표기법, Pandas
4th week, 6/15: Pandas, 지역변수, 젼역변수, random
5th week, 6/22: 삽입정렬, 재귀함수, 클래스, 코드업(self-number), matplotlib, folium, 인공지능 개념
6th week, 6/29: FINAL about whole contents
"""

# Reflector 체인 생성
reflection_chain = REFLECTION_PROMPT | reflector_llm

# 전체 프로세스 (수도 코드)
def get_final_answer(user_input, chat_history):
    # 1. Planner & Executor를 통해 초기 답변 생성
    initial_answer = final_chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    # 2. Reflector를 통해 답변 검증
    reflection_result = reflection_chain.invoke({
        "syllabus": syllabus,
        "question": user_input,
        "initial_answer": initial_answer['answer'] # RAG chain 결과가 dict 형태일 수 있음
    }).content

    # 3. 검증 결과에 따라 최종 답변 결정
    if "아니오" in reflection_result:
        # 수정된 답변을 파싱해서 반환
        # (실제 구현 시에는 "수정된 답변:" 같은 특정 키워드 뒤의 텍스트를 추출)
        final_answer = reflection_result.split("수정된 답변:")[-1].strip()
        return final_answer
    else:
        return initial_answer['answer']