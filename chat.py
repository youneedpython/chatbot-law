import uuid
import streamlit as st
from llm import stream_ai_message


st.set_page_config(page_title='전세사기피해 상담 챗봇', page_icon='🍀')
st.title('🍀 전세사기피해 상담 챗봇')

print('\n\n== start ==')
print('before) st.session_state >>', st.session_state)

## URL의 parameter에 session id 가져오기/저장 =====================================
query_params = st.query_params

if 'session_id' in query_params:
    session_id = query_params['session_id']
else :
    session_id = str(uuid.uuid4())
    st.query_params.update({'session_id': session_id})

## Streamlit 내부 세션: session id 저장 
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = session_id

## Streamlit 내부 세션: 메시지 리스트 초기화
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

print('after) st.session_state >>', st.session_state)

## 이전 채팅 내용 화면 출력 =========================================================
for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message['content'])

## 사용자 질문 -> AI 답변 ============================================================
placeholder = '전세사기피해와 관련된 궁금한 내용을 질문하세요.'

if user_question := st.chat_input(placeholder=placeholder): ## prompt 창
    ## 사용자 메시지 ##############################
    with st.chat_message('user'):
        ## 사용자 메시지 화면 출력
        st.write(user_question)
    st.session_state.message_list.append({'role': 'user', 'content': user_question})

    ## AI 메시지 ##################################
    with st.spinner('답변 생성하는 중입니다.'):
        session_id = st.session_state.session_id
        ai_message = stream_ai_message(user_question, session_id=session_id)

        with st.chat_message('ai'):
            ## AI 메시지 화면 출력
            ai_message = st.write_stream(ai_message)
        st.session_state.message_list.append({'role': 'ai', 'content': ai_message})
