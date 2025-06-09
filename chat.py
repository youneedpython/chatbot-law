import streamlit as st

st.set_page_config(page_title='전세사기피해 상담 챗봇', page_icon='🤖')
st.title('🤖 전세사기피해 상담 챗봇')

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

print(f'before: {st.session_state.message_list}')

## 이전 채팅 내용 화면 출력
for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message['content'])


## prompt 창(채팅 창) #############################################################
placeholder = '전세사기피해와 관련된 궁금한 내용을 질문하세요.'
if user_question := st.chat_input(placeholder=placeholder):
    with st.chat_message('user'):
        ## 사용자 메시지 화면 출력
        st.write(user_question)
    st.session_state.message_list.append({'role': 'user', 'content': user_question})

with st.chat_message('ai'):
    st.write('여기는 AI 메시지')
st.session_state.message_list.append({'role': 'ai', 'content': '여기는 AI 메시지'})

print(f'after: {st.session_state.message_list}')

print(user_question)