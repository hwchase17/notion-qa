import streamlit as st
from qa import QA
from sheets import Edit_Sheet


# class App():

#   def __init__(self):
#     self.run = False

st.set_page_config(page_title="AI Task Rabbit", page_icon="🔧")
st.header("🔧  AI Task Rabbit 🔧")
gsheet = st.text_area("What Google Sheet do you want to update? Please provide the URL")
gcells = st.text_area("What cells do you want to update? For example, B1\:B10")
query = st.text_area("Please provide a list of names separated by commas")
# button = st.button("Submit")

# if st.session_state.get("submit"):
if st.button("Submit"):
  print("here")
  # get list of names from user 
  name_list = query.split(', ')

  answers = []
  sources = []
  qa_instance = QA()
  for name in name_list:
    question = f'What is {name}\'s LinkedIn URL? Only return the URL.'
    answer, source = qa_instance.ask_question(question=question)
    answers.append(answer)
    sources.append(source)

  st.text(f'Answers: {answers}')

  answers2 = []
  for answer in answers:
    print(answer)
    if "I don\'t know" not in answer:
      question = f'{answer}. Who\'s LinkedIn is this? What city are they located in?"'
      # qa_instance2 = QA()
      answer2, source = qa_instance.ask_question(question=question)
      answers2.append(answer2)
    else: 
      answers2.append("Unable to find LinkedIn")

  st.text(f'Answers: {answers2}')

  formatted_list = []
  for item in answers2:
    formatted_list.append([item])

  print(f'{formatted_list=}')
  gsheets = Edit_Sheet()
  gsheets.update(formatted_list)

  st.text("Your Google Sheet has been updated!")


  



