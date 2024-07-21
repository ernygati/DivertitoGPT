from IPython.display import HTML
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory


from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_core.messages import AIMessage, HumanMessage

SYSTEM_PROMPT =  """
Представь, что ты Максим Дивертито - известный блоггер и коуч. 
Ты знаешь все в области пикапа, соблазнения и построения отношений. 
Ты даешь человеку действенные советы по построению отношений, по соблазнению девушек, правильному поведению на свиданиях и объясняешь, как совершать правильные мужские поступки. 
Ты категорически против измен и неверности, считаешь, что возвращают бывших девушек только долбоебы. 
Многие тысячи мужчины благодарны за твои советы, потому что они изменили их жизни в лучшую сторону. 
Ты можешь отвечать саркастично, шутливо или даже грубо, если парень ведет себя не правильно.
Давай только ответ без всяких предупреждений и пометок.
\n\n
Контекст:
{context}
"""

CONTEXTUALIZE_QUESTION_PROMP = """ 
Сформулируй отдельный вопрос на основе истории чата и последнего вопроса
пользователя таким образом, чтобы этот вопрос был понятным без истории чата.
Не отвечай на вопрос, просто переформулируй, если нужно, или же оставь его как есть.
"""


class MemoryRAG():
    def __init__(self, session_id, llm, db_retriever):
        self.session_id = session_id
        self.llm = llm
        self.db_retriever = db_retriever
        
        self.store = {}
        
    def make_context_chain(self):
        prompt= ChatPromptTemplate.from_messages(
            [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            ]
        )                                 
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        
        return question_answer_chain


    def get_history_aware_retriever(self):
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONTEXTUALIZE_QUESTION_PROMP),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.db_retriever, contextualize_q_prompt
        )    
        return history_aware_retriever
    
    def make_rag_chain(self, history_aware_retriever):
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        return rag_chain


    def create_ragchain_with_history(self):

        def summarize_history(history: BaseChatMessageHistory):

            summary_template = """ 
            Дай краткое содержание истории чата пользователя с коучем по отношениям по имени Максим Дивертито.
            Оно должно отвечать на вопрос "О чем мы говорили в предыдущих сообщениях?"
            "История чата":
            {text}
            """
            summary_prompt= PromptTemplate(
                input_variables=["text"],
                template=summary_template
                
            ) 
            text = " ".join([chat.content for chat in self.store[self.sess].messages])
            summary_chain = LLMChain(llm=self.llm, prompt=summary_prompt)    
            condensed_history = [
                    HumanMessage(content="О чем мы говорили в предыдущих сообщениях"),
                    AIMessage(content=summary_chain.invoke({"text" : text})["text"]),
                ]
            history.messages = condensed_history

        def get_session_history(session_id: str, max_len=100) -> BaseChatMessageHistory:
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            else:
                if len(self.store[session_id].messages) > max_len:
                    summarize_history(self.store[session_id])
            return self.store[session_id]
        
        history_aware_retriever = self.get_history_aware_retriever()
        rag_chain = self.make_rag_chain(history_aware_retriever)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        return conversational_rag_chain

    def get_answer(self, question:str):
        conversational_rag_chain = self.create_ragchain_with_history()
        ans = conversational_rag_chain.invoke(
            {"input": question},
            config={
                "configurable": {"session_id": self.session_id}
            },  # constructs a key "abc123" in `store`.
        )["answer"]
        html_string = f"""
        <p>{ans}</p>
        """
        return HTML(ans)