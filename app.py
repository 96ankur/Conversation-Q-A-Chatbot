# Check Execution flow.md and image to understant the flow
import os
import streamlit as st
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## Set up Streamlit
st.title("Conversatinal RAG With PDF uploads and chat histroy")
st.write("Upload PDF and Chat with their content")

## Input the Groq API key
api_key = st.text_input("Enter your Groq API key:", type="password")

## Check if groq is provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile")

    session_id=st.text_input("Session ID", value="default_session")

    ## statefully manage the chat history

    if 'store' not in st.session_state:
        st.session_state.store = {}
    
    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)

    ## Process uploaded PDF's
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            tempPdf=f"./temp.pdf"
            with open(tempPdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFLoader(tempPdf)
            docs = loader.load()
            documents.extend(docs)
        
        # Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        ## Prompt to reformulate user questions into standalone queries for better retrieval, considering chat history. 
        # Example: 'What about that?' -> 'What about the RAG system we discussed?'
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        """
        (create_history_aware_retriever)
        - What create_history_aware_retriever does: Normally, a retriever searches your uploaded PDFs for documents that match the user's exact question. But in a chat, questions often refer 
                                                    to previous topics (e.g., "What about that?"). This function uses the AI (your Llama model) to first "fix" or clarify the question based on
                                                    chat history, then searches for relevant PDF chunks. It combines the base retriever (which searches embeddings) with a prompt that handles 
                                                    history.
        - Why it's useful: Without it, vague or follow-up questions might not find the right info from your PDFs, leading to poor answers. It makes the app feel more like a natural conversation.
        - How it works in your code: It takes three inputs:
            - llm: Your AI model (Llama 3.3) to reformulate the question.
            - retriever: The basic search tool from your Chroma vector database.
            - contextualize_q_prompt: The prompt we discussed earlier, which tells the AI how to rephrase questions.
        
        Example: Imagine you upload a PDF about "Generative AI" and ask:
        - First question: "What is RAG?"
            - The retriever searches for "RAG" in the PDF and finds relevant chunks. The AI answers based on that.
        - Follow-up question: "How does it work?" (referring to RAG from the previous message).
            - Without history-aware retriever: It might search for "How does it work?" generically, missing context and pulling wrong info.
            - With history-aware retriever: The AI sees the history ("What is RAG?" → answer about RAG). It reformulates the question to "How does RAG work?" before searching. Now it finds the 
                                            right PDF sections about RAG's mechanics and gives a better answer.
        """
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        ## Prompt to guide the AI in generating concise answers using retrieved context and chat history. 
        # Example: Answer in max 3 sentences, e.g., 'RAG combines retrieval and generation. It uses documents to ground responses. This prevents hallucinations.'
        system_prompt = (
            "You are an assistant for question-answering tasks."
            "Use the following prieces of retrieced context tot answer"
            "the question. If you don;t know the answer, say that you"
            "don't know. Use three sentences maximum and keet the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        """
        create_stuff_documents_chain takes the documents retrieved by the retriever, inserts (stuffs) them into the prompt, 
        and asks the LLM to generate an answer using those documents.
        It does not retrieve data itself—it only handles how documents are passed to the LLM for answering.
        """
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        """
        create_retrieval_chain connects the retriever and the answer-generation chain into a single RAG pipeline.
        It ensures the user’s question is used to retrieve documents and that those documents are automatically passed to the LLM for answering.
        """
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        """
        get_session_history function:
         - Creates one memory per user/session
         - Prevents different users from sharing history
         - Uses Streamlit’s session state safely
        """
        def get_session_history(session:str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]
        
        """
        RunnableWithMessageHistory adds conversational memory to your chain by storing and replaying past user and AI messages.
        It enables follow-up questions, context awareness, and multi-turn conversations in your chatbot.
        """
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id": session_id}
                },
            )
            st.write(st.session_state.store)
            st.write("Assistant: ", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter Groq API key")