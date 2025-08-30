from semantic_search_engine import SSE
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

class RAGSystem:
    def __init__(self, db_path: str, ollama_model: str = "mistral"):
        # Initialize vector store
        self.vector_db = SSE(db_path)
        self.vector_db.data_reader()

        # Load the model
        self.llm = OllamaLLM(model=ollama_model)

        # Define your prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Use the following context to answer the question. If the answer is not in the context, say The information is not available."
                "Context:\n{context}\n\n"
                "Question: {question}\n"
                "Answer:"
            )
        )

    def query(self, question: str, top_k: int = 4) -> str:
        print("-----------------------------")
        # Step 1: Retrieve top-k relevant documents using semantic search
        docs = self.vector_db.qdrant.similarity_search(question, k=1)
        print(docs)
        context = "\n\n".join([doc.page_content for doc in docs])
        print(context)
        print("-------------------------------")
        # Step 2: Format the prompt with the retrieved context
        prompt = self.prompt_template.format(context=context, question=question)
        # Step 3: Pass the prompt to the LLM and get the answer
        return self.llm.invoke(prompt)

if __name__ == "__main__":
    rag = RAGSystem(db_path="TEST")
    print("RAG System ready. Type your questions (type 'exit' to quit).")

    while True:
        query = input("🔍 Ask: ")
        if query.lower() in ["exit", "quit"]:
            break
        answer = rag.query(query)
        print(f"🧠 Answer: {answer}\n")
