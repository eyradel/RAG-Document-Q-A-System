from rag_app import RAGApplication
import os

def main():
    # Initialize the RAG application
    rag_app = RAGApplication()
    
    # Process the PDF file
    pdf_path = os.path.join('data', 'Eyram_Dela.pdf')
    print(f"Processing PDF file: {pdf_path}")
    
    try:
        # Process the document
        rag_app.process_documents([pdf_path])
        
        # Save the index
        rag_app.save_index('index')
        
        print("PDF processed successfully!")
        
        # Example query
        question = "What is the main topic of this document?"
        print(f"\nAsking question: {question}")
        result = rag_app.query(question)
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    main() 