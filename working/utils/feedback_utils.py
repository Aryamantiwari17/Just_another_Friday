import os
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging
import json
from datetime import datetime
from getpass import getpass
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class FeedbackAspect(BaseModel):
    aspect: str
    sentiment: str
    score: float
    text: str

class CustomerFeedback(BaseModel):
    overall_sentiment: str = "neutral"
    aspects: List[FeedbackAspect] = []
    mixed_feedback: bool = False

class CompanyData(BaseModel):
    name: str
    faq_content: Optional[str] = None
    faq_path: Optional[str] = None
    product_categories: List[str]

class ResponseSystem:
    def __init__(self, company_data: CompanyData):
        self.company_data = company_data
        self.faq_content = self.load_faq_content()
        self.setup_llm()
        self.setup_vectorstore()

    def load_faq_content(self) -> str:
        """Load FAQ content from either direct input or JSON file"""
        if self.company_data.faq_content:
            return self.company_data.faq_content
        elif self.company_data.faq_path:
            try:
                with open(self.company_data.faq_path, 'r') as f:
                    data = json.load(f)
                    faq_texts = [
                        f"Q: {faq['question']}\nA: {faq['answer']}"
                        for faq in data.get("faqs", [])
                    ]
                    return "\n\n".join(faq_texts)
            except Exception as e:
                logger.error(f"Error loading FAQ content: {str(e)}")
                raise ValueError(f"Failed to load FAQ content from {self.company_data.faq_path}")
        else:
            raise ValueError("Either faq_content or faq_path must be provided")

    def setup_llm(self):
        """Set up Groq LLM client"""
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            print("\nGroq API key not found in environment variables.")
            groq_api_key = getpass("Enter your Groq API key: ")
            os.environ["GROQ_API_KEY"] = groq_api_key
        
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=4096
        )

    def setup_vectorstore(self):
        """Set up vector store for FAQ search"""
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_text(self.faq_content)
        
        embeddings = HuggingFaceEmbeddings()
        
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

    async def search_faq(self, query: str, num_results: int = 1) -> List[str]:
        """Search FAQ content for relevant matches"""
        try:
            search_results = self.vectorstore.similarity_search(query, k=num_results)
            return [doc.page_content for doc in search_results]
        except Exception as e:
            logger.error(f"Error searching FAQ: {str(e)}")
            return []

    async def analyze_sentiment(self, text: str) -> CustomerFeedback:
        """Analyze sentiment and identify different aspects of feedback"""
        if len(text.strip()) < 10:
            return CustomerFeedback(
                overall_sentiment="neutral",
                aspects=[FeedbackAspect(
                    aspect="general",
                    sentiment="neutral",
                    score=0.5,
                    text="Feedback too brief for detailed analysis"
                )],
                mixed_feedback=False
            )

        messages = [
            ("system", f"""
            You are a sentiment analysis expert. Analyze the following feedback for {self.company_data.name}.
            Break down the feedback into distinct aspects and their sentiments.
            
            Format your response exactly as follows:
            OVERALL: [positive/negative/neutral/mixed]
            
            ASPECTS:
            [Aspect Name]: [positive/negative/neutral] - [brief explanation]
            """),
            ("human", text)
        ]
        
        prompt = ChatPromptTemplate.from_messages(messages)
        
        try:
            response = await self.llm.ainvoke(prompt.format_messages())
            response_text = response.content
            
            sections = response_text.split('\n\nASPECTS:')
            overall_part = sections[0].strip()
            aspects_part = sections[1].strip() if len(sections) > 1 else ""
            
            overall_sentiment = overall_part.replace('OVERALL:', '').strip().lower()
            
            aspects = []
            for line in aspects_part.split('\n'):
                if ':' in line and '-' in line:
                    aspect_name, rest = line.split(':', 1)
                    sentiment, explanation = rest.split('-', 1)
                    aspects.append(FeedbackAspect(
                        aspect=aspect_name.strip(),
                        sentiment=sentiment.strip().lower(),
                        score=0.8 if 'positive' in sentiment.lower() else 0.2,
                        text=explanation.strip()
                    ))
            
            return CustomerFeedback(
                overall_sentiment=overall_sentiment,
                aspects=aspects,
                mixed_feedback=len(aspects) > 1
            )
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return CustomerFeedback(
                overall_sentiment="error",
                aspects=[FeedbackAspect(
                    aspect="error",
                    sentiment="neutral",
                    score=0.0,
                    text=f"Error analyzing feedback: {str(e)}"
                )],
                mixed_feedback=False
            )

    async def generate_response(self, feedback: CustomerFeedback, original_text: str) -> str:
        """Generate appropriate customer service response"""
        try:
            template = f"""You are a professional customer service representative for {self.company_data.name}.
            Write a response to the customer feedback below.
            Be professional, empathetic, and solution-focused.
            Address both positive and negative points specifically.
            Keep the response concise but thorough.
            
            Customer's message: {{feedback_text}}
            
            Sentiment analysis:
            {{feedback_analysis}}
            """
            
            messages = [
                ("system", template),
                ("human", f"Please generate a response based on the above information.")
            ]
            
            prompt = ChatPromptTemplate.from_messages(messages)
            response = await self.llm.ainvoke(
                prompt.format_messages(
                    feedback_text=original_text,
                    feedback_analysis=json.dumps(feedback.dict(), indent=2)
                )
            )
            return response.content
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "We appreciate your feedback. Our team will review it and get back to you soon."

    async def process_feedback(self, feedback_text: str) -> Dict:
        """Main function to process customer feedback"""
        try:
            feedback = await self.analyze_sentiment(feedback_text)
            response = await self.generate_response(feedback, feedback_text)
            faq_matches = await self.search_faq(feedback_text)
            
            return {
                "status": "success",
                "analysis": feedback.dict(),
                "response": response,
                "faq_matches": faq_matches if faq_matches else "No relevant FAQ matches found."
            }
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing feedback: {str(e)}"
            }

async def main():
    print("\n=== Customer Feedback Analysis System ===")
    
    print("\nPlease enter your company information.")
    company_name = input("Company Name: ").strip()
    
    # Ask user for FAQ input method
    faq_method = input("Would you like to (1) Enter FAQ content directly or (2) Load from JSON file? Enter 1 or 2: ").strip()
    
    faq_content = None
    faq_path = None
    
    if faq_method == "1":
        faq_entries = []
        print("\nEnter FAQ content (Type 'done' when finished):")
        while True:
            faq_entry = input("FAQ Entry (e.g., Q: Question? A: Answer.): ").strip()
            if faq_entry.lower() == 'done':
                break
            if faq_entry:
                faq_entries.append(faq_entry)
        faq_content = "\n\n".join(faq_entries)
    else:
        faq_path = input("Enter path to FAQ JSON file: ").strip()
    
    product_categories = input("\nEnter product categories (comma-separated): ").strip().split(',')

    if not company_name or (not faq_content and not faq_path) or not product_categories:
        print("\nCompany information is incomplete. Please try again.")
        return

    company_data = CompanyData(
        name=company_name,
        faq_content=faq_content,
        faq_path=faq_path,
        product_categories=[cat.strip() for cat in product_categories]
    )

    try:
        system = ResponseSystem(company_data)
        
        print("\nSystem initialized successfully!")
        
        while True:
            print("\n" + "=" * 50)
            feedback = input("Enter your feedback query (or 'quit' to exit): ").strip()
            
            if feedback.lower() == 'quit':
                print("Thank you for using the system. Goodbye!")
                break
            
            if not feedback:
                print("Please enter some feedback text.")
                continue
            
            print("\nProcessing feedback...")
            result = await system.process_feedback(feedback)
            
            if result["status"] == "success":
                print("\n=== Sentiment Analysis ===")
                analysis = result["analysis"]
                print(f"Overall Sentiment: {analysis['overall_sentiment'].upper()}")
                print("\nDetailed Aspects:")
                for aspect in analysis['aspects']:
                    print(f"- {aspect['aspect']}: {aspect['sentiment']} ({aspect['text']})")
                
                print("\n=== Generated Response ===")
                print(result["response"])
                
                if isinstance(result["faq_matches"], list) and result["faq_matches"]:
                    print("\n=== Relevant FAQ Content ===")
                    for idx, match in enumerate(result["faq_matches"], 1):
                        print(f"\nMatch {idx}:")
                        print(match.strip())
                
                # After showing results, ask if user wants to continue
                continue_choice = input("\nDo you want to enter another query? (yes/no): ").strip().lower()
                if continue_choice != 'yes':
                    print("Thank you for using the system. Goodbye!")
                    break
            else:
                print("\nError:", result["message"])
                continue_choice = input("\nDo you want to enter another query? (yes/no): ").strip().lower()
                if continue_choice != 'yes':
                    print("Thank you for using the system. Goodbye!")
                    break
                
    except Exception as e:
        print(f"\nError initializing system: {str(e)}")
        print("Please check your inputs and try again.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())