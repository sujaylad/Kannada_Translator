import os
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_groq import ChatGroq

def translate_review(review_text, api_key):
    # Initialize the LLM
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.5,
        max_completion_tokens=1024,
        top_p=1,
        reasoning_effort="medium",
        stream=False,
        stop=None,
        api_key=api_key
    )

    # Examples for Few-Shot learning
    examples = [
        {
            "review": "The movie was fantastic.",
            "answer": "ಸಿನಿಮಾ ಅದ್ಭುತವಾಗಿತ್ತು."
        },
        {
            "review": "The food tasted horrible.",
            "answer": "ಆಹಾರದ ರುಚಿ ಬಹಳ ಕೆಟ್ಟದಾಗಿತ್ತು."
        },
        {
            "review": "The phone is okay.",
            "answer": "ಫೋನ್ ಪರವಾಗಿಲ್ಲ."
        }
    ]

    example_prompt = PromptTemplate(
        input_variables=["review", "answer"],
        template="\nReview:\n{review}\n\nAnswer:\n{answer}\n"
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="""You are an expert Translating Text from English to Kannada.\n\nLook at the examples below.""",
        suffix="""\nReview:\n{review}\n\nAnswer:\n""",
        input_variables=["review"]
    )

    chain = few_shot_prompt | llm
    response = chain.invoke({"review": review_text})
    return response.content

if __name__ == "__main__":
    # Example usage
    import sys
    
    # Retrieve API key from environment variable
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable is not set.")
        sys.exit(1)

    test_review = "The laptop performance is excellent."
    print(f"Original Review: {test_review}")
    translation = translate_review(test_review, api_key)
    print(f"Translation: {translation}")
