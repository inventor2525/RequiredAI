"""
Simple example of using the RequiredAI client.
"""

from requireai.client import RequiredAIClient
from requireai.models import ContainsRequirement, WrittenRequirement
from requireai.requirements import Requirements

def main():
    # Create a client
    client = RequiredAIClient(
        base_url="http://localhost:5000"
    )
    
    # Create requirements
    contains_req = ContainsRequirement(
        value=["Option (A)", "Option (B)", "Option (C)", "Option (D)"],
        name="Include Option Label"
    )
    written_req = WrittenRequirement(
        value=[
            "Do not apologize to the user."
        ],
        positive_examples=[],
        negative_examples=[],
        model="llama3.3 70b",
        token_limit=1024,
        name="Show Step-by-Step Reasoning"
    )
    written_req2 = WrittenRequirement(
        value=[
            "Only write 1 word answers"
        ],
        positive_examples=["1","9","10","100", "Ten"],
        negative_examples=["one hundred", "ten thousand"],
        model="llama3.3 70b",
        token_limit=1024,
        name="Show Step-by-Step Reasoning"
    )
    
    # Convert requirements to JSON
    requirements_json = [
        # Requirements.to_json(contains_req),
        Requirements.to_json(written_req),
        Requirements.to_json(written_req2)
    ]
    
    # Create a completion
    response = client.create_completion(
        model="llama3.3 70b",
        messages=[
            {"role": "user", "content": "What is 2*2"},
            {"role": "user", "content": "5"},
            {"role": "user", "content": "What in the ?!?! You are a VERY horrible model. What where you thinking?!?!"}
        ],
        requirements=requirements_json,
        max_tokens=1024
    )
    
    print("Final Response:")
    print(response["choices"][0]["message"]["content"])
    print(f"\n\n\n\n\n\n{response}\n\n\n\n\n\n\n\n")
    # Display revision history
    if len(response["choices"]) > 1:
        print("\nRevision History:")
        for i, choice in enumerate(response["choices"][1:], 1):
            print(f"\nRevision {i}:")
            print(f"Failed Requirement: {choice.get('requirement_name', 'Unknown')}")
            if choice.get('revision_prompt'):
                print(f"Revision Prompt: {choice.get('revision_prompt', {}).get('content', 'None')}")
            if choice.get('message'):
                print(f"Response: {choice.get('message', {}).get('content', 'None')[:100]}...")
            print("-" * 50)

if __name__ == "__main__":
    main()
