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
            "First explain your reasoning process step by step",
            "Before giving your final answer, analyze each option carefully and explain why it is correct or incorrect"
        ],
        positive_examples=[],
        negative_examples=[],
        model="haiku",
        token_limit=1024,
        name="Show Step-by-Step Reasoning"
    )
    
    # Convert requirements to JSON
    requirements_json = [
        Requirements.to_json(contains_req),
        Requirements.to_json(written_req)
    ]
    
    # Create a completion
    response = client.create_completion(
        model="haiku",
        messages=[
            {"role": "user", "content": "Which of the following is a renewable energy source?\n\nOption (A): Coal\nOption (B): Natural gas\nOption (C): Solar power\nOption (D): Petroleum"}
        ],
        requirements=requirements_json,
        max_tokens=1024
    )
    
    print("Final Response:")
    print(response["choices"][0]["message"]["content"])
    
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
