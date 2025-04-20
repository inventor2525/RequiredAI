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
        value=["(A)", "(B)", "(C)"],
        name="Contains ABC Options"
    )
    written_req = WrittenRequirement(
        value=[
            "Think through your answer before providing one",
            "There shall be thoughts as to think though what the answer should be BEFORE *any* answer in the correct format is written at the end."
        ],
        positive_examples=[],
        negative_examples=[],
        model="sonnet",
        token_limit=1024,
        name="Show Reasoning Process"
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
            {"role": "user", "content": "Is the sky (A) orange, (B) green, (C) blue or (D) all the above?"}
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
