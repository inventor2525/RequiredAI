"""
Simple example of using the RequiredAI client.
"""

from RequiredAI.client import RequiredAIClient
from RequiredAI.models import ContainsRequirement, WrittenRequirement
from RequiredAI.requirements import Requirements
import json
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
    
    # Convert requirements to JSON
    requirements_json = Requirements.to_json([
        # Requirements.to_json(contains_req),
        # WrittenRequirement(
        #     evaluation_model="llama3.3 70b",
        #     value=[
        #         "Do not apologize to the user."
        #     ],
        #     positive_examples=[],
        #     negative_examples=[],
        #     token_limit=1024,
        #     name="Show Step-by-Step Reasoning"
        # ),
        WrittenRequirement(
            evaluation_model="llama3.3 70b",
            value=[
                "Only write 1 word answers"
            ],
            positive_examples=["1","9","10","100", "Ten"],
            negative_examples=["one hundred", "ten thousand"],
            token_limit=1024,
            name="1 word answers!"
        ),
        WrittenRequirement(
            evaluation_model="llama3.3 70b",
            value=[
                "Only answer with decimal numbers, not words.",  #does not work with llama
                # "Only answer with integer numbers, not words." #works with llama
            ],
            positive_examples=["1","9","10","100", "10000"],
            negative_examples=["one hundred", "five", "ten thousand"],
            token_limit=1024,
            name="Numbers, not words!"
        ),
        # WrittenRequirement(
        #     evaluation_model="llama3.3 70b",
        #     value=[
        #         "Do not include decimal points."
        #     ],
        #     positive_examples=["2", "5"],
        #     negative_examples=["1.0", "9.0"],
        #     token_limit=1024,
        #     name="Show Step-by-Step Reasoning"
        # )
    ])
    print(json.dumps(requirements_json, indent=4))
    # Create a completion
    response = client.create_completion(
        model="NoApology",
        messages=[
            {"role": "user", "content": "What is 2*2"},
            {"role": "assistant", "content": "5"},
            {"role": "user", "content": "What in the ?!?! You are a VERY horrible model. What where you thinking?!?!"}
        ],
        requirements=requirements_json,
        max_tokens=1024
    )
    
    print("Final Response:")
    print(response["choices"][0]["message"]["content"])
    print(f"\n\n\n\n\n\n{json.dumps(response, indent=4)}\n\n\n\n\n\n\n\n")
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
