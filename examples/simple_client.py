"""
Simple example of using the RequiredAI client.
"""

from requireai.client import RequiredAIClient
from requireai.models import ContainsRequirement, WrittenRequirement
from requireai.requirements import Requirements

def main():
    # Create a client
    client = RequiredAIClient(
        base_url="http://localhost:5000",
        api_key="your_api_key_here"
    )
    
    # Create requirements
    contains_req = ContainsRequirement(value=["(A)", "(B)", "(C)"])
    written_req = WrittenRequirement(
        value=[
            "Think through your answer before providing one",
            "There shall be thoughts as to think though what the answer should be BEFORE *any* answer in the correct format is written at the end."
        ],
        positive_examples=[],
        negative_examples=[],
        model="GPT-4",
        token_limit=1024
    )
    
    # Convert requirements to JSON
    requirements_json = [
        Requirements.to_json(contains_req),
        Requirements.to_json(written_req)
    ]
    
    # Create a completion
    response = client.create_completion(
        model="claude-3-7-sonnet-20250219",
        messages=[
            {"role": "user", "content": "Hello, Claude"}
        ],
        requirements=requirements_json,
        max_tokens=1024
    )
    
    print("Response:")
    print(response["choices"][0]["message"]["content"])

if __name__ == "__main__":
    main()
