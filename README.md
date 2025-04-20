# RequiredAI
RequiredAI is a client and server api that provides a way to add requirements to AI responses.

Rather than telling an AI "please dont apologize to me" in something like a system message, you provide a list of requirements with your completion request, and you only get a response back that satisfies those requirements.

# Requirement Model
```python
@requirement("Contains")
@dataclass
class ContainsRequirement:
    value:List[str]

    def evaluate(self, messages:List[dict]) -> bool:
        '''
        Checks that the last message in the passed conversation (which is presumed to be from an AI), contains any of the values in value.
        '''
        pass

    @property
    def prompt(self) -> str:
        '''Returns a string explaining how the conversation in the last call to evaluate that returned false did not meet this requirement'''
```
> all requirements have evaluate and prompt which maybe should be an interface but I want a registry and web name for each class that im stupidly going back and forth on if it should be a decprator or base class with a field setter in the child class for the name and some reflection to find inherited classes.... cause why wouldnt I focus on that rather than utility?!?!

The requirement decorator registers it as a requirement, in "Requirements" at the specified string key name used in the json equivalent.

```python
class Requirements:
    @staticmethod
    def to_json(requirement) -> dict:
        pass

    @staticmethod
    def from_json(j:dict) -> Any:
        '''Finds the requirement type from the registry, creates an instance and populates it'''
    
T = TypeVar("T")
def requirement(name:str) -> Callable[[T],T]:
    '''Registers the requirement in a static list of requirements held in Requirements class and stores the passed name in the decorated requirement class as __web_name__.'''
```

Notice how every requirement type has an evalute function.

# Client API
Requests work very similar to any chat llm api (and intentionally mirror some of the most common APIs), all the user needs to do is provide the list of requirements in json. For example:

```python
import anthropic

message = anthropic.Anthropic().messages.create(
    model="claude-3-7-sonnet-20240219",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ],
    requirements=[
        {
            "type":"Contains",
            "value":["(A)", "(B)", "(C"]
        },
        {
            "type":"Written",
            "value":["There shall be thoughts as to think though what the answer should be BEFORE *any* answer in the correct format is written at the end.", "Think through your answer before providing one"],
            "possitive_examples"=[],
            "negative_examples"=[],
            "model"="sonnet",
            "token_limit"=1024
        }
    ]
)
print(message)
```

In this example 2 forms of requirements are shown.
1. A simple contains check where the response from the AI will only be considered as valid if it contains something that can be easilly machine parsed as the answer.
2. A written requirement whos value is a list of any number of ways of saying the same thing.
Each of those different ways of stating the requirement will be sent along with any examples to the specified model (or the model the chat is sent to if no requirement specific model is provided). On the server side so many examples are combined with each way of the ways of saying the requirement, up to token_limit, and the request is made.


# Server API
The server has config for any number of ai frameworks to route to things like anthropic, grok, openai, etc, and it simply adds a layer onto the most common of chat formats and endpoints for chat completion (implemented in flask), but uses the added requirements field to:

1. strip the requirements from the request and simply send the chat completion to the model provided at the route of the chat completion request
2. after a response is received, we iterate the requirements passing the conversation with the response added to each of their evaluate functions
3. if a requirement evaluate is false the model listed in the requirement (or the model in the root of the chat if none provided) is prompted with the origional conversation+response + a prompt from the requirement telling the model how it should follow that requirement (aka the property) that it didnt follow and then prompted (with just a blanket prompt loaded from the server config) to revise its response to meet the requirement, returning a new draft of the response.

So... to be clear, the first ai appends response to conversation that we call prospective response. Prospective response is then iterated to convergence as:

```sudo code
requirements = [...]
chat = [...]
chats_ai
revise_prompt_template = load(...)

prospective_response = chats_ai.complete(chat)

failed_requirements = set

try_again:
for requirement in requirements:
    c = chat + prospective_response
    m = requirement.model else chats_ai
    if not requirement.evaluate(c):
        p = revise_prompt_template.render(
            requirement.prompt, c)
        prospective_response = m.complete(c + p)
        goto try_again #we try again from scratch here because we want to not include information about prior attempts to meet the requirement in the conversation as to not lead the witness. we also want to check that it now meets the requirement and now that its revised, we want it to not violate the other ones.
```
