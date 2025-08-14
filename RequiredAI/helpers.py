from typing import Dict, List
from dataclasses import dataclass, field, asdict, fields

def get_id(response :Dict[str,str]) -> str:
    '''
    Safely get id from response['id']
    '''
    return response.get('id', f"Value Error '{response}' has no id")

def get_msg(response :Dict[str,List[Dict[str,Dict[str,str]]]]) -> Dict[str,str]:
    '''
    Safely get message dict from response['choices'][0]['message']
    '''
    choices = response.get('choices', [])
    if len(choices) == 0:
        return ""
    return choices[0].get('message', {'role':'user', 'content':f"Value Error '{response}' has no message object"})

def get_msg_content(response :Dict[str,List[Dict[str,Dict[str,str]]]]) -> str:
    '''
    Safely get message content from response['choices'][0]['message']['content']
    '''
    msg = get_msg(response)
    return msg.get('content', f"Value Error '{response}' has no message content")

def get_finish_reason(response :Dict[str,List[Dict[str,str]]]) -> str:
    '''
    Safely get finish reason from response['choices'][0]['finish_reason']
    '''
    choices = response.get('choices', [])
    if len(choices) == 0:
        return ""
    return choices[0].get('finish_reason', f"Value Error '{response}' has no finish_reason")

def code_block_text(text:str, language:str='txt'):
    '''Wraps text in a markdown code block of the specified language.'''
    return f"```{language}\n{text}\n```"

def indent_text(text:str, indent:str='\t') -> str:
    '''Indent text over with 'indent' str.'''
    return indent + f'\n{indent}'.join(text.split('\n'))

from typing import Type, TypeVar, Any

T = TypeVar('T')

def json_dataclass(cls: Type[T]) -> Type[T]:
    """
    A decorator that applies @dataclass, @dataclass_json, and overrides from_json
    to call __post_init__ after deserialization.
    """
    from dataclasses_json import dataclass_json
    from functools import wraps
    # Step 1: Apply @dataclass
    cls = dataclass(cls)
    
    # Step 2: Apply @dataclass_json
    cls = dataclass_json(cls)
    
    # Step 3: Store the original from_json method
    original_from_json = cls.from_json
    
    # Step 4: Define a new from_json that calls the original and then __post_init__
    @classmethod
    @wraps(original_from_json)
    def custom_from_json(cls: Type[T], json_data: str, **kwargs: Any) -> T:
        # Call the original from_json to get the instance
        instance = original_from_json(json_data, **kwargs)
        # Call __post_init__ if it exists
        if hasattr(instance, '__post_init__'):
            instance.__post_init__()
        return instance
    
    # Step 5: Replace the class's from_json with the custom one
    cls.from_json = custom_from_json
    
    return cls