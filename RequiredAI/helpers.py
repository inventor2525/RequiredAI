from typing import Dict, List

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