from RequiredAI.client import RequiredAIClient
from RequiredAI.ModelConfig import ModelConfig

llama_70b = ModelConfig(
	name="Llama 70b 3.3",
	provider="groq",
	provider_model="llama-3.3-70b-versatile"
)
gemini_flash_lite = ModelConfig(
	name="Gemini 2.5 Flash Lite",
	provider="gemini",
	provider_model="gemini-2.5-flash-lite"
)

client = RequiredAIClient(
	base_url="http://localhost:5432"
)

# response = client.create_completion(gemini_flash_lite.name, [
# 	{
# 		'role':'user',
# 		'content':open('/home/charlie/Projects/test 2025-01-06 17-54-16.py').read()
# 	},
# 	{
# 		'role':'assistant',
# 		'content':'Ok... Cool! What the heck am I suppose to do with that?'
# 	},
# 	{
# 		'role':'user',
# 		'content':'nothing... Just say Hi and tell me like 2 sentences describing what I sent you.'
# 	}
# ])
# print(response['choices'][0]['message']['content'])

while True:
	response = client.create_completion(llama_70b.name, [
		{
			'role':'user',
			'content':'Say Hello'
		}
	])
	import json
	import datetime

	with open(f'/home/charlie/Projects/model_output/{datetime.datetime.now()}', 'w') as f:
		json.dump(response, f, indent=4)

	print(response['choices'][0]['message']['content'])