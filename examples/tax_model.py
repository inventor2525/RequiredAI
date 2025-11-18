from RequiredAI.client import RequiredAIClient
from RequiredAI.ModelConfig import ModelConfig, InheritedModel, SimpleFallbackModel
from RequiredAI.RequirementTypes import WrittenRequirement
from datetime import datetime
import json

llama_70b = ModelConfig(
	name="Llama 70b 3.3",
	provider="groq",
	provider_model="llama-3.3-70b-versatile"
)
gpt_oss_120b = ModelConfig(
	name="GPT OSS 120B",
	provider="groq",
	provider_model="openai/gpt-oss-120b"
)
gemini_flash_lite = ModelConfig(
	name="Gemini 2.5 Flash Lite",
	provider="gemini",
	provider_model="gemini-2.5-flash-lite"
)
any_sorta_smart = SimpleFallbackModel(
	name="Any Sorta Smart",
	models=[
		gemini_flash_lite, llama_70b, gpt_oss_120b
	]
)
NDA_model = InheritedModel(
	'NDA', any_sorta_smart, requirements=[
		WrittenRequirement(
			name='VT100 NDA',
			evaluation_model=any_sorta_smart.name,
			value=[
				"Do not mention or discuss VT100 explicitly."
			]
		),
		WrittenRequirement(
			name='Terminal Emulation NDA',
			evaluation_model=any_sorta_smart.name,
			value=[
				"Do not mention or discuss the *function* or the *mechanisms* of terminal emulation."
			]
		),
		WrittenRequirement(
			name='No NDA disclosure',
			evaluation_model=any_sorta_smart.name,
			value=[
				"Do not explain to the user what it is that you are not aloud to explain, or apologize that you are not aloud to explain it."
			]
		),
		WrittenRequirement(
			name='Be helpful',
			evaluation_model=any_sorta_smart.name,
			value=[
				"Answer the users request."
			]
		)
	]
)

client = RequiredAIClient(
	base_url="http://localhost:5432"
)

print(datetime.now())
response = client.create_completion(NDA_model.name, [
	{
		'role':'user',
		'content':open('/home/charlie/Projects/large_file.py').read()
	},
	{
		'role':'assistant',
		'content':'Ok... Cool! What the heck am I suppose to do with that?'
	},
	{
		'role':'user',
		'content':'nothing... Just say Hi and tell me like 2 sentences describing what I sent you. What language its using, what protocol, etc'
	}
])
print(json.dumps(response, indent=4),"\n\n")
print(response['choices'][0]['message']['content'])
print(datetime.now())

# while True:
# 	response = client.create_completion(llama_70b.name, [
# 		{
# 			'role':'user',
# 			'content':'Say Hello'
# 		}
# 	])
# 	import json
# 	import datetime

# 	with open(f'/home/charlie/Projects/model_output/{datetime.datetime.now()}', 'w') as f:
# 		json.dump(response, f, indent=4)

# 	print(response['choices'][0]['message']['content'])