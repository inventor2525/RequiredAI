from setuptools import setup, find_packages

setup(
	name="RequiredAI",
	version="0.1.0",
	author="Charlie Mehlenbeck",
	author_email="charlie_inventor2003@yahoo.com",
	description="A client and server API for LLM chat that follow strict requirements.",
	long_description=open("README.md").read(),
	long_description_content_type="text/markdown",
	url="https://github.com/inventor2525/RequiredAI",
	packages=find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
	python_requires=">=3.10.12",
	install_requires=[
		"flask",
		"requests",
		"anthropic",
		"groq",
		"google-genai",
		"dataclasses-json",
	],
	extras_require={
		"dev": ["unittest"],
	},
)