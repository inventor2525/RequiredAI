from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="RequiredAI",
    version="0.1.0",
    author="Charlie Mehlenbeck",
    author_email="charlie_inventor2003@yahoo.com",
    description="A client and server API for adding requirements to AI responses",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/RequiredAI",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "flask",
        "werkzeug",
        "requests",
        "anthropic",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "isort>=5.9.1",
        ],
    },
)
