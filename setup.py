from setuptools import setup, find_packages

setup(
    name="mbti-game",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'fastapi',
        'uvicorn',
        'pydantic',
        'transformers',
        'torch',
        'sentencepiece',
        'protobuf',
        'pytest',
        'httpx',
    ],
    python_requires='>=3.8',
)
