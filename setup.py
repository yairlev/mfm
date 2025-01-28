from setuptools import setup, find_packages

setup(
    name="mfm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "typer[all]",
        "langchain",
        "langchain-community",
        "langchain-google-vertexai",
        "langchain-core",
        "langgraph",
        "openinference-instrumentation-langchain",
        "arize-phoenix-otel",
        # ... other dependencies ...
    ],
    entry_points={
        'console_scripts': [
            'mfm=mfm.cli:app'
        ],
    },
) 