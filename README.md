# Grant Question-Answering Bot

🤖Ask questions to your database in natural language🤖

💪 Built with [LangChain](https://github.com/hwchase17/langchain) and the [Facebook AI Similarity Search](https://github.com/facebookresearch/faiss) (faiss). 

# 🌲 Environment Setup

In order to set your environment up to run the code here, first install all requirements:

```shell
pip install -r requirements.txt
```

Then set your OpenAI API key (if you don't have one, get one [here](https://beta.openai.com/playground))

```shell
export OPENAI_API_KEY=....
```

# 📄 What is in here?
- Example xml data from the NSF
- Python script to query NSF xml data with a question
- Code to deploy on StreamLit
- Instructions for ingesting your own dataset

## 📊 Example Data
This repo uses the NSF grant data in xml as an example.

## 💬 Ask a question
In order to ask a question, run a command like:

```shell
python qa.py "Who was the PI for NSF grant about contaminated water?"
```

You can switch out `Who was the PI for NSF grant about contaminated water?` for any question of your liking!

This exposes a chat interface for interacting with a Notion database.
IMO, this is a more natural and convenient interface for getting information.

## 🚀 Code to deploy on StreamLit

The code to run the StreamLit app is in `main.py`. 
Note that when setting up your StreamLit app you should make sure to add `OPENAI_API_KEY` as a secret environment variable.

## 🧑 Instructions for ingesting your own dataset

Export your dataset in xml format.

Run the following command to ingest the data.

```shell
python ingest.py
```

Boom! Now you're done, and you can ask it questions like:

```shell
python qa.py "Who was the PI for NSF grant about contaminated water?"
```
