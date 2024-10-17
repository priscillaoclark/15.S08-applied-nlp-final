# sample app
from langchain_ollama import OllamaLLM

#model = OllamaLLM(model='llama3.2:1b')
#model = OllamaLLM(model='llama3.1:8b')
model = OllamaLLM(model='llama3.1:70b')

result = model.invoke(input='Hello!')
print(result)