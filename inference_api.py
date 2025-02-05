import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

def get_model():
      
	llm = HuggingFaceEndpoint(
		repo_id=os.getenv("MODEL_ID"),
		task="text-generation",
		max_new_tokens=100,
		do_sample=False
	)

	llm_chat_engine = ChatHuggingFace(llm=llm)
	return llm_chat_engine
