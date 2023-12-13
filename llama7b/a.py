from langchain.llms import CTransformers

# llm = CTransformers(model="marella/gpt-2-ggml")
llm= CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',model_type='llama', config={'max_new_tokens':256, 'temperature':0.01})

response=llm("wtf is happing, i'm not having good time. any ideas to make mood better?")
print(response)


# print(llm)