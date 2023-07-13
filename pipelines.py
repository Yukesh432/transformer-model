from transformers import pipeline

"""
Pipelines are made of:

a. A tokenizer in charge of mapping raw textual input to token.
b. A model to make predictions from the inputs
c. other optional parameters.....

Since distill bert is a pretrained model , we can not perform sentiment analysis task directly on this raw model
However, we can use this model for either masked language modeling or next sentence prediction.
To do our downstream task , we must first fine tune the model. here the output is shown but it is not recommended to use this way
"""

# pipe= pipeline("text-classification", model= "distilbert-base-uncased", device_map='cuda:0')
# x= pipe("This movie is full of horror and drama!!. I'm very much horrified")
# print(x)


# #using distillbert for masked language modeling
# unmasker= pipeline('fill-mask', model= 'distilbert-base-uncased')
# x= unmasker("Hello i am doing [MASK] in the morning.")
# print(x) 


# # Named entity Recognition
# # Named entity recognition (NER) is a task where the model has to find which parts of the input text correspond to
# # entities such as persons, locations, or organizations
# ner= pipeline("ner", aggregation_strategy= 'simple')
# yy= ner("Susan is studying at tribhuwan university near tirpureshwor")
# print(yy)


# question_answerer = pipeline("question-answering")
# answer= question_answerer(
#     question="Where do I work?",
#     context="My name is Sylvain and I work at Hugging Face in Brooklyn",
# )

# print(answer)

hf_name= "slauw87/bart_summarisation"
summarizer = pipeline("summarization", hf_name, device=0, max_length= 142)
summary= summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""
)
print(summary)