
from time import time
from engine import ChatBotEngine
import yaml


with open('botconfig.yaml', 'r') as file:
    botConfig = yaml.safe_load(file)
    chatbot_engine = ChatBotEngine(
        document_file_path=botConfig['corpus_file'],
        doc2vec = 'chunk_mean', 
        explain = botConfig['include_explanation'], 
        rephrase_explain = botConfig['rephrase_explanation'],
        exp_sentences = botConfig['num_explanation_sentence'], 
        confidence_threshold = botConfig['answer_score_threshold'], 
        max_num_samples = botConfig['max_corpus_samples'],
        window_size = botConfig['doc2vec_window_size'],
        stride = botConfig['doc2vec_stride_size'],
        topic_search= botConfig['topic_search']
    )
    
    text = "What is the difference between a data scientist and a data engineer?"
    response = chatbot_engine.query(text)
    print(response)