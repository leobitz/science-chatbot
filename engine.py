import json
from typing import List
from sentence_transformers import SentenceTransformer, util
import numpy as np
from transformers import pipeline
from summarizer.sbert import SBertSummarizer
import yaml
from top2vec import Top2Vec


class SlidingWindowDoc2Vec:
    """
    Document encoder class that uses chunking strategy to manage a very long document

    With the given window size and stride, it convolves on the document and encode each window.
    Then the average of the encoded windows is taken as a final document encoding.

    window_size: the window or chunk size which will be used to convolve the document
    stride: the stride which will be used to move the window

    """
    def __init__(self, encoder, window_size=256, stride=128) -> None:
        self.window_size = window_size
        self.stride = stride
        self.encoder = encoder
    
    def encode(self, text):
        """
        Takes a document and if the number of tokens is more than the window size, 
        it will be chunked into the window size and encoded.

        Inputs: 
            text: str - input string which will be encoded
        Returns:
            numpy.ndarray - vector representing the document
        """
        words = text.split()

        if len(words) <= self.window_size:
            return self.encoder([text])
        
        windows = []
        for i in range(0, len(words) - self.stride, self.stride):
            sub_text = " ".join(words[i:i + self.window_size])
            windows.append(sub_text)

        return self.encoder(windows).mean(axis=0).reshape((1, -1))


class ChatBotEngine:
    """
    Chat bot class
    """
    def __init__(self, 
                document_file_path,
                doc2vec = 'summarize', 
                 explain = False, 
                 rephrase_explain = False, 
                 exp_sentences = 3, 
                 confidence_threshold = 0.5, 
                 max_num_samples = 10,
                 window_size: int = 128, 
                 stride: int = 128,
                 topic_search = True) -> None:
        self.explain = explain
        self.rephrase_explain = rephrase_explain
        self.exp_sentences = exp_sentences
        self.confidence_threshold = confidence_threshold
        self.window_size = window_size
        self.stride = stride
        self.max_num_samples = max_num_samples
        self.topic_search = topic_search

        self.SUMMARIZER_MODEL = SBertSummarizer('paraphrase-MiniLM-L6-v2')
        self.SENTENCE_ENCODER = SentenceTransformer('all-MiniLM-L6-v2')
        self.Q_ANSWERER = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')
        self.REPHRASER = self.SUMMARIZER_MODEL
        self.topic_model = None
        self.CORPUS_EMB = None

        if doc2vec == 'summarize':
            self.doc2vec_encode = self.SENTENCE_ENCODER.encode
        elif doc2vec == 'chunk_mean':
            encoder = SlidingWindowDoc2Vec(self.SENTENCE_ENCODER.encode, self.window_size, self.stride)
            self.doc2vec_encode = encoder.encode
        else:
            raise Exception("doc2vec strategy not specified")

        
        if self.topic_search:
            self.CORPUS_DB, self.topic_model = self.prepare_top_vec(document_file_path, self.max_num_samples)
        else:
            self.CORPUS_DB, self.CORPUS_EMB = self.prepare_doc_encoding(document_file_path, self.max_num_samples)
            

    def bert_summarize_text(self, texts: List[str], max_word_length: int) -> str:
        """
        Given a text, it will summarize it with max maximum word length

        Inputs:
            - texts: list of string 
            - max_word_length: maximum tokens in the summarized text
        Returns:
            - list of summarized texts
        """
        result = self.SUMMARIZER_MODEL(texts, min_length=max_word_length)
        return result

    def prepare_top_vec(self, file_name, num_samples: int):
        """
        Builds the topic level representation of the corpus so that hierarchical search can be applied

        Inputs:
            - file_name: path to the corpus
            - num_samples: number of the samples to include in the corpus
        Returns:
            - list of raw articles
            - topic model 
        """
        file = open(file_name, encoding='utf-8', mode='r')
        raw_articles = []
        
        counter = 0
        while True:

            line = file.readline()
            if len(line) == 0 or counter == num_samples:
                break

            art = json.loads(line)
            passage = " ".join(art['article_text'])
            raw_articles.append(passage)

            counter += 1

        file.close()
        topic_model = Top2Vec(raw_articles, embedding_model="all-MiniLM-L6-v2")
        return raw_articles, topic_model

    def prepare_doc_encoding(self, file_name: str, num_samples: int) -> List:
        """
        Builds a linearly searchable encoding of the corpus

        Inputs:
            - file_name: path to the corpus
            - num_samples: number of the samples to include in the corpus
        Returns:
            - list of raw articles
            - document to vector
        """
        file = open(file_name, encoding='utf-8', mode='r')
        raw_articles = []
        doc_vec = []

        counter = 0
        while True:

            line = file.readline()
            if len(line) == 0 or counter == num_samples:
                break

            art = json.loads(file.readline())
            passage = " ".join(art['article_text'])
            raw_articles.append(passage)

            if len(passage.split()) > self.window_size:
                passage = self.bert_summarize_text(passage, max_word_length=self.window_size)
            
            enc = self.doc2vec_encode(passage)
            doc_vec.append(enc)

            counter += 1

        file.close()
        
        doc_vec = np.vstack(doc_vec)
        return raw_articles, doc_vec


    def search(self, question: str, corpus_emb: np.ndarray,  top_k: int = 5) -> str:
        """
        Searches a question from the corpus embedding
        If the topic model is chosen, then search is hierarchical
        else, linear search will be applied
        Inputs:
            - question: string of the question
            - corpus embedding: the vector representation of each document
        Output:
            - index and score tuple representing the answer document 
                    and confidence score of having the answer
            - question embedding 
        """
        question_emb = self.SENTENCE_ENCODER.encode(question)
        if self.topic_model:
            hits = self.topic_model.query_documents(question, 1)
            top = (hits[2][0], hits[1][0])
        else:
            hits = util.semantic_search(question_emb, corpus_emb, top_k=top_k)
            top = (hits[0][0]['corpus_id'], hits[0][0]['score'])
        
        return top, question_emb

    def answer(self, question: str, context: str) -> str:
        """
        Given the question and context, returns the answer
        """
        result = self.Q_ANSWERER(question=question, context=context)
        return result

    def rephrase_explanation(self, final_context):
        """
        given the final context that the has the answer, returns a summarized explanation of the answer
        """
        new_explanation = self.REPHRASER(final_context, num_sentences=self.exp_sentences)
        return new_explanation

    def search_within_passage(self, question_emb, context, top_k=1):
        """
        A document might very large. Hence, a specific answer can be searched within the
        document in a chunked manner

        Inputs:
            - Question embedding
            - Context or document
        Return:
            - the index and the confidence tuple of search result
            - the chunks
        """
        words = context.split()

        if len(words) <= self.window_size:
            return self.encoder([context])
        
        windows = []
        for i in range(0, len(words) - self.stride, self.stride):
            sub_text = " ".join(words[i:i + self.window_size])
            windows.append(sub_text)

        chunk_emb = self.SENTENCE_ENCODER.encode(windows)
        hits = util.semantic_search(question_emb, chunk_emb, top_k=top_k)
        
        return hits, windows


    def query(self, question: str):
        """
        Given a question, it searches through out the corpus
            - if the answer is not in the corpus, it returns the string 'Sorry, I don't know'
            - if it found the answer, it returns the answer with an explanation
            - explanation should be activated when creating the class
        Inputs:
            - question string
        Returns:
            - answer string
        """
        if question == None or question.strip() == '' or len(question.split()) <= 2:
            return "Please give me a question with at least 3 words"

        top_hit, question_emb = self.search(question, self.CORPUS_EMB)
        # if a document that is similar to the question intent is found
        if top_hit[1] >= self.confidence_threshold: 
            
            context = self.CORPUS_DB[top_hit[0]]
            result = self.answer(question, context)
            
            exp = None

            score, ans = result['score'], result['answer']
            if self.explain:
                exp = context
                if self.rephrase_explain:
                    exp = self.rephrase_explanation(exp)
                ans = f"{ans} \n Explanation: {exp}"
            
            print(score)
            if score > self.confidence_threshold:
                return ans
        
        return "Sorry, I don't know. Please ask me another question" 

if __name__ == "__main__":

    with open('botconfig.yaml', 'r') as file:
        botConfig = yaml.safe_load(file)
        # global chatbot_engine
        chatbot_engine = ChatBotEngine(
            document_file_path=botConfig['corpus_file'],
                doc2vec = 'summarize', 
                    explain = botConfig['include_explanation'], 
                    rephrase_explain = botConfig['rephrase_explanation'],
                    exp_sentences = botConfig['num_explanation_sentence'], 
                    confidence_threshold = botConfig['answer_score_threshold'], 
                    max_num_samples = botConfig['max_corpus_samples'],
                    window_size = botConfig['doc2vec_window_size'],
                    stride = botConfig['doc2vec_stride_size'],
                    topic_search= botConfig['topic_search']
        )

    question = "who decides whether stroke status is correct?"
    ans = chatbot_engine.query(question)

    print(ans)

