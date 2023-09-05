## **Simple Scientific Q&A Chatbot**
A powerful tool designed to navigate the vast world of scientific literature and provide concise and accurate answers to your science-related questions. This chatbot is equipped with  a document summarizer, a fast retrieval mechanism, question answerer, and a context rephrasing model. With access to the extensive Pubmed Scientific Dataset and the potential for future fine-tuning on real scientific conversations, this chatbot is a valuable resource for anyone seeking scientific knowledge. Its adaptable hyperparameters allow for customization to suit your specific needs, whether you want detailed explanations or quick answers. 

Demo: [Jupyter Notebok](https://github.com/leobitz/science-chatbot/blob/main/demo.ipynb)

You can use Docker or just run *prepare.sh* to handle all the dependency issues.

## Architecture

The architecture includes the following major components:

- **Document Encoder** - Documents have different sizes and each document can be represented with a fixed-size vector. I used two options
  - BERT Summarizer: Summarize any text to a fixed token size text (256) and apply sentence encoder (BERT-Sentence Encoder)
  - Chunking: Apply the BERT sentence encoder on fixed-size chunks of the document and finally average them to get the final fixed-size representation

Chunking is faster since it's easier to process short-length documents than long sizes in Transformers

- **Document Retrieval -** Given a question, a search operation is applied to the corpus to find the right context. I implemented two methods
  - **Linear Search -** on the document vector space. Very slow
  - **Hierarchical Search** - Using Topic2Vec, we can build a tree and search on clusters rather than search on an array

- **Question Answerer -** Once the right context is found, a seq2seq model can be fed the question and the context. It will produce the answer.

- **Rephraser -** A small explanation can be provided along with the answer. As such, the context can be rephrased to a small text with a specified amount of sentences

## Dataset

[Pubmed Scientific Dataset](https://huggingface.co/datasets/scientific_papers) - I used this database. It has 119k samples on medical topics

## Improvements

Several improvements can be made to make the system more useable

- Extract scientific conversation from the [Stanford SHP dataset](https://huggingface.co/datasets/stanfordnlp/SHP) and fine-tune the whole system using RLHF. This will make answers more humanly
- Using the same backbone model for encoding documents, summarization, and answer inference. This will result in a massive memory footprint reduction
- A context might be long and answer generation inference can take longer time. Such an operation can be minimized using chunked search. I have applied that already, but the performance is not that great. With the right hyperparameter (window size) search, inference can be boosted.


## Hyperparameters

Please change the attributes in [YAML Config](https://github.com/leobitz/sci-bot/blob/main/botconfig.yaml)

- **corpus_file**: pubmed-dataset/pubmed-dataset/train.txt

- **max_corpus_samples**:  maximum samples of the corpus. if -1, then all samples will be taken. (Default: -1)

- **doc2vec_window_size**: a chunk will have 256 tokens. This is used to create the document encoding if **topic_search** is not activated. (Default: 256)
- **doc2vec_stride_size**: a window will slide with stride. (Default: 128)
- **include_explanation**: Boolean value whether the model should  explain the answer or not. (Default: True)
- **rephrase_explanation**: Boolean value whether the model should rephrase the explanation or give me raw context or not. (Default: True)
- **num_explanation_sentence**: maximum number sentences for explanation. (Default: 2)
- **answer_score_threshold**: confidence level for the model about the answer. if model's confidence is more than this value, it will answer it. If not, then it will reply *Sorry, I don't know*. (Default: 0.5)
- **topic_search**: Use hierarchical instead of linear search (Default: True)