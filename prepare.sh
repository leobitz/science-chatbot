RUN pip install -r requirements.txt
RUN sudo apt-get install unzip
RUN wget https://archive.org/download/armancohan-long-summarization-paper-code/pubmed-dataset.zip
RUN unzip pubmed-dataset.zip -d pubmed-dataset
