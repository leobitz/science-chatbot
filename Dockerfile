FROM pytorch/pytorch:latest

RUN mkdir app
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

RUN sudo apt-get install unzip
RUN wget https://archive.org/download/armancohan-long-summarization-paper-code/pubmed-dataset.zip
RUN unzip pubmed-dataset.zip -d pubmed-dataset

EXPOSE 5000
CMD ["sh","start.sh"]