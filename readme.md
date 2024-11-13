# NLP-Driven Analysis of Banking Policies 


## About

The 2023 Silicon Valley Bank (SVB) collapse had a strong influence on financial regulations. Our project aims to utilize natural language processing topic modeling techniques to identify primary themes in our corpus of financial regulatory documents scraped from [regulations.gov](https://regulations.gov/). The project objective is to identify and visualize topics, reveal shifts in regulatory focus, and how topics shift with market trends. 

In this project, we compare the texts of proposed and implemented regulations in a 36-month window surrounding the SVB collapse - 18 months pre, and 18 months post. To obtain a baseline comparison and to control for confounding factors, a similar process is done for a comparable financial crisis - the 2008 collapse of Lehman Brothers. 

In speculation, we expect to see increased scrutiny on mid-sized banks vs. the historical focus on Global Systematically Important Banks (G-SIBs). We also expect key themes to include increased capital requirements, liquidity risk, and discussions around the appropriate levels of FDIC deposit insurance. Uncovering the root causes of the SVB collapse and its impact on regulatory trends will allow for better impact mitigation should similar crises arise. 

## Built With

The main packages used in our project:
* pandas
* numpy
* os
* re
* json
* sklearn
* nltk
* gensim
* wordcloud
* matplotlib.pyplot
* openai
* keybert
* rake_nltk
* yake
* spacy
* bertopic
* umap

### Methods 
* Keyword Extraction (RAKE, KeyBERT)
* Topic Modeling (BERTopic, LDA)
* Clustering 
* Visualization Techniques 


### Technologies 
* Python


## Team Members
|Name     |  Handle   | 
|---------|-----------------|
|[Priscilla Clark](https://github.com/priscillaoclark)| @priscillaoclark      |
|[Nicholas Wong](https://github.com/nicwjh)| @nicwjh        |
|[Harsh Kumar](https://github.com/-) |     @-    |
|[Elaine Zhang](https://github.com/-) |     @-    |

## License
MIT License. Project Link: [https://github.com/priscillaoclark/15.S08-applied-nlp-final)

## Acknowledgements
We would like to thank Mike Chen, Andrew Zachary, and Chengfeng Mao for their help and guidance throughout this project.
