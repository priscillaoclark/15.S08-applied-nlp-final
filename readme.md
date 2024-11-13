# NLP-Driven Analysis of Banking Policies 

[![MIT License][license-shield]][license-url]


<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="https://github.com/priscillaoclark/15.S08-applied-nlp-final/blob/main/figures/project_logo.jpg" alt="Logo" width="150" height="100">
  </a>

  <h3 align="center">NLP-Driven Analysis of Banking Policies</h3>

  <p align="center">
    Topic Modeling of Regulatory Documents using Natural Language Processing
    <br />
    <br />
  </p>
</p>



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

## Getting Started

The corpus of documents used in our project can be accessed in our documents folder. Implementation and data preprocessing is done in each respective python script file within the keywords folder. 

### Prerequisites

- [pandas](https://pandas.pydata.org/) (A data manipulation and analysis library providing data structures like DataFrames for Python.)
- [numpy](https://numpy.org/) (A library for numerical computing in Python, providing support for large, multi-dimensional arrays and matrices.)
- [scikit-learn](https://scikit-learn.org/) (A machine learning library for Python, offering tools for classification, regression, clustering, and dimensionality reduction.)
- [nltk](https://www.nltk.org/) (The Natural Language Toolkit, a platform for building Python programs to work with human language data.)
- [gensim](https://radimrehurek.com/gensim/) (A library for topic modeling and document similarity analysis in Python.)
- [bertopic](https://github.com/MaartenGr/BERTopic) (A topic modeling library that leverages BERT embeddings for creating interpretable topics.)
- [PyTorch](https://pytorch.org/) (An open-source machine learning framework for deep learning.) 
- [scipy](https://scipy.org/) (A library for scientific computing in Python.)

## Usage

Each of the python script files serve separate purposes and can be used to keyword extraction or to topic model our corpus of regulatory documents. Sample visualization can be found in the figures folder. 


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


[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://opensource.org/licenses/MIT

