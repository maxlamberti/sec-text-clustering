# Topic Modelling SEC Text Data
 📄 Clustering public companies based on their SEC report filings

## Description ##

Companies are often sorted into sectors (retail, tech, ...) according to some accepted classification standard (SIC, GICS, ...).  
Historically, companies within the same sector or product space exhibit correlation, ie.
they tend to move in tandem in reaction to market shocks. However, traditional classification standards might be inefficient 
in capturing similarities between companies or products. This project prototypes an unsupervised NLP 
approach to measuring similarity between companies by performing topic modelling on yearly 10-k report 
filings submitted by each company to the SEC. These filings lay out a description of the company, its markets and 
products and hence are well suited for the task. 

The prototype results are located in the python notebook. An HTML export is provided as well. 

## Get Set Up Locally

The main code is located in the jupyter notebook, auxiliary code is in the utils folder. 
To download and get set up clone the repository and install the python dependencies.

```
$ git clone https://github.com/hexamax/sec-text-clustering
$ pip install -r requirements.txt
```

