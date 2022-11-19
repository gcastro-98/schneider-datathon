# schneider-datathon
Schneider Electric European Hackathon 2022 datathon dedicated repository 

## Competition details

### Dataset

The dataset will consist of the following variables of interest:
* **Features**:
  * Images dataset:
* **Labels**:
  * Labels .csv with binary value (LabelEncoded)

### Evaluation

The evaluation will be taken into consideration the following:

* 800/1200:(**OBJECTIVES**) This will be obtained from the f1-score of the 
predictive model. Comparing the predictions your model has made about test_x versus ground truth.
* 100/1200:(**OBJECTIVES**) If you are able to automate the extraction of data 
from PDFs and incorporate them into the training of the model. 100 points will be awarded.
* 100/1200:(**DOC**) Brief presentation explaining what you have done and how you have done it.
* 200/1200:(**QUALITY**) Code quality and automation/industrialisation.

### Delivery requirements:
Once completed, we have to submit the repository project, which must contain at least these 4 files:

* main.py: main script of the program, ``run.py`` in our case
* predictions.csv: ``.csv`` submission file containing ``test_index`` and ``label`` as columns
* predictions.json: same but in ``.json`` format (use ``pandas.DataFrame.to_json`` method)
* presentation.pdf: 4 slides max. explaining the solution

## Code execution
In order to launch the code for the predictive model, you have to type:
```commandline
python -m fpds.run
```

## Local environment

conda create -n schneider python=3.9
conda activate schneider
conda install -c anaconda tensorflow==2.9.0 -y 
conda install -c anaconda sphinx numpydoc \
    sphinx_rtd_theme recommonmark python-graphviz -y
pip install --upgrade myst-parser

## Documentation

Whenever the modules have been updated, the documentation can be re-generated 
from the ``docs`` folder by typing ():
```console
(<condaenv>) schneider-datathon/docs $ make html
```

Then, opening the ``schneider-datathon/docs/build/html/index.html`` file 
in the browser will display the generated documentation.