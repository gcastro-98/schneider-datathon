# schneider-datathon
Schneider Electric European Hackathon 2022 datathon dedicated repository 

## Competition details

For this challenge, you will have to predict the class of the deforested area, that is between:
 * **Number 0**: 'Plantation'
 * **Number 1**: 'Grassland/Shrubland'
 * **Number 2**: 'Smallholder Agriculture'


### Dataset

The dataset will consist of the following variables of interest. 
They are both in the ``training.csv`` and ``testing.csv`` and can be divided as:
* **Features**:
  * ``latitude``: Where the photo latitude was taken.
  * ``latitude``: Where the photo longitude was taken.
  * ``year``: Year, in which the photo was taken.
  * ``example_path``: Path where the sample image is located.

* **Labels**:
  * label: In this column you will have the following categories:
    * 'Plantation':Encoded with **number 0**, Network of rectangular plantation blocks, connected by a well-defined road grid. In hilly areas the layout of the plantation may follow topographic features. In this group you can find: Oil Palm Plantation, Timber Plantation and Other large-scale plantations.
    * 'Grassland/Shrubland': Encoded with **number 1**, Large homogeneous areas with few or sparse shrubs or trees, and which are generally persistent. Distinguished by the absence of signs of agriculture, such as clearly defined field boundaries.
    * 'Smallholder Agriculture': Encoded with **number 2**, Small scale area, in which you can find deforestation covered by agriculture, mixed plantation or oil palm plantation.

### Evaluation

The evaluation will be taken into consideration the following:
* **100/1200:(DOC)** Brief presentation explaining what you have done and how you have done it.
* **700/1200:(OBJECTIVES)** This will be obtained from the f1-score(macro) of the predictive model. Comparing the predictions your model has made about versus ground truth.
* **400/1200:(QUALITY)** Code quality and automation, complexity, maintainability, reliability, and security.



### Delivery requirements:
Once completed, we have to submit the repository project, which must contain at least these 4 files:

* main.py or main.ipynb: main script of the program, ``run.py`` in our case
* predictions.csv: ``.csv`` submission file containing ``test_index`` and ``label`` as columns
* predictions.json: same but in ``.json`` format (use ``pandas.DataFrame.to_json`` method)
* presentation.pdf: 4 slides max. explaining the solution

## Code execution
In order to launch the code for the predictive model, you have to type:
```commandline
python -m fpds.run
```

## Local environment

```commandline
conda create -n schneider python=3.9
conda activate schneider
conda install -c anaconda tensorflow==2.9.0 -y 
conda install -c anaconda sphinx numpydoc \
    sphinx_rtd_theme recommonmark python-graphviz -y
pip install --upgrade myst-parser
```

## Documentation

Whenever the modules have been updated, the documentation can be re-generated 
from the ``docs`` folder by typing ():
```console
(<condaenv>) schneider-datathon/docs $ make html
```

Then, opening the ``schneider-datathon/docs/build/html/index.html`` file 
in the browser will display the generated documentation.