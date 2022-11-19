# schneider-datathon
Schneider Electric European Hackathon 2022 datathon dedicated repository

## Code execution
In order to launch the code for the predictive model, you have to type:
```commandline
python -m fpds.run
```

Bear in mind that if the code is running locally, the ``LOCAL`` global
variables in the modules have to keep as ``True``; while in case

It also assumes that ``data`` exists with the default structure:
* 

### Other relevant links

In the following links there are interesting resources:
* [EfficientNet notebook](https://www.kaggle.com/code/sergibechsala/baseline-pretrained-cnn-4d0b11)
* [Transformer (ViT) notebook](https://www.kaggle.com/goodieml/baseline-kaggle-fine-tuning)
* [Models' serialization]()

## Local environment

```bash
conda create -n schneider python=3.9
conda activate schneider
conda install -c anaconda tensorflow -y
conda install -c conda-forge keras matplotlib transformers -y
pip install vit-keras
conda install -c anaconda seaborn pillow -y
conda install -c esri tensorflow-addons -y
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