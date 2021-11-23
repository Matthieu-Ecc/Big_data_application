# Big_data_application

## Dependencies management

	pip install poetry
	poetry install
	
Poetry will install all the package store in the .toml

## Poetry function
### pyproject.toml
It contains the list of packages that we need in the virtual environment. 
In fact, it does the same thing as the "requirements.txt" file.

To add a package we will just make a shell command in the root of our project :
	`poetry add [package_name]`
	
And if you need to remove one :
	`poetry remove [package_name]`
### poetry.lock
If you don't want the latest version of your package or just an specific version you have to configure it here.

<br>
<br>

## Naming Convention
### Function
Use a lowercase word or words. Separate words by underscores to improve readability.
`function`,  `my_function`

###  Variable
Use a lowercase single letter, word, or words. Separate words with underscores to improve readability.
`x`,  `var`,  `my_variable`

### Class
Start each word with a capital letter. Do not separate words with underscores. This style is called camel case.
`Model`,  `MyClass`

### Method
Use a lowercase word or words. Separate words with underscores to improve readability.
`class_method`,  `method`

### Constant
Use an uppercase single letter, word, or words. Separate words with underscores to improve readability.
`CONSTANT`,  `MY_CONSTANT`,  `MY_LONG_CONSTANT`

### Module

Use a short, lowercase word or words. Separate words with underscores to improve readability.
`module.py`,  `my_module.py`

### Package

Use a short, lowercase word or words. Do not separate words with underscores.
`package`,  `mypackage`

<br>
<br>

## Starting the project

The first thing you must do after clonning this repository is to create a folder named "data" this will be the place where we save all the data set

	from Big_data_application_folder >> mkdir data

Then copy the __application_train.csv__ and the __application_test.csv__ in the data folder

<br>

### Now that you have your data, before predict any kind of features, you have to prepare and normalize the data

	from Big_data_application_folder >> python Data-prep.py

### Then execute

	from Big_data_application_folder >> python Features-ingeneering.py

### From this point you can choose to retrain the model or to simply use it

	from Big_data_application_folder >> python Training-model.py

## or

	from Big_data_application_folder >> python Predict.py

<br>
<br>

## MLFlow execution

### Navigate throught __mlflow__ folder then choose which model you want to deploy with MLFlow.
### For each model you will find : 
	* conda.yaml >> contain the conda environnement necessary to process the training
	* MLproject >> contain the parameters and specify the environnement
	* train python script >> the python script used for the training phase
### Now you can test the model by simply executing from a command prompt the following command
	python train_{model_name} {parameters}

### To execute the training with MLFlow simply execute this command line from the command prompt : 
	mlflow run . -P {parameters}
	
### Then execute the UI to monitor and track the execution
	mlflow ui 

### Now we are able to serve the model to make prediction as a local REST server 
	mlflow models serve -m mlruns\0\{RUN_ID}\artifacts\model_{name}\ -p 1234
