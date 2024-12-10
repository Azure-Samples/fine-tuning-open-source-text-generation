export HOME := $(HOME)
.ONESHELL:

SHELL = /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate


setup:
	rm -rf ~/.pyenv
	curl https://pyenv.run | bash
	$(HOME)/.pyenv/bin/pyenv --version
	$(HOME)/.pyenv/bin/pyenv install 3.12 --skip-existing
	$(HOME)/.pyenv/bin/pyenv local 3.12
	python --version
	conda create -n condav0 python=3.12
	$(CONDA_ACTIVATE) activate condav0
	conda install -c conda-forge poetry
	poetry config virtualenvs.create true
	poetry config virtualenvs.in-project true
	poetry lock --no-update
	poetry install
	conda install pip
	conda install ipykernel
	python -m ipykernel install --user --name condav0 --display-name "condav0"