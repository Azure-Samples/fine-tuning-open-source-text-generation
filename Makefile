export HOME := $(HOME)
.ONESHELL:

ifeq ($(OS),Windows_NT)
	SHELL = cmd
	CONDA_ACTIVATE = call %CONDA_PREFIX%\Scripts\activate.bat
else
	SHELL = /bin/bash
	CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
endif

setup_aml:
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


USERPROFILE := $(USERPROFILE)
CURRENT_DIR := $(shell cd)
setup_win:
	if exist %USERPROFILE%\.pyenv rmdir /s /q %USERPROFILE%\.pyenv
	git clone https://github.com/pyenv-win/pyenv-win.git "%USERPROFILE%\.pyenv"
	$(USERPROFILE)\.pyenv\pyenv-win\bin\pyenv --version
	$(USERPROFILE)\.pyenv\pyenv-win\bin\pyenv install 3.12 --skip-existing
	$(USERPROFILE)\.pyenv\pyenv-win\bin\pyenv local 3.12
	python --version
	python -m venv venv
	echo $(CURRENT_DIR)
	"$(CURRENT_DIR)/venv/Scripts/activate"
	pip install poetry
	poetry config virtualenvs.create true
	poetry config virtualenvs.in-project true
	poetry lock
	poetry install
	conda install pip

