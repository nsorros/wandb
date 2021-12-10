VIRTUALENV := venv

virtualenv:
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	python3 -m venv $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip install --upgrade pip
	$(VIRTUALENV)/bin/pip install -r requirements.txt

update-requirements-txt: VIRTUALENV := /tmp/venv
update-requirements-txt:
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	python3 -m venv $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip install --upgrade pip
	$(VIRTUALENV)/bin/pip install -r unpinned_requirements.txt
	echo "# DO NOT EDIT. Automatically generated by make update-requirements-txt" > requirements.txt
	$(VIRTUALENV)/bin/pip freeze | grep -v pkg_resources >> requirements.txt        
