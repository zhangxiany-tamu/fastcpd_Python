PYFILES=$(shell find fastcpd -name "*.py")
PYTESTFILES=$(shell find test -name "*.py")

PYTHON ?= python

.PHONY: venv install install_fastcpd format test test_mac_python_310 test_mac_python_38 test_short run lint generate_stubs_macos_python_310 wheel sdist upload upload_prod paper

venv:
	$(PYTHON) -m venv venv

install:
	$(PYTHON) -m pip install --upgrade pip && \
		$(PYTHON) -m pip install -r requirements_lock.txt

install_fastcpd:
	$(PYTHON) -m pip install . --verbose

format:
	isort $(PYFILES) $(PYTESTFILES) && \
			black $(PYFILES) $(PYTESTFILES)

test: install install_fastcpd
	$(PYTHON) -m pytest $(PYTESTFILES)

test_mac_python_310:
	# Copy bazel out file interface.so to fastcpd/interface.so
	#cp -f bazel-out/darwin_arm64-opt/bin/fastcpd/libinterface.dylib fastcpd/interface.cpython-310-darwin.so
#	cp -rf bazel-out/darwin_arm64-opt/bin/_solib_darwin_arm64 ./
	$(PYTHON) -m pytest $(PYTESTFILES)

test_mac_python_38:
	# Copy bazel out file interface.so to fastcpd/interface.so
	cp -f bazel-out/darwin_arm64-opt/bin/fastcpd/libinterface.dylib fastcpd/interface.cpython-38-darwin.so
	$(PYTHON) -m pytest $(PYTESTFILES)

test_short:
	# Make tmp dir, run tests in there (make sure to activate venv)
	source venv/bin/activate && \
	mkdir -p tmp && \
		cd tmp && \
		cp -R ../test  . && \
		$(PYTHON) -m pytest test -m "not long"
	#$(PYTHON) -m pytest $(PYTESTFILES) -m "not long"
	rm -rf tmp

run: install
	. venv/bin/activate && PYTHONPATH=$(PYTHONPATH) $(PYTHON) main.py

lint: install
	vulture $(PYFILES) $(PYTESTFILES) && \
		$(PYTHON) -m pylint $(PYFILES) $(PYTESTFILES) && \
		mypy $(PYFILES) $(PYTESTFILES)

generate_stubs_macos_python_310:
	cp -f bazel-out/darwin_arm64-opt/bin/fastcpd/libinterface.dylib fastcpd/interface.cpython-310-darwin.so
	. venv/bin/activate  && \
		cd fastcpd && \
		PYTHONPATH=. pybind11-stubgen -o . interface

wheel:
	$(PYTHON) -m build -xn .

sdist:
	$(PYTHON) -m build . --sdist

upload:
	$(PYTHON) -m twine upload --repository testpypi dist/* --skip-existing

upload_prod:
	$(PYTHON) -m twine upload dist/* --skip-existing

paper:
	docker run --rm \
    --volume ./paper:/data \
    --user $(id -u):$(id -g) \
    --env JOURNAL=joss \
    openjournals/inara
