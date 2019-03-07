# scikit-learn_benchmarks
Benchmark suite for scikit-learn performances using [airspeed velocity](https://asv.readthedocs.io/en/stable/).

To get started, 

* install miniconda and make sure that the conda command is in your path: `export PATH=${HOME}/miniconda/bin:${PATH}`
* install asv via pip
* clone and move to this repository `git clone https://github.com/jeremiedbb/scikit-learn_benchmarks.git 
* then run `asv run` (see the docs to change the environments config or run only selected benchmarks)
* to publish the results in html format, run `asv publish` and `asv preview`

Special instructions to run the benchmarks with the daal4py patches of scikit-learn:

* create a conda environment with scikit-learn and daal4py installed.
* install asv with `pip install git+git://github.com/jeremiedbb/asv.git@commit-label`.
* edit the `benchmarks/__init__.py` file with these 2 lines:
  ```python
  import daal4py.sklearn
  daal4py.sklearn.patch_sklearn()
  ```
* then run `asv run --python=same --commit-label=<chose_a_label>`.
