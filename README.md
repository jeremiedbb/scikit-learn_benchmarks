# scikit-learn_benchmarks

Benchmark suite for scikit-learn performances using [airspeed velocity](https://asv.readthedocs.io/en/stable/).

To get started, 

* Create a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) and activate it and install pip in it (you can get conda by [installing miniconda](https://docs.conda.io/en/latest/miniconda.html)).

* Install asv (airspeed velocity):

```
pip install asv
```

* Clone this repo:

```
git clone https://github.com/jeremiedbb/scikit-learn_benchmarks.git
cd scikit-learn_benchmarks
```

* To run the full benchmark suite (warning: this can take several hours) use the following (see the docs to change the environments config or run only selected benchmarks):

```
asv run -b _bench
```

* To run the benchmarks for a specific model, use the `-b <modelname>` flag, e.g.:

```
asv run -b KMeans
```

* You can configure the benchmarks by editing the `benchmark/config.json` file.


* To publish the results in html format, run `asv publish` and `asv preview`

## Special instructions to run the benchmarks with the [daal4py patches of scikit-learn](https://github.com/IntelPython/daal4py/blob/master/doc/sklearn.rst):

* Create a conda environment with scikit-learn, joblib, pillow and daal4py installed.
* Install asv with `pip install git+git://github.com/jeremiedbb/asv.git@commit-label` ([see this PR](https://github.com/airspeed-velocity/asv/pull/794))
* To benchmark scikit-learn vanilla, run:

```
asv run --python=same --commit-label=vanilla_sklearn -b _bench
```

* To benchmark scikit-learn with the patches, first edit the `benchmarks/__init__.py` file with:

```
import daal4py.sklearn
daal4py.sklearn.patch_sklearn()
```

* Then run:

```
asv run --python=same --commit-label=daal4py_sklearn -b _bench
```

* Finally to compare both benchmarks, run:

```
asv compare vanilla_sklearn daal4py_sklearn
```
