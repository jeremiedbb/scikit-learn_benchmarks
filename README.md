# scikit-learn_benchmarks
Benchmark suite for scikit-learn performances using [airspeed velocity](https://asv.readthedocs.io/en/stable/).

To get started, 

* install miniconda and make sure that the conda command is in your path: `export PATH=${HOME}/miniconda/bin:${PATH}`
* install asv via pip
* clone and move to this repository `git clone https://github.com/jeremiedbb/scikit-learn_benchmarks.git 
* then run `asv run` (see the docs to change the environments config or run only selected benchmarks)
* to publish the results in html format, run `asv publish` and `asv preview`