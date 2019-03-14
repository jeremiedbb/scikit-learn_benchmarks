import sys
import os
import json
import itertools


def find_results(label1, label2):
    machines = list(os.walk(os.path.join('.', 'results')))[0][1]

    result1 = None
    result2 = None

    for machine in machines:
        path = os.path.join('results', machine)
        for f in os.listdir(path):
            f_path = os.path.join(path, f)
            if f == 'machine.json':
                continue

            with open(f_path, 'r') as result_file:
                result = json.load(result_file)
            label = result['commit_hash']

            if label == label1:
                result1 = f_path
            if label == label2:
                result2 = f_path

    if result1 is None:
        raise(ValueError, "{} not found".format(label1))
    if result2 is None:
        raise(ValueError, "{} not found".format(label2))

    return result1, result2


def compare_to_csv(label1, label2):
    result1, result2 = find_results(label1, label2)

    out_path = label1 + '_' + label2 + '_compared.csv'
    with open(out_path, "w") as out:
        out.write("module,estimator,function,params,"
                  + label1 + ',' + label2 + ',ratio\r\n')

        with open(result1, "r") as result_file:
            result1 = json.load(result_file)['results']
        with open(result2, "r") as result_file:
            result2 = json.load(result_file)['results']
        benchmarks_path = os.path.join('results', 'benchmarks.json')
        with open(benchmarks_path, "r") as bench_file:
            benchmarks = json.load(bench_file)

        for key, bench1 in result1.items():
            bench2 = result2[key]

            module, estimator, func = key.split('.')
            estimator = estimator[:-6]

            param_names = benchmarks[key]['param_names']
            params = list(itertools.product(*benchmarks[key]['params']))

            params_check1 = list(itertools.product(*bench1['params']))
            params_check2 = list(itertools.product(*bench2['params']))
            if not params_check1 == params_check2 == params:
                raise(ValueError, "Benchmarks not run with same parameters "
                                  "can't be compared.")

            for i, param_val in enumerate(params):
                param = '; '.join(['='.join(l)
                                   for l in list(zip(param_names, param_val))])

                res1 = bench1['result'][i]
                res2 = bench2['result'][i]

                ratio = res2 / res1

                line = (module + ',' + estimator + ',' + func + ','
                        + param + ',' + str(res1) + ',' + str(res2) + ','
                        + str(ratio) + '\r\n')

                out.write(line)


if __name__ == '__main__':
    label1, label2 = sys.argv[1:3]

    compare_to_csv(label1, label2)
