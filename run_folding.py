import argparse
import ray

from radab.tools.folding.fold import folding,align
from radab.tools.folding.base import TaskScanner





@ray.remote(num_gpus=1, num_cpus=0.25)
def run_folding_remote(task):
    funcs = []
    funcs.append(folding)
    for f in funcs:
        task = f(task)
    return task
@ray.remote(num_cpus=0.25)
def run_align_remote(task):
    return align(task)

@ray.remote
def pipeline_fold_align(task):
    funcs = [
        run_folding_remote,
        run_align_remote,
    ]
    for fn in funcs:
        task = fn.remote(task)
    return ray.get(task)
def main():
    ray.init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='')
    args = parser.parse_args()

    final_pfx = 'fold'
    scanner = TaskScanner(args.root, final_postfix=final_pfx)
    while True:
        tasks = scanner.scan()
        futures = [pipeline_fold_align.remote(t) for t in tasks]
        if len(futures) > 0:
            print(f'Submitted {len(futures)} tasks.')
        while len(futures) > 0:
            done_ids, futures = ray.wait(futures, num_returns=1)
            for done_id in done_ids:
                done_task = ray.get(done_id)
                print(f'Remaining {len(futures)}. Finished {done_task.current_path}')
        if len(futures) == 0:
            break
        

if __name__ == '__main__':
    main()