import pandas as pd
import numpy as np
import argparse
import os, errno, sys
from joblib import Parallel, delayed


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(
            description="An analyst for quick ML applications.",
            add_help=False)
    parser.add_argument('INPUT_DATA', type=str,
                        help='Data file to analyze; ensure that the '
                        'target/label column is labeled as "class".')    
    parser.add_argument('PROTECTED', type=str,
                        help='protected attributes file')    
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-ml', action='store', dest='LEARNERS',
            default=None,type=str, 
            help='Comma-separated list of ML methods to use (should correspond'
            ' to a py file name in learners/)')
    parser.add_argument('--lsf', action='store_true', dest='LSF', 
            default=False, help='Run on an LSF HPC (using bsub commands)')
    parser.add_argument('-n_jobs',action='store',dest='N_JOBS',default=1,
            type=int, help='Number of parallel jobs')
    parser.add_argument('-seeds',action='store',dest='SEEDS',default='',
            type=str, help='specific trial numbers comma-separated')
    parser.add_argument('-label',action='store',dest='LABEL',default='class',
            type=str,help='Name of class label column')
    parser.add_argument('-results',action='store',dest='RDIR',
            default='results',type=str,help='Results directory')
    parser.add_argument('-q',action='store',dest='QUEUE',
            default='epistasis_normal',type=str,help='LSF queue')
    parser.add_argument('-m',action='store',dest='M',default=4096,type=int,
            help='LSF memory request and limit (MB)')

    args = parser.parse_args()
      
    learners = [ml for ml in args.LEARNERS.split(',')]  # learners
    print('learners:',learners)

    model_dir = 'ml'

    dataset = args.INPUT_DATA.split('/')[-1].split('.')[0]
    print('dataset:',dataset)

    # results_path = '/'.join([args.RDIR, dataset]) + '/'

    # make the results_path directory if it doesn't exit 
    try:
        os.makedirs(args.RDIR)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if len(args.SEEDS)>0:
        seeds = args.SEEDS.split(',')
        N_SEEDS = len(seeds)
    else:
        seeds = [42]
        N_SEEDS = 1

    # write run commands
    all_commands = []
    job_info=[]
    for t in range(N_SEEDS):
        # random_state = np.random.randint(2**15-1)
        random_state = seeds[t]
        # print('random_seed:',random_state)
        
        for ml in learners:
            # save_file = results_path + '/' + dataset + '_' + ml + '.csv'  
            
            all_commands.append(
            'python {ML}.py {DATASET} {ATTS} {RS} {RDIR}'.format(
                                  ML=ml,
                                  DATASET=args.INPUT_DATA,
                                  ATTS=args.PROTECTED,
                                  RDIR=args.RDIR, 
                                  RS=random_state))
            job_info.append({
                'ml':ml,
                'dataset':dataset,
                'results_path':args.RDIR,
                'seed':random_state
                })

    if args.LSF:    # bsub commands
        for i,run_cmd in enumerate(all_commands):
            job_name = (
                    job_info[i]['ml'] + '_' 
                    + job_info[i]['dataset'] + '_' 
                    + job_info[i]['seed']
                    )
            out_file = job_info[i]['results_path'] +'/'+ job_name + '_%J.out'
            error_file = out_file[:-4] + '.err'

            # choose uniformly among queues if more than one available
            if ',' in args.QUEUE:
                queue = np.random.choice(args.QUEUE.split(','))
            else: 
                queue = args.QUEUE
            
            bsub_cmd = ('bsub -o {OUT_FILE} -n {N_CORES} -J {JOB_NAME} '
                    '-q {QUEUE} -R "span[hosts=1] rusage[mem={M}]" -M {M} '
                    '').format(
                               OUT_FILE=out_file,
                               JOB_NAME=job_name,
                               QUEUE=queue,
                               N_CORES=args.N_JOBS,
                               M=args.M)
            
            bsub_cmd +=  '"' + run_cmd + '"'
            print(bsub_cmd)
            os.system(bsub_cmd)     # submit jobs 

    else:   # run locally  
        for run_cmd in all_commands: 
            print(run_cmd) 
        Parallel(n_jobs=args.N_JOBS)(delayed(os.system)(run_cmd) 
                for run_cmd in all_commands )
