from glob import glob
import os
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Submit long jobs.",
                                     add_help=False)
    parser.add_argument('-ml',action='store',dest='mls', type=str, 
            default='gerryfair_ml,gerryfair_xgb,feat,feat_tourn,feat_flex2,'
            'feat_nsga2,feat_flex2_nsga2,feat_random_p100_g0')
    parser.add_argument('--long',action='store_true',dest='LONG', default=False)
    parser.add_argument('-seeds',action='store',type=str,dest='SEEDS', 
            default='14724,24284,31658,6933,1318,16695,27690,8233,24481,6832,'
            '13352,4866,12669,12092,15860,19863,6654,10197,29756,14289,'
            '4719,12498,29198,10132,28699,32400,18313,26311,9540,20300,'
            '6126,5740,20404,9675,22727,25349,9296,22571,2917,21353,'
            '871,21924,30132,10102,29759,8653,18998,7376,9271,9292')
    parser.add_argument('-datasets',action='store',type=str,dest='datasets', 
            default='student,adult,lawschool,communities')
    parser.add_argument('-results',action='store',dest='RDIR',
            default='results/',type=str,help='Results directory')
    parser.add_argument('-n_trials',action='store',dest='N_TRIALS',default=50,
            type=int, help='Number of trials to run')
    parser.add_argument('-n_jobs',action='store',dest='N_JOBS',default=1,
            type=int, help='Number of parallel jobs')
    parser.add_argument('-m',action='store',dest='M',default=8000,type=int,
                        help='LSF memory request and limit (MB)')
    parser.add_argument('--lsf', action='store_true', dest='LSF', 
            default=False, help='Run on an LSF HPC (using bsub commands)')
    args = parser.parse_args()

    n_trials = len(args.SEEDS) if args.N_TRIALS < 1 else args.N_TRIALS
    seeds = ','.join(args.SEEDS.split(',')[:n_trials])
    print('running these datasets:',args.datasets)
    print('using these seeds:',seeds)
    print('and these methods:',args.mls)
    if args.LONG:
        q = 'epistasis_long,mooreai_long'
    else:
        # q = 'epistasis_normal,epistasis_normal,epistasis_normal,mooreai_normal'
        q = 'epistasis_normal'

    lpc_options = '' if not args.LSF else '--lsf -q {Q} -m {M} -n_jobs {NJ}'.format(
            Q=q, 
            M=args.M, 
            NJ=args.N_JOBS)


    for f in args.datasets.split(','):
        dataset = 'GerryFair/dataset/'+f+'.csv'
        attributes = 'GerryFair/dataset/'+f+'_protected.csv'

        jobline =  ('python analyze.py {DATA} {DATA_PROTECTED} '
                   '-ml {ML} -results {RDIR} -seeds {T} {LPC}'
                   ).format(DATA=dataset,
                            DATA_PROTECTED=attributes,
                            LPC=lpc_options,
                            ML=args.mls,
                            RDIR=args.RDIR,
                            T=seeds
                            )
        print(jobline)
        os.system(jobline)
