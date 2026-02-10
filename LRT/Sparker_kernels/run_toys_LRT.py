import os, json, argparse, glob, time, datetime
import numpy as np
import os.path

#### launch python script ###########################                                                                                                                              
if __name__ == '__main__':
    parser   = argparse.ArgumentParser()
    parser.add_argument('-p','--pyscript', type=str, help="name of python script to execute", required=True)
    parser.add_argument('-f','--folderpath', type=str, help="name of the folder", required=True)
    parser.add_argument('-c', '--calibration', type=int, help="is it a calibration toy", required=True)
    parser.add_argument('-l','--local',    type=int, help='if to be run locally',             required=False, default=0)
    parser.add_argument('-t', '--toys',    type=int, help="number of toys to be processed",   required=False, default=100)
    parser.add_argument('-s', '--firstseed', type=int, help="first seed for toys (if specified the the toys are launched with deterministic seed incresing of one unit)", required=False, default=-1)
    args     = parser.parse_args()
    ntoys    = args.toys
    pyscript = args.pyscript
    firstseed= args.firstseed
    folderpath = args.folderpath
    calibration = args.calibration
    ntest = 2000
    nensemble = 10
    
    pyscript_str = pyscript.replace('.py', '')
    pyscript_str = pyscript_str.replace('_', '/')
    if firstseed<0: seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
    else: seed = firstseed
    if args.local:
        for i in range(ntoys):
            seed+=1
            os.system("python %s/%s -f %s -n %i -e %i -s %i -c %i"%(os.getcwd(), pyscript, folderpath, ntest, nensemble, seed, calibration))
    else:
        label = "logs"
        os.system("mkdir %s" %label)
        for i in range(ntoys):
            seed+=1
            script_sbatch = open("%s/submit_%i.sh" %(label, seed) , 'w')
            script_sbatch.write("#!/bin/bash\n")
            script_sbatch.write("#SBATCH -c 1\n")
            script_sbatch.write("#SBATCH --gpus 1\n")
            script_sbatch.write("#SBATCH -t 0-0:10\n")
            script_sbatch.write("#SBATCH -p iaifi_gpu_priority\n")
            script_sbatch.write("#SBATCH --mem=10000\n")
            script_sbatch.write("#SBATCH -o ./logs/%s"%(pyscript_str)+"_%j.out\n")
            script_sbatch.write("#SBATCH -e ./logs/%s"%(pyscript_str)+"_%j.err\n")
            script_sbatch.write("\n")
            script_sbatch.write("module load python/3.10.9-fasrc01\n")
            script_sbatch.write("\n")
            script_sbatch.write("python %s/%s -f %s -n %i -e %i -s %i -c %i"%(os.getcwd(), pyscript, folderpath,ntest,nensemble,seed, calibration))
            script_sbatch.close()
            os.system("chmod a+x %s/submit_%i.sh" %(label, seed))
            os.system("sbatch %s/submit_%i.sh"%(label, seed) )
