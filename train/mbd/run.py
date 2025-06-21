import logging
import subprocess
import re
import time
from os import environ
from os.path import abspath, dirname

def runMBD(config=None, output_path=None, args="", ensemble=None, cluster=False, array_path=None, wait=False):
    if config:
        config = abspath(config)
    if output_path:
        output_path = abspath(output_path)
    if cluster:
        _runMBDcluster(config, output_path, args, ensemble, array_path, wait)
    else:
        _runMBDlocal(config, output_path, args, ensemble, array_path, wait)

def _runMBDlocal(config=None, output_path=None, args="", ensemble=None, array_path=None, wait=False):
    executable = "bin/mbd"
    if ensemble:
        executable = "tools/scripts/ensemble.sh"
    if array_path:
        executable = "tools/scripts/array.sh"
    command = [executable, f"-c{config}" if config else "", f"-p{output_path}" if output_path else "", *args.split(), f"-a{abspath(array_path)}" if array_path else "", f"-x{ensemble}" if ensemble else ""]
    command = [x for x in command if x]
    cwd = dirname(abspath(__file__)) + "/../.."
    consoleOut = None if wait else subprocess.DEVNULL
    process = subprocess.Popen(command, cwd=cwd, env=environ, stdout=consoleOut, stderr=consoleOut)
    if wait:
        process.wait()

def _runMBDcluster(config=None, output_path=None, args="", ensemble=None, array_path=None, wait=False):
    command = ["tools/hpc/deploy.sh", f"-c{config}" if config else "", f"-p{output_path}" if output_path else "", *args.split(), "-y", f"-a{abspath(array_path)}" if array_path else "", f"-x{ensemble}" if ensemble else ""]
    command = [x for x in command if x]
    cwd = dirname(abspath(__file__)) + "/../.."
    submit = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if submit.returncode == 0:
        jobid = re.search(r'\d+', submit.stdout.splitlines()[-1]).group()
        logging.warning(f"Job {jobid} deployed")
    else:
        logging.warning(submit.stdout)
        logging.warning(submit.stderr)
        raise OSError("Could not deploy job on cluster. Aborting!")
    if wait:
        while True:
            time.sleep(60)
            jobInQueue = subprocess.run(['tools/hpc/checkJobInQueue.sh', jobid], cwd=cwd, capture_output=True, text=True)
            if jobInQueue.returncode != 0:
                logging.warning(f"Job {jobid} finished")
                return
