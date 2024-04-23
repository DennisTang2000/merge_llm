#!/usr/bin/python3

import os

import pathlib

 

slurm_jobid = os.getenv('2187428')

cgroup_path = pathlib.Path(f'/sys/fs/cgroup/system.slice/slurmstepd.scope/job_{slurm_jobid}/')

memory_current = cgroup_path / 'memory.current'

memory_max = cgroup_path / 'memory.max'

 

def print_memusage():

    print("Current Memory in G", int(memory_current.read_text())/1024**3)

    print("Percent Used", round(int(memory_current.read_text())/int(memory_max.read_text()), 2))

 

print_memusage()

one_gb = bytearray(1024*1024*1000)

print_memusage()