'''
This file is used to define classes and functions related to log
'''
import datetime
import os
import platform
import subprocess
import re
import torch
import argparse
# from pip._internal.utils.misc import get_installed_distributions
import pkg_resources


class Log(object):
    def __init__(self, args: argparse.Namespace):
        if args.save_name == 'None':
            self.dir = os.path.join(args.save_dir, args.dataset_name, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        else:
            self.dir = os.path.join(args.save_dir, args.dataset_name, args.save_name+'_'+datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        self.emb_dir = os.path.join(self.dir, 'embs')
        self.metric_path = os.path.join(self.dir, 'metrics.txt')

        if ~os.path.exists(self.emb_dir):
            os.makedirs(self.emb_dir, exist_ok=True)

        # write git information
        git_revision_short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        with open(self.metric_path, 'a') as f:
            f.write('-----------------git----------------\n')
            f.write('git version: '+git_revision_short_hash+'\n')
            f.write('------------------------------------\n')

        # write hardware information
        hardware_info = self.get_hardware_info()
        with open(self.metric_path, 'a') as f:
            f.write('--------------hardware--------------\n')
            for name, param in hardware_info.items():
                f.write(name+': '+str(param)+'\n')
            f.write('------------------------------------\n')

        # write package information
        # packages = get_installed_distributions()
        packages = [package for package in pkg_resources.working_set]
        with open(self.metric_path, 'a') as f:
            f.write('------packages(python=='+platform.python_version()+')------\n')
            for package in packages:
                f.write(f"{package.project_name}=={package.version}\n")
            f.write('------------------------------------\n')

        # write config
        with open(self.metric_path, 'a') as f:
            f.write('---------------config---------------\n')
            for name, param in args._get_kwargs():
                f.write(name+': '+str(param)+'\n')
            f.write('------------------------------------\n')

    def get_hardware_info(self):
        info = {}

        # get cpu's name
        if platform.system() == "Windows":
            cpu = platform.processor()
        elif platform.system() == "Darwin":
            os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
            command = "sysctl -n machdep.cpu.brand_string"
            cpu = subprocess.check_output(command).strip()
        elif platform.system() == "Linux":
            command = "cat /proc/cpuinfo"
            all_info = subprocess.check_output(command, shell=True).decode().strip()
            for line in all_info.split("\n"):
                if "model name" in line:
                    cpu = str.strip(re.sub(".*model name.*:", "", line, 1))
                    break
        info['cpu'] = cpu

        # get gpu's name
        gpu = torch.cuda.get_device_name()
        info['gpu'] = gpu

        return info

    def info(self, msg):
        with open(self.metric_path, 'a') as f:
            f.write(msg+'\n')

    def save_emb(self, emb: torch.Tensor, name):
        torch.save(emb, os.path.join(self.emb_dir, name+'.pt'))
