* Submit the job below
```
#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=72:00:00
#PBS -q k40
#PBS -N bionoi_autoencoder
#PBS -A loni_bionoi01
#PBS -j oe

sleep 100000000000
```

* ssh into the given node
```
ssh node
```


* use the following commands

```
1. mkdir -p /var/scratch/jfeins1/
    * makes the directory to work from
2. singularity shell -B /project,/work --nv /home/admin/singularity/pytorch-1.0.0-dockerhub-v3.simg

    * initializes a singularity shell that will allow pytorch to be run on an os that cannot run pytorch otherwise
    * '-B /directory0, /directory1' binds the directories to the singularity shell, allowing the user to access all files within while inside the shell

3. unset PYTHONPATH

4. unset PYTHONHOME
    * 2 commands that remove any prior python paths so that the virtual environment is used
    * must be done everytime before an environment is created or activated

5. conda create -n env_name python=3.7
    * creates a virtual conda environment named 'pytorch.' can be skipped if an environment has already been made

6. source activate env_name
    * activates the virtual conda environment. depending on the conda version, 'source' may be replaced by 'conda'

7. conda install pytorch torchvision cudatoolkit=9.0 -c env_name
    * installs pytorch and torchvision into the environment.
    * steps 5 and 7 can be skipped if a conda virtual environment has already been created with pytorch installed

8. export LD_LIBRARY_PATH=/usr/local/onnx/onnx:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/usr/lib64:/.singularity.d/libs
    * fixes an error: "undefined symbol: _ZTIN2at11TypeDefaultE....."

9. cd /work/user/resnet
    * cd into directory containing the resnet codes

10. stdbuf -o0 python freeze_layers_train.py -tar_extract_path /var/scratch/jfeins1/ -batch_size 512 -epoch 100 -learning_rate 0.001 > progress-optimizer-randomweightedsamlper.log 2>&1
    * an example command to run a resnet file. 
    
```

* You must have a .condarc file in your home directory with the paths to your anaconda environments and packages.  Note that .condarc is a hidden file. Must be formatted as below.
```
envs_dirs:
- /work/user/anaconda3/envs
pkgs_dirs:
- /work/user/anaconda3/pkgs 
```