# SamCon

[PyBullet](https://github.com/bulletphysics/bullet3) implementation of [SamCon](http://libliu.info/Samcon/Samcon.html) (SIGGRAPH 2010 paper "Sampling-based Contact-rich Motion Control").  

SamCon is an easy-to-understand method for physics-based character animation.  
This repository has these following features:  
- Core parts of SamCon: (1) samples generation, simulation and evaluation, (2) elite samples selection, (3) final best trajectory searching.  
- Humanoid URDF with similar topology to [SMPL](https://smpl.is.tue.mpg.de/), which means we can get reference motion from a large-scale human motion dataset [AMASS](https://amass.is.tue.mpg.de/).
- Scripts for generating reference motion from AMASS.
- Master-worker parallelization (tested on windows & ubuntu).  

Hope this repo can contribute to the physics-based character animation community. :innocent:  

## What's SamCon?

Given a reference motion (see the transparent humanoid), directly feeding it to PD controllers will make the simulated humanoid fall down.  
<img src="https://github.com/liangpan99/SamCon_dev/blob/main/data/images/cartwheel_raw.gif" width="576" height="324" alt="gif"/><br/>

Running SamCon to optimize the reference motion, we can get a corrected motion. Tracking it again, the simulated humanoid will behave like the reference motion.  
<img src="https://github.com/liangpan99/SamCon_dev/blob/main/data/images/cartwheel_samcon.gif" width="576" height="324" alt="gif"/><br/>

## Getting Started

### Installation
``` python
git clone https://github.com/liangpan99/SamCon.git
pip install -r requirements.txt
```
code tested on windows & ubuntu  

### Reference motion
We use a large-scale 3D human motion dataset, i.e. [AMASS](https://amass.is.tue.mpg.de/), as the reference motion database.  

To use pre-processed reference motion **(only contains 252 sequences of "ACCAD")**, download from [google drive](https://drive.google.com/file/d/1hf2NKmZMyX1jnfegwJIZPQzbNxogvk3J/view?usp=share_link) and place in the directory ```./data/motion/```.  

To manually generate reference motion from AMASS, follow the following instructions:  
1. download AMASS dataset, and organize it into the following structure:  
```
AMASS
|   - ACCAD
|         - Female1General_c3d
|                            - A1 - Stand_poses.npz
|                            - A2 - Sway_poses.npz
|                            - A2 - Sway t2_poses.npz
|                            - ...
|         - Female1Gestures_c3d
|         - Female1Running_c3d
|         - ...
|   - BioMotionLab_NTroje
|   - BMLhandball
|   - ...
```  

2. download ```amass_copycat_occlusion.pkl``` from [google drive](https://drive.google.com/uc?id=1ZAHbM3iYe1Wq0ShTGeDpFhaZQAqhxHQb) and place in the directory ```./data/motion/```. It's an annotation file that help avoid invalid motion sequences in AMASS, such as sitting on a chair, provided by [Kin-Poly](https://github.com/KlabCMU/kin-poly).  

3. change ```amass_dir``` & ```sequences``` variables in file ```process_amass_raw.py```, one indicates the path to AMASS dataset and the other indicates sequences chosen to process. Then, run:  
``` python
python process_amass/process_amass_raw.py
python process_amass/amass_to_bullet.py
```

4. visualize reference motion:  
``` python
python process_amass/vis_motion.py
```
It will ask you to input a sequence name, all names are lied in ```./data/motion/all_seq_names.txt```, you can pick one. Press [Q] to change sequence.  

<img src="https://github.com/liangpan99/SamCon_dev/blob/main/data/images/reference_motion.gif" width="576" height="324" alt="gif"/><br/>

### Examples
We provide two configs (walk & cartwheel) and corresponding results to show you how to use this repo. All hyper-parameters are lied in ```.yml``` config, e.g. nIter, nSample, nSave and so on. Note that, because code requires a lot of disk I/O, **please set ```tmp_dir``` to SSD disk for speeding up.**
#### Run SamCon
``` python
python scripts/run_samcon.py --cfg walk --num_processes 8
python scripts/run_samcon.py --cfg cartwheel --num_processes 8
```
maximum number of num_processes is equal to your computer's CPU cores


#### Evaluate SamCon
``` python
python scripts/eval_samcon.py --cfg walk --file "walk_ACCAD_Female1Walking_c3d_B12 - walk turn right (90)_poses.pkl"
python scripts/eval_samcon.py --cfg walk --file "walk_ACCAD_Male1Walking_c3d_Walk B10 - Walk turn left 45_poses.pkl"
python scripts/eval_samcon.py --cfg cartwheel --file "cartwheel_ACCAD_Female1Gestures_c3d_D6- CartWheel_poses.pkl"
```
You can find cost distribution images in the directory ```./results/samcon/cfg_name/info/```.
<img src="https://github.com/liangpan99/SamCon_dev/blob/main/results/samcon/walk/info/walk_ACCAD_Female1Walking_c3d_B12%20-%20walk%20turn%20right%20(90)_poses_pose_cost.png" width="100%" alt="gif"/><br/>

<img src="https://github.com/liangpan99/SamCon_dev/blob/main/results/samcon/walk/info/walk_ACCAD_Female1Walking_c3d_B12%20-%20walk%20turn%20right%20(90)_poses_total_cost.png" width="100%" alt="gif"/><br/>



## References
Algorithm: [SamCon paper](http://libliu.info/Samcon/Samcon.html) + [Zhihu tutorial](https://zhuanlan.zhihu.com/p/58458670)  
Humanoid URDF: [ScaDiver](https://github.com/facebookresearch/ScaDiver)  
Process AMASS: [Kin-Poly](https://github.com/KlabCMU/kin-poly)  
