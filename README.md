# SamCon_pybullet
[Pybullet](https://pybullet.org/wordpress/) implementation of [SamCon](http://libliu.info/Samcon/Samcon.html) (SIGGRAPH 2010 paper "Sampling-based Contact-rich Motion Control").  

Humanoid cannot track the reference motion (exhibited in transparency) by using PD controllers.  
<img src="https://github.com/liangpan-github/SamCon/blob/main/images/roll_track_referenceMotion.gif" width="576" height="432" alt="gif"/><br/>

Use SamCon to reconstruct a modified motion which can be tracked by using PD controllers.  
<img src="https://github.com/liangpan-github/SamCon/blob/main/images/roll_track_modifiedMotion.gif" width="576" height="432" alt="gif"/><br/>


## Dependencies
``` python
pybullet
numpy
json
time
math
```

## How to use
``` python
cd SamCon
python test/test_samcon.py
```

## Citation
If you find SamCon_pybullet useful in your research, please cite our repository using the following BibTeX entry.
```
@Misc{pan2021samcon_pybullet,
  author =       {Pan, Liang, et al.},
  title =        {samcon_pybullet - implementing SamCon algorithm using pybullet},
  howpublished = {Github},
  year =         {2021},
  url =          {https://github.com/liangpan-github/SamCon}
}
```

## References
```
@incollection{liu2010sampling,
  title={Sampling-based contact-rich motion control},
  author={Liu, Libin and Yin, KangKang and van de Panne, Michiel and Shao, Tianjia and Xu, Weiwei},
  booktitle={ACM SIGGRAPH 2010 papers},
  pages={1--10},
  year={2010}
}
```
[Zhihu tutorial - SamCon: sampling based controller](https://zhuanlan.zhihu.com/p/58458670)  
[Github repository - kevinxie4c/samcon](https://github.com/kevinxie4c/samcon)