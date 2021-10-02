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
tensorboardX
```

## How to use
We provide an example motion at ```example/run.txt```.  
To visualize the example, you can follow:
``` python
cd SamCon
python test/test_samcon.py
```
If you want to use SamCon on other motion sequences, you can also use the ```test/test_samcon.py```, but you need to make some changes on it.

We use ```tensorboardX``` to quantitatively compare the sampled and reference pose.  
To compare ```example\run.txt``` and reference run motion, you can follow:  
``` python
cd SamCon/example/tensorboardX
tensorboard --logdir ./info
```  
<img src="https://github.com/liangpan-github/SamCon/blob/main/images/tensorboardX.png" width="576" height="432" alt="png"/><br/>  
If you want to perform on your own result, modify and run ```test/draw_curve.py```.


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