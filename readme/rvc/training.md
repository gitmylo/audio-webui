# RVC training

## General guidelines

### Train for about 2000 computer minutes (AKA minutes of audio on disk)
#### To calculate: you would need (2000/training audio duration) epochs
* If you have 20 minutes of training audio, 100 epochs will usually be enough.
* if you have 10 minutes of training audio, 200 epochs will usually  be enough.

If you still don't understand, refer to this graph. Click the graph to try for yourself:

<a href='https://www.desmos.com/calculator/dtrzof6mjv' target='_blank'>![graph.png](graph.png)</a>

**Green**: Epochs required (y) per x minutes of audio

$f(x)=\frac{2000}{x}$

**Black**: Epochs required (y) per t<sub>minutes</sub> minutes of audio

$f(x)=\frac{2000}{t_{minutes}}>$