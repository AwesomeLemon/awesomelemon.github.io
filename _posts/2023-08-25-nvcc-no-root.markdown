---
layout: post
title:  "Install nvcc without root"
date:   2023-08-25 17:58:38 +0200
comments: true
---
I needed to install ``nvcc`` on our group server, where I lack root privileges. I found a nice [guide](https://github.com/pyg-team/pytorch_geometric/issues/392#issuecomment-503335625), in this post I will slightly expand on it by explicitly mentioning every step I had to take. Hopefully this will make life easier for future-me and for my colleagues :)

Download a [runfile](https://developer.nvidia.com/cuda-downloads) for your OS:

```
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
```

On our servers there is not enough space in ``/tmp`` for the next step, so:
```
export TMPDIR='/export/scratch2/data/aleksand/tmp'
```

Install toolkit only (assumes that the drivers are already installed)
```
bash cuda_11.7.0_515.43.04_linux.run --silent --override --toolkit --toolkitpath=/export/scratch2/data/aleksand/cuda117
```

Finally, export three paths (you can also add them to ``~/.bashrc``)
```
export LD_LIBRARY_PATH=/export/scratch2/data/aleksand/cuda117/lib64:$LD_LIBRARY_PATH
export PATH=/export/scratch2/data/aleksand/cuda117/bin:$PATH
export CPATH=/export/scratch2/data/aleksand/cuda117/include:$CPATH
```

P.S. My use case (StyleGAN 2) needed gcc (and g++) version < 11, they can be easily installed via conda:
```
conda install -c conda-forge gcc=10.4.0 gxx=10.4.0
```

{% include disqus_comments.html %}