# Full waveform inversion with multiple a priori information

This repo provides the code for FWI with multiple a priori information.
For more information, please refer to the Ph.D. thesis:
https://prism.ucalgary.ca/handle/1880/113148
- The Gradient projection methods with inexact projection is provided in Chapter 5.
- For application in FWI please refer to Chapter 6.


## Installation
```
git clone https://github.com/zzar43/FWI_multi_priori
```

## How to use

An Overthrust test data is saved in the folder ``model_data''.

Forward modelling with the true model (parallel computing with 2 workers):
```
julia -p 2 make_data.jl
```
The received signal generated with true model are saved in the folder ``temp_data''.

Perform full waveform inversion with the L-BFGS method:
```
julia -p 2 l-BFGS.jl
```

Perform full waveform inversion with the proposed optimization method:

```
julia -p 2 l-BFGS_CFP.jl
```

The inverse results for each iterations are saved in the folder ``temp_data''.

## Demo

Demo 1:
![Crosswell model](https://github.com/zzar43/FWI_multi_priori/blob/main/demo/cross_well2.jpg)

![Overthrust model](https://github.com/zzar43/FWI_multi_priori/blob/main/demo/overthrust.jpg)