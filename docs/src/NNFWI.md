# NNFWI 

**Integrating Deep Neural Networks withFull-waveform Inversion: Reparametrization, Regularization, and UncertaintyQuantification**
 
## Architecture
 
![](./asset/nnfwi/NNFWI.png)
 
## Forward Simulation
 
| Marmousi model  | Inital model   |
| --------------- | -------------- |
| ![](./asset/nnfwi/marmoursi-model.png) | ![](./asset/nnfwi/marmoursi-initial.png) |

### Wavefield snapshot
![](./asset/nnfwi/forward_wavefield.gif)
 
## Inversion based on Automatic Differentiation
 
| noise | Traditional  FWI  | NNFWI             |
| ------| ----------------- | ----------------- |
| $\sigma=0$ | ![](./asset/nnfwi/marmoursi-noise0.png) | ![](./asset/nnfwi/marmoursi-reg-noise0.png) |
| $\sigma=1$ | ![](./asset/nnfwi/marmoursi-noise1.png) | ![](./asset/nnfwi/marmoursi-reg-noise1.png) |
| $\sigma=2$ | ![](./asset/nnfwi/marmoursi-noise2.png) | ![](./asset/nnfwi/marmoursi-reg-noise2.png) |
 
 
## Uncertainty quantification using Dropout

|   |   |
| - | - |
| Inverted $V_p$ | ![](./asset/nnfwi/marmoursi-UQ-noise1.png)    |
| std($V_p$) | ![](./asset/nnfwi/marmoursi-UQ-std1.png) | 
| std($V_p$)/$V_p$ * 100% | ![](./asset/nnfwi/marmoursi-UQ-std1-norm.png) |

 
 
 

