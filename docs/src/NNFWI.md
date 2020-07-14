# NNFWI 

**Integrating Deep Neural Networks withFull-waveform Inversion: Reparametrization, Regularization, and Uncertainty Quantification**

## Architecture

![](./asset/nnfwi/NNFWI.png)

## Forward Simulation

| Marmousi model                             | Inital 1D model                              |
| ------------------------------------------ | -------------------------------------------- |
| ![](./asset/nnfwi/marmousi_model_true.png) | ![](./asset/nnfwi/marmousi_model_smooth.png) |

| BP2004 model  | Inital 1D model |
| --------------- | -------------- |
| ![](./asset/nnfwi/BP_model_true.png) | ![](./asset/nnfwi/BP_model_smooth.png) |

## Inversion based on Automatic Differentiation

### loss function

| noise level |  Marmousi model | BP2004 Model             |
| ------| ----------------- | ----------------- |
| $\sigma=0$ | ![](./asset/nnfwi/loss_marmousi.png) | ![](./asset/nnfwi/loss_BP.png) |
| $\sigma=0.5$ | ![](./asset/nnfwi/loss_marmousi_noise05.png) | ![](./asset/nnfwi/loss_BP_noise05.png) |


### Marmousi model

| noise level | Traditional  FWI  | NNFWI             |
| ------| ----------------- | ----------------- |
| $\sigma=0$ | ![](./asset/nnfwi/FWI_marmousi.png) | ![](./asset/nnfwi/NNFWI_marmousi.png) |
| $\sigma=0.5$ | ![](./asset/nnfwi/FWI_marmousi_noise05.png) | ![](./asset/nnfwi/NNFWI_marmousi_noise05.png) |
| $\sigma=1$ | ![](./asset/nnfwi/FWI_marmousi_noise10.png) | ![](./asset/nnfwi/NNFWI_marmousi_noise10.png) |

### BP2004 model

| noise level | Traditional  FWI  | NNFWI             |
| ------| ----------------- | ----------------- |
| $\sigma=0$ | ![](./asset/nnfwi/FWI_BP.png) | ![](./asset/nnfwi/NNFWI_BP.png) |
| $\sigma=0.5$ | ![](./asset/nnfwi/FWI_BP_noise05.png) | ![](./asset/nnfwi/NNFWI_BP_noise05.png) |
| $\sigma=1$ | ![](./asset/nnfwi/FWI_BP_noise10.png) | ![](./asset/nnfwi/NNFWI_BP_noise10.png) |



## Uncertainty Quantification using Dropout


| Inverted $V_p$  | std($V_p$)  | std($V_p$)/$V_p$ * 100% |
| ------| ----------------- | ----------------- |
| ![](./asset/nnfwi/NNFWI_marmousi_UQ.png)    |  ![](./asset/nnfwi/NNFWI_marmousi_UQ_std.png) | ![](./asset/nnfwi/NNFWI_marmousi_UQ_std100.png) |
| ![](./asset/nnfwi/NNFWI_BP_UQ.png)    |  ![](./asset/nnfwi/NNFWI_BP_UQ_std.png) | ![](./asset/nnfwi/NNFWI_BP_UQ_std100.png) |


