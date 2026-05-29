# A Physics-Informed Neural Network approach for compartmental epidemiological models

Notebooks for PINN solution of the SIR compartmental model presented [in the paper](https://arxiv.org/abs/2311.09944):

```
@misc{millevoi2023physicsinformed,
      title={A Physics-Informed Neural Network approach for compartmental epidemiological models}, 
      author={Caterina Millevoi and Damiano Pasetto and Massimiliano Ferronato},
      year={2023},
      eprint={2311.09944},
      archivePrefix={arXiv},
      primaryClass={math.NA}
}
```

## Abstract
Compartmental models provide simple and efficient tools to analyze the relevant transmission processes during an outbreak, to produce short-term forecasts or transmission scenarios, and to assess the impact of vaccination campaigns. However, their calibration is not straightforward, since many factors contribute to the rapid change of the transmission dynamics during an epidemic. For example, there might be changes in the individual awareness, the imposition of non-pharmacological interventions and the emergence of new variants. As a consequence, model parameters such as the transmission rate are doomed to change in time, making their assessment more challenging. Here, we propose to use Physics-Informed Neural Networks (PINNs) to track the temporal changes in the model parameters and provide an estimate of the model state variables. 
PINNs recently gained attention in many engineering applications thanks to their ability to consider both the information from data (typically uncertain) and the governing equations of the system. The ability of PINNs to identify unknown model parameters makes them particularly suitable to solve ill-posed inverse problems, such as those arising in the application of epidemiological models. Here, we develop a reduced-split approach for the implementation of PINNs to estimate the temporal changes in the state variables and transmission rate of an epidemic based on the SIR model equation and infectious data. The main idea is to split the training first on the epidemiological data, and then on the residual of the system equations. The proposed method is applied to five synthetic test cases and two real scenarios reproducing the first months of the COVID-19 Italian pandemic. Our results show that the split implementation of PINNs outperforms the standard approach in terms of accuracy (up to one order of magnitude) and computational times (speed up of 20\%).  
