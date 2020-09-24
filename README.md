This repository contains a balanced network of inhibitory and excitatory neurons that undergoes inhibitory perturbations and can elicit the "paradoxical" effects observed in cortical networks. One can summarize the paradoxical effect as counterintuitive observation of changing population firing rate of inihibitory neurons in the opposite direction of the applied perturbations (reducing inputs to inhibitory population increase the firing rate of this population). The simulation scripts were used for simulations in "Sadeh, Sadra, et al. "Assessing the role of inhibition in stabilizing neocortical networks requires large-scale perturbation of the inhibitory population." Journal of Neuroscience 37.49 (2017): 12050-12067." and shared under https://figshare.com/articles/Inhibitory_Stabilized_Network_models/4823212.

**Requirements**

The model is setup and run using NEST simulation tool. Therefore, installing NEST is necessary for running the simulations. Possible methods for installing NEST and their detailed instructions and dependencies can be found under https://nest-simulator.readthedocs.io/en/nest-2.20.1/installation/index.html. 

**Running simulations**

```bash
$ python simulateNetworks.py
$ python analyzeResults.py
$ python figureAllSims.py
$ python figureSample.py
```

Model's parameters are stored in `defaultParams.py `