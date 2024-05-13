# NAS-SegNet

# Main Code (THIS IS NOT THE MAIN CODE BRANCH) [**Main Project Code Branch**](https://github.com/vishgoki/nas-seg-net/tree/nas-nuclei)


## Description

**NAS-SegNet** Goal of this project is to implement NASSegNet for nuclei image segmentation with analog/hardware-based neural architecture search (HW-NAS). Until now, only classification architectures have been implemented in this domain, so the unique value of this solution is the new segmentation implementation.


# Main Code (THIS IS NOT THE MAIN CODE BRANCH) [**Main Project Code Branch**](https://github.com/vishgoki/nas-seg-net/tree/nas-nuclei)

## Code Updates:
[**NAS Run**](https://github.com/vishgoki/nas-seg-net/blob/nas-nuclei/AnalogNAS_Run.ipynb)   |   
[**Digital Training**](https://github.com/vishgoki/nas-seg-net/blob/nas-nuclei/NAS_NUCLEI_RESNETSEG_DIGITAL.ipynb)   |    
[**Analog Training**](https://github.com/vishgoki/nas-seg-net/blob/nas-nuclei/NAS_NUCLEI_RESNETSEG_ANALOG_Latest.ipynb)

## Approach

* Use the MONAI (Medical Open Network for AI) framework for dataset pre-processing and augmentation.
* Adapt the existing IBM Analog-NAS Macro-Architecture which performs image classification by default to a novel Macro-Architecture which is utilized to run a neural architecture search for generating an optimized NASSegNet model architecture for nuclei segmentation.
* The Analog-NAS approach explores different neural network configurations, evaluating their performance on the target task and hardware constraints, to find the most efficient architecture. This is done by using the pretrained surrogate models.
* We then train the model architecture with the best accuracy using digital and analog methods.


## Results

* Successfully implemented the NASSegNet architecture for nuclei segmentation. 
* Leveraged the IBM Analog-NAS tool to perform a neural architecture search, resulting in an optimized NASSegNet model with best accuracy for this task.
* Trained the optimal network generated model (digital and analog training) with the best model using IBM AIHWKIT.
![image](https://github.com/vishgoki/nas-seg-net/assets/57005975/2655ae1a-c31b-460d-98d1-f08953261867)
<img width="683" alt="image" src="https://github.com/vishgoki/nas-seg-net/assets/57005975/4197b2e1-ea12-4d5a-bb42-913d567acc85">


## Technical Challenges
 
AnalogaiNAS package offers the following features: 

* Utilizing BootstrapNAS by Intel to create a search space and macro architecture for segmentation using UNet Architecture – This was tightly coupled with Intel’s NNCF library and doesn’t support integration with AnalogNAS.
* AnalogNAS and its Image classification dependency: AnalogNAS’s search space, macro architecture are suited for image classification.
* Implementing NASSegNet for nuclei segmentation poses significant technical challenges. 

## Dataset and Data Preparation

Nuclear segmentation in digital microscopic tissue images can enable extraction of high-quality features for nuclear morphometric and other analyses in computational pathology. However, conventional image processing techniques such as Otsu and watershed segmentation do not work effectively on challenging cases such as chromatin-sparse and crowded nuclei. In contrast, machine learning-based segmentation techniques are able to generalize over nuclear appearances. 
Finally, data augmentation and preprocessing transforms are applied using train_transforms and val_transforms and supplied to the model via dataloaders.

Before:
<img width="227" alt="image" src="https://github.com/vishgoki/nas-seg-net/assets/57005975/6eb8be9e-e9db-45bc-ba25-101c0360cc3e">
After:
<img width="227" alt="image" src="https://github.com/vishgoki/nas-seg-net/assets/57005975/78cc45bd-504d-4cd7-a074-82adde19b0bc">

## Training the NASSegNet Model with Analog AI NAS

![image](https://github.com/vishgoki/nas-seg-net/assets/57005975/6a7457e0-f5f0-403c-9834-242130bdecdc)
We define the NASSegNet model architecture, specifying parameters like input/output channels and number of units, downsample, upsample/transpose conv layers.

<img width="409" alt="image" src="https://github.com/vishgoki/nas-seg-net/assets/57005975/aed216d0-91f5-409f-b422-39aff4414d0b">
Set up the Dice Loss as the loss function and the Dice Metric as the evaluation metric for training and validating the model.
![image](https://github.com/vishgoki/nas-seg-net/assets/57005975/2d4e10be-7de0-46d3-be6f-282482ca8fb4)

## Observations and Conclusion

* Successfully implemented the NASSegNet architecture for nuclei segmentation. Generated architecture of model with best accuracy which was then trained digitally and also analog training was performed with aihwtoolkit.
* Leveraged the IBM Analog-NAS tool to perform a neural architecture search. Also worked with AIHWToolkit for analog training.
* This solution represents a implementation of NASSegNet for medical image segmentation, going beyond the previous use of ResNet-like architectures in this domain.







* A customizable resnet-like search space, allowing to target CIFAR-10, Visual Wake Words, and Keyword Spotting 
* A configuration space object allows to add any number or type of architecture and training hyperparameters to the search 
* An analog-specific evaluator which includes: 
  * An 1-day accuracy ranker 
  * An 1 month accuracy variation estimator 
  * A 1-day standard deviation estimator 
* A flexible search algorithm, enabling the implementation and extension of state-of-the-art NAS methods. 

## Structure 
In a high-level AnalogAINAS consists of 4 main building blocks which (can) interact with each other:

* Configuration spaces (```search_spaces/config_space.py```): a search space of architectures targeting a specific dataset.
* Evaluator (```evaluators/base_evaluator.py```): a ML predictor model to predict: 
    * 1-day Accuracy: the evaluator models the drift effect that is encountered in Analog devices. The accuracy after 1 day of drift is then predicted and used as an objective to maximize. 
    * The Accuracy Variation for One Month (AVM): The difference between the accuracy after 1 month and the accuracy after 1 sec. 
    * The 1-day accuracy standard deviation: The stochasticity of the noise induces different variation of the model's accuracy depending on its architecture. 
    
    The weights of these models are provided in (```evaluators/weights```).
* Optimizer (```search_algorithms/```): a optimization strategy such as evolutionary algorithm or bayesian optimization. 
* Worker (```search_algorithms/worker.py```): A global object that runs the architecture search loop and the final network training pipeline

## Setup 
While installing the repository, creating a new conda environment is recomended.

Firstly, refer to [AIHWKit installation](https://aihwkit.readthedocs.io/en/latest/install.html) to install Pytorch and the AIHWKit toolkit. 

Install the additional requirements, using:
```
pip install -r requirements.txt 
```

Afterwards, install AnalogNAS by running the ```setup.py``` file:
``` 
python setup.py install 
```

Alternatively, you can also download the package through pip: 
```
pip install analogainas
```

## Example 

```python
from analogainas.search_spaces.config_space import ConfigSpace
from analogainas.evaluators.xgboost import XGBoostEvaluator
from analogainas.search_algorithms.ea_optimized import EAOptimizer
from analogainas.search_algorithms.worker import Worker

CS = ConfigSpace('CIFAR-10') # define search space, by default a resnet-like search space 
evaluator = XGBoostEvaluator() # load the evaluators 
optimizer = EAOptimizer(evaluator, population_size=20, nb_iter=10)  # define the optimizer with its parameters 

NB_RUN = 2
worker = Worker(CS, optimizer=optimizer, runs=NB_RUN) # The global runner 

worker.search() # start search

worker.result_summary() # print results 

``` 

## Usage
To get started, check out ```nas_search_demo.py``` and ```starter_notebook.ipynb``` to ensure the installation went well. 

## Authors 
AnalogNAS has been developed by IBM Research, 

with Hadjer Benmeziane, Corey Lammie, Irem Boybat, Malte Rasch, Manuel Le Gallo, 
Smail Niar, Hamza Ouarnoughi, Ramachandran Muralidhar, Sidney Tsai, Vijay Narayanan, 
Abu Sebastian, and Kaoutar El Maghraoui

You can contact us by opening a new issue in the repository. 

## How to cite?

In case you are using the _AnalogNas_ toolkit for
your research, please cite the IEEE Edge 2023 paper that describes the toolkit:

> Hadjer Benmeziane, Corey Lammie, Irem Boybat, Malte Rasch, Manuel Le Gallo,
> Hsinyu Tsai, Ramachandran Muralidhar, Smail Niar, Ouarnoughi Hamza, Vijay Narayanan,
> Abu Sebastian and Kaoutar El Maghraoui
> "AnalogNAS: A Neural Network Design Framework for Accurate Inference with Analog In-Memory Computing" (2023 IEEE INTERNATIONAL CONFERENCE ON EDGE
> COMPUTING & COMMUNICATIONS (IEEE Edge))

> https://arxiv.org/abs/2305.10459


## Awards and Media Mentions 

* We are proud to share that AnalogNAS open source project the prestigious **IEEE OPEN SOURCE SCIENCE** in 2023 at the [IEEE 2023 Services Computing Congress](https://conferences.computer.org/services/2023/awards/).
  
 <img width="809" alt="image" src="https://github.com/IBM/analog-nas/assets/7916630/730120f7-7ca1-4ddb-a432-c992470322bc">
 
* AnalogNAS paper received the **Best Paper Award** at [2023 IEEE EDGE (INTERNATIONAL CONFERENCE ON EDGE COMPUTING & COMMUNICATIONS)](https://conferences.computer.org/edge/2023/)

  <img width="796" alt="image" src="https://github.com/IBM/analog-nas/assets/7916630/922a655f-b5fd-4131-80d2-c5b8761c572e">

  


## References
* [IBM/analog-nas]([https://www.ijcai.org/proceedings/2021/592](https://github.com/IBM/analog-nas))
* [AIHWKit](https://ieeexplore.ieee.org/abstract/document/9458494)

## License
This project is licensed under [Apache License 2.0].

[Apache License 2.0]: LICENSE.txt
