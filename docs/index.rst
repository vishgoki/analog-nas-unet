AnalogNAS
=========

.. toctree::
    :maxdepth: 3
    :caption: Get started
    :hidden:

    install
    getting_started


.. toctree::
   :maxdepth: 3
   :caption: References
   :hidden:

   api_references
   references

AnalogNAS is a Python library designed to empower researchers and practitioners in efficiently exploring and optimizing neural network architectures specifically for in-memory computing scenarios. AnalogNAS is built on top of the IBM Analog Hardware Acceleration Kit that enables efficient hardware-aware training with simulated noise injection on multiple IMC devices. By capitalizing on the advantages of in-memory computing, AnalogNAS opens new avenues for discovering architectures that can fully exploit the capabilities of this emerging computing paradigm.

AnalogNAS offers a comprehensive set of features and functionalities that facilitate the neural architecture search process. From seamlessly exploring a vast space of architectural configurations to fine-tuning models for optimal performance, AnalogNAS provides a versatile framework that accelerates the discovery of efficient and effective neural network architectures for in-memory computing.

.. warning::
    This library is currently in beta and under active development.
    Please be mindful of potential issues and keep an eye for improvements,
    new features and bug fixes in upcoming versions.

Features
--------


- A customizable resnet-like search space, allowing to target CIFAR-10, Visual Wake Words, and Keyword Spotting 
- A configuration space object allows to add any number or type of architecture and training hyperparameters to the search 
- An analog-specific evaluator which includes: 

  - An 1-day accuracy ranker 
  - An 1 month accuracy variation estimator 
  - A 1-day standard deviation estimator 

- A flexible search algorithm, enabling the implementation and extension of state-of-the-art NAS methods. 

Installation
------------

Install analogNAS by running:

    pip install analogainas


How to cite
-----------

In case you are using the *AnalogNAS* for
your research, please cite:

.. note::

    Benmeziane, H., Lammie, C., Boybat, I., Rasch, M., Gallo, M. L., Tsai, H., ... & Maghraoui, K. E. (2023). AnalogNAS: A Neural Network Design Framework for Accurate Inference with Analog In-Memory Computing. IEEE Edge 2023. 
    
    https://arxiv.org/abs/2305.10459
