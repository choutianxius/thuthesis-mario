# Perform Network-Dissection on the Super Mario Bros DQN

This module performs 'network dissection' on our trained dqn model which plays the super mario bros game. The 'network dissection' methods heavily rely on [David Bau's work](https://github.com/davidbau/dissect) (`netdissect` and `experiment`).

## Set up

Create the `results` folder in the current directory

~~~shell
mkdir results
~~~

Then download the necessary resources (images and model parameters) from [link](https://drive.google.com/uc?id=1LBHXpqs3glKa1VBQoRv5dKpwJlf_PbJm&export=download) and extract it with `tar -xzf` into the `resources` folder.

## Torch Installation

- Python 3.10.x
- `torch`, tested with `2.1.0.dev20230514+cu121` (go to pytorch website for installation)
- `torchvision`, tested with `0.16.0.dev20230514+cu121` (install with `pip`)
- May need `ninja` for some modules from `netdissect` (install with `pip`)
