# Network Dissection of a DQN Agent Playing Super Mario Bros

![mario gif](mario.gif)

(gif from [gifer.com](https://gifer.com/en/33HU))

This repository contains three modules:

1. `rl`: Train a DQN agent in the gym environment that plays Super Mario Bros;
2. `segmentation`: Train a segmenter model that produces the semantic segmentations of Super Mario Bros game scenes (following [Javier Montalvo et al.'s work](https://link.springer.com/article/10.1007/s11042-022-13695-1));
3. `dissect`: Performs `network dissection` on the trained DQN network (following [David Bau's work](https://dissect.csail.mit.edu/)).

The first part is rather independent and is included as a git submodule. For each module, check their `README.md`s for detailed information. Note that the first module runs with Python 3.9.x and the other two run with Python 3.10.x.
