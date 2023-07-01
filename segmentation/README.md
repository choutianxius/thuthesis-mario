# Train a Semantic Segmenter for Super Mario Bros Game Scenes

To perform 'network dissection', the semantic segmentations of input images are needed. As there is no ground truth for game scenes, a segmenter (that takes game scenes as input and returns the semantic segmentation bitmap as output) is required.

We rely on [Javier Montalvo et al.](https://link.springer.com/article/10.1007/s11042-022-13695-1) to train the segmentation model. Simulated game scenes are produced by a generator alongside with the corresponding semantic bitmaps. The simulated images and semantic bitmaps are used as the dataset to train the segmenter. Training is achieved by fine-tuning the [DeepLab V3 ResNet-50](https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.deeplabv3_resnet50.html) image segmenter model.

- `dataset_generator` contains the code to make the dataset of simulated game scenes and their segmantation bitmaps. Check `tutorial.md` inside it to build your dataset. (`generate_frames.py` has been modified from the original author's code so that only images resembling world 1 stage 1 are generated.)
- `training` contains a jupyter notebook where the segmentation model is trained. The notebook is modified from the original author's work.
