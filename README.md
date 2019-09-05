# CycleGAN and pix2pix in PyTorch

Forked from [Auto Painter] (https://github.com/irfanICMLL/Auto_painter/blob/master/preprocessing/gen_sketch/sketch.py)
[Paper] (https://www.sciencedirect.com/science/article/pii/S0925231218306209?via%3Dihub)

Sketch-to-image synthesis using Conditional Generative Adversarial Networks (cGAN)

# Usage

python3 sketch.py [--input_dir INPUT_FOLDER_PATH --input_image INPUT_IMAGE_PATH] [--gen PATH_FOR_SKETCH] [--gentoorg PATH_FOR_GENTOORG] [--orgtogen PATH_FOR_ORGTOGEN] [--facecrop value] [--remove_dots value]

To save only the generated sketch, run it with --gen option with the directory (image path) you want to store the generated images

```bash
                python sketch.py --input_image INPUT_IMAGE_PATH --gen SKETCH_PATH
                python sketch.py --input_dir INPUT_DIR_PATH --gen SKETCH_PATH
```

To store a concatenated image with the original and generated pictures

```bash
                python sketch.py --input_image INPUT_IMAGE_PATH --orgtogen SKETCH_PATH
                python sketch.py --input_dir INPUT_DIR_PATH --orgtogen SKETCH_PATH
```

To store a concatenated image with the generated and original picture

```bash
                python sketch.py --input_image INPUT_IMAGE_PATH --gentoorg SKETCH_PATH
                python sketch.py --input_dir INPUT_DIR_PATH --gentoorg SKETCH_PATH
```

To use facecrop before sketch Generation

```bash
                python sketch.py --input_image INPUT_IMAGE_PATH --gen SKETCH_PATH --facecrop 100
                python sketch.py --input_dir INPUT_DIR_PATH --gen SKETCH_PATH --facecrop 100
```

To remove dots after sketches

```bash
                python sketch.py --input_image INPUT_IMAGE_PATH --gen SKETCH_PATH --facecrop 100 --remove_dots 50

                python sketch.py --input_dir INPUT_DIR_PATH --gen SKETCH_PATH --facecrop 100 --remove_dots 50
```

# TODO

- Incorporate with face detect, crop and remove dots codebase
