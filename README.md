<h1 align="center">Fingertip detection</h1>
The project is made as part of a deep learning course at ITMO University. The goal was to train a model for fingertip detection and compare with pre-trained models.

<h3>Dataset</h3>

<a href="https://github.com/hukenovs/hagrid">HAGRID</a> dataset was used in a training process. It contains 552992, 
although in order to speed up the training process only a subset of 1800 images were used in both training and validation sets.  

```
git clone https://github.com/hukenovs/hagrid.git
python hagrid/download.py --save_path dataset --subset --annotations --dataset
```

<h3>Training</h3>

The training was done in Google Colab using pre-trained MobileNetV2 model with a classifier containing 3 neurons on the linear layer, since we have to detect only one key point (x, y, conf).
To train the model 1352 and 338 examples were used on both train and validation sets during 10 ecpochs.

<a href="notebooks/model.ipynb">Notebook can be found here</a>.

<h3>Validation and comparison</h3>

When running the trained MobilenetV2 model, Mediapipe hand detector was used to cut a piece of a frame containing hand, and predict on a model.

<table>
    <tr>
        <th>Mediapipe</th>
        <th>MobilenetV2</th>
    </tr>
    <tr>
        <td><img src="imgs/output_mp.gif"></td>
        <td><img src="imgs/output_mb.gif"></td>
    </tr>
</table>


<h3>Inference</h3>

It is recommended to install on a fresh conda environment

```
conda create --name fingertip python==3.9
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

To run the inference use ``inference.py`` with the following commands:
* -m --model *mediapipe* or *mobilenet* (choose one)
* -p --path *models/weights/model_2.pth* (path to the weights)
* -v --video *video.mp4* (path to a video when video mode is on)
* -o --output *output.mp4* (Optional. Path to an output file)

To run the inference using web camera, use the following command:
```
python inference.py camera -m mediapipe -o output.mp4
```

To run the inference using a video, use the following command:
```
python inference.py video -m mediapipe -o output.mp4
```
