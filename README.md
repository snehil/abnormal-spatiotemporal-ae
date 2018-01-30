# Anomaly detection in Smart Audio Videos using a Convolutional Spatio-temporal Auto-encoder and Convolutional Long-Short Term Memory (LSTM) Recurrent Neural Network based architecture.

## Project dependencies:
- keras
- tensorflow
- h5py
- scikit-image
- scikit-learn
- sk-video
- tqdm (for progressbar)
- coloredlogs (optional, for colored terminal logs only)

To build the docker image run the following docker command. This will take around 13GB space on your machine so makes sure you clean up any unused docker images and containers to avoid out of free space issues.
```docker build -t snehil/video_anomaly_ai:v1 .```

You can then enter the environment using ```nvidia-docker run --rm -it -v HOST_FOLDER:/share DOCKER_IMAGE bash```.

or run the following command:
```docker run --rm -it -v ~/:/share snehil/<IMAGE_ID> bash```

To login to a running container:
```docker attach <CONTAINER_ID>```

## To train the model 

 - Just run `python start_train.py`. 
 - Default configuration can be found at `config.yml`. 
 - You need to prepare video dataset you plan to train/evaluate on (avi or mp4). For each dataset, put the training videos into ```   
   ./data/videos/training_videos``` and testing videos into ```./data/videos/testing_videos```. Example structure of training videos for `avenue` dataset:
 - `VIDEO_ROOT_PATH/avenue/training_videos`
  - `01.avi`
  - `02.avi`
  - ...
  - `16.avi`

## Avenue dataset
The Avenue dataset can be downloaded from here: http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html

## To test the model
Once you have trained the model, you may now run `python start_test.py` after setting the parameters at the beginning of the file.

# Troubleshooting

In case you get a No free space left error (inspite of having free space) while running the docker build command, try cleaning up unused docker images and containers using the following commands
```
docker rm $(docker ps -q -f 'status=exited')
docker rmi $(docker images -q -f "dangling=true")

```


Please cite the following paper if you use our code / paper:
```
@inbook{Chong2017,
  author    = {Chong, Yong Shean and
               Tay, Yong Haur},
  editor    = {Cong, Fengyu and
               Leung, Andrew and
               Wei, Qinglai},
  title     = {Abnormal Event Detection in Videos Using Spatiotemporal Autoencoder},
  bookTitle = {Advances in Neural Networks - ISNN 2017: 14th International Symposium, ISNN 2017, Sapporo, Hakodate, and Muroran, Hokkaido, Japan, June 21--26, 2017, Proceedings, Part II},
  year      = {2017},
  publisher = {Springer International Publishing},
  pages     = {189--196},
  isbn      = {978-3-319-59081-3},
  doi       = {10.1007/978-3-319-59081-3_23},
  url       = {https://doi.org/10.1007/978-3-319-59081-3_23}
}
```
