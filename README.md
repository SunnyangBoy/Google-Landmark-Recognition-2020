## Google Landmark Recognition 2020 Competition

### Acknowledge
This work was supported by the National Key R&D Program of China under Grant No. 2020AAA0103804(Sponsor: <a  href =" ">Hefu Liu</a >). This work belongs to the University of science and technology of China.


### Environment:

1. pip install -r requirements.txt

2. git clone https://github.com/NVIDIA/apex.git & cd apex

3. python3 setup.py install

4. pip install -v --no-cache-dir ./


### Train Data preparation

1. Download the Google Landmarks Dataset v2 to `./data` using the scripts at https://github.com/cvdfoundation/google-landmark This is our training data.

2. Download the train label csv file from [here](https://bhpan.buaa.edu.cn:443/link/D73F0068AC99184B3FCAEE38A85EBD03) and put it in `./train.csv`

3. Download ReXNet_V1-2.0x pretrained model weights from [here](https://bhpan.buaa.edu.cn:443/link/47F364D7C2604C6EFAFA049B081386AA) and put it in `./rexnetv1_2.0x.pth`

4. `python preprocess.py` to get `./train_0.csv` and `./idx2landmark_id.pkl`

### Training

After training, models will be saved in `./weights/` Tranning logs will be saved in `./logs/` by default.

```
bash run.sh
```

### Test Data preparation

1. Download test data set, put it in `./data/test`, download [test.tar.part1](https://bhpan.buaa.edu.cn:443/link/CBB6F11C81E66C98E72998EC96DE257A), [test.tar.part2](https://bhpan.buaa.edu.cn:443/link/C84A0EE6A907E46CF8714744E8C28F96),  then `cat test.tar.* | tar -xv`

2. `python preprocess_test.py` to generate `./test.csv` label for test, or download from [here](https://bhpan.buaa.edu.cn:443/link/B71380278F2C78A33DFAF6D47B57F316)

3. Download final checkpoint `./weights/final_model.pth` from [here](https://bhpan.buaa.edu.cn:443/link/74E1ED2A3920425588709F9021BA3416)

### Predicting

```
bash run_test.sh
```
