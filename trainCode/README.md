
To run the code, could check the example in the `processed_configs/`
write your own config file and place it under `configs/`

then run

```bash
nohup bash batchTest.sh &
```

* If you want to prematurely stop the training, look into the `save_pid.txt` that will locate in the current directory, and extract the pid number `xxx`, then in terminal type

```bash
    kill -9 xxx
```

## some params for configs

* `bm_method`: 
    "pos" - for position
    "deepIA" - for single stage
    "narrowLabel" - for labeling the choosed narrow beam label
    "twostage" - for two stage

* `data_folder`: 

    where the data matrix is stored

* `squareBoundCoords`:

    used in two-stage to segmentize the region
    see my thesis for a clarification

* `saveFolderName` :

    folder name to save the results

* `norm_types`:

    the way to normalize the data

    interesting are

    1: which is use the max and min of the arry

    6: specify the max and min 

    the params are specified in :

```ini

[pos_params]
arr_min = 0
arr_max = 350
[rss_params]
arr_min = -174
arr_max = -37
```

* `loadModel`:

    whether to use trained model or not, if so, model path specified in
    `loadModelPath`

*  `n_beams_list`:

    [rxNum * txNum]          

* `predTxNum` 

    predict tx num
* `predRxNum`:
    predict RX num

* `txNum`:

    the thinking txNum, useful when dealing with deepIA try to extrat RSS

* `rxNum`:

    the real rxNum

    n_beams = predTxNum * predRxNum



## A word on dataset


***IMPORTANT!*** 

* The data are available in the server of the iNets intstitue bowie server, under the path
`/storage/archive/Panwei/DATA/`. If you are interested, please contact the colleague in the institute, [Aleksandar Ichkov](mailto:aic@inets.rwth-aachen.de) for more information.


* to help you get a flavor, a mock 
dataset is provided in the [dataFolder/dataMatrixAll_tx1_mock_20.csv](./trainCode/dataFolder/dataMatrixAll_tx1_mock_20.csv)

(the last `20` in the name just means it is generated via selecting each `20`  row from the one real dataset)

Note to adjust for the smaller size, I changed the batch_size in the config to `128` instead of `1024` stated in the thesis. But you can definitely pick one number you prefer to see the result. 

Have fun experimenting!
