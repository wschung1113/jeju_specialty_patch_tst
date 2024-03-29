# DACON Jeju Island Specialty Price Prediction Hackathon - Applying Official PatchTST Module

## reference

- authors' official github:

    https://github.com/yuqinie98/PatchTST

## introduction

For the previous two endeavors to predict price of Jeju island specialties, I have tried by applying AutoGluon-TS, timeseries forecasting library from Amazon, and with tree-based regression models.

This time, again, I will go with the traditional timeseries forecasting methodology. _PatchTST_ is a novel transformer-based long term timeseries forecasting model architecture recently developed by Yuqi Nie and accepted to ICLR 2023.

Many AutoML libraries such as AutoGluon-TS, NeuralForecast, tsai have adopted this model but I wanted to try adopting the _official module_ and also go through the code for better understanding.

I will try applying this model to predict the price of Jeju island specialties and see what score I can get on the leaderboard for the DACON hackathon.

## task

Task given is to predict various specialty prices for March 4, 2023~March31, 2023 (test set).

Training data given is the daily price of various specialties and covariates such as daily supply amount for each specialty from Jan 1, 2019~Mar 3, 2023.

Ground truth price for March 4, 2023~March31, 2023 is not given and model prediction score can only be shown on the public leaderboard via submission to the hackathon.

## data

1. Daily price and supply amount during Jan 1, 2019~Mar 3, 2023 - train set
   
   ![train_data.png](https://github.com/wschung1113/jeju_specialty/blob/main/images/train_data.png)

2. Monthly international trade data by specialty item during Jan 1, 2019~Mar 3, 2023 - subsidiary set

   ![international_trade.png](https://github.com/wschung1113/jeju_specialty/blob/main/images/international_trade.png)

3. Timestamp and specialty status during March 4, 2023~March31, 2023 - test set

   ![test_data.png](https://github.com/wschung1113/jeju_specialty/blob/main/images/test_data.png)

## application
### steps
1. pivot data (make_data.py)
    
    - Custom dataset other than reference data already set up for the module (i.e., ETTh, ETTm, electricity, weather, etc.) must follow format as below. Data pivoting of the original train set was necessary.

    ![train_data_pivoted_by_timestamp.png](https://github.com/wschung1113/jeju_specialty_patch_tst/blob/main/images/train_data_pivoted_by_timestamp.png)

2. pre-train model in a self-supervised way (patchtst_pretrain.py)
    
    a. Prior to executing patchtst_pretrain.py, code in datautils.py and pred_dataset.py was modified
    
    + datautils.py modifications
        
        ![datautils_mod_1.png](https://github.com/wschung1113/jeju_specialty_patch_tst/blob/main/images/datautils_mod_1.png)

        ![datautils_mod_2.png](https://github.com/wschung1113/jeju_specialty_patch_tst/blob/main/images/datautils_mod_2.png)
    
    + pred_dataset.py modifications
        
        ![pred_dataset_mod_1.png](https://github.com/wschung1113/jeju_specialty_patch_tst/blob/main/images/pred_dataset_mod_1.png)

    b. Execute patchtst_pretrain.py from root with custom arguments

    + --dset refers to your custom dataset alias that was added in _DSETS_ list in datautils.py
    + --context_points refers to the length of look-back window
    + --target_points refers to the prediction horizon length
    + --n_epochs_pretrain is self-explanatory, number of epochs for pre-training
    + --pretrained_model_id refers to the model ID to be given to the pre-trained model
    + --features refers to the feature size of your custom dataset ('M', 'MS', 'S')
        
        * For my speculation, 'M' should stand for multivariate timeseries
        * 'MS' should stand for a multivariate timeseries but for a smaller number of columns a dataset
        * 'S' should stand for univariate timeseries
    + --mask_ratio refers to the masking ratio of timeseries patches for reconstruction learning
    ```bash
    python patchtst_pretrain.py --dset jeju_specialty --context_points=365 --target_points=28 --n_epochs_pretrain=10 --pretrained_model_id=13 --features='M'  --mask_ratio 0.4
    ```

    