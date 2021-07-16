import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import os
import collections
from tqdm.auto import tqdm
import pickle
import cv2
import torch
from preprocess_data import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# # ************************************************************************************************************
# # *************************************** Mask out and resize all set ****************************************
# # ************************************************************************************************************
print("Start maskout and resize images")
pro_image_path = "../splitted_dataset/test_2015/sky_images/20150101/20150101075000.pro.png"
raw_image_path = "../splitted_dataset/test_2015/sky_images/20150101/20150101075000.raw.jpg"

training_data_path = "../splitted_dataset/training/sky_images/"
save_path = "../processed_dataset/training/sky_images/"
maskout_and_resize(pro_image_path, raw_image_path, training_data_path, save_path)

val_2011_data_path = "../splitted_dataset/validation_2011/sky_images/"
save_path = "../processed_dataset/validation_2011/sky_images/"
maskout_and_resize(pro_image_path, raw_image_path, val_2011_data_path, save_path)

val_2012_data_path = "../splitted_dataset/validation_2012/sky_images/"
save_path = "../processed_dataset/validation_2012/sky_images/"
maskout_and_resize(pro_image_path, raw_image_path, val_2012_data_path, save_path)

test_2015_data_path = "../splitted_dataset/test_2015/sky_images/"
save_path = "../processed_dataset/test_2015/sky_images/"
maskout_and_resize(pro_image_path, raw_image_path, test_2015_data_path, save_path)

test_2016_data_path = "../splitted_dataset/test_2016/sky_images/"
save_path = "../processed_dataset/test_2016/sky_images/"
maskout_and_resize(pro_image_path, raw_image_path, test_2016_data_path, save_path)
print("Finish masking out and resizing images")


# # # ************************************************************************************************************
# # # ******************************************** Extract sequence  *********************************************
# # # ************************************************************************************************************
# define data path and save path
print("Start extracting sequence id")
training_save_path = "../processed_dataset/training/"
val_2011_save_path = "../processed_dataset/validation_2011/"
val_2012_save_path = "../processed_dataset/validation_2012/"
test_2015_save_path = "../processed_dataset/test_2015/"
test_2016_save_path = "../processed_dataset/test_2016/"

# Extract sequence id from all set (time interval = 10 minute, 1 hour and 4 hours)
forecast_horizon_list = ["10m", "1h", "4h"]
for forecast_horizon in forecast_horizon_list:
    extract_data(training_data_path, training_save_path, forecast_horizon=forecast_horizon)
    extract_data(val_2011_data_path, val_2011_save_path, forecast_horizon=forecast_horizon)
    extract_data(val_2012_data_path, val_2012_save_path, forecast_horizon=forecast_horizon)
    extract_data(test_2015_data_path, test_2015_save_path, forecast_horizon=forecast_horizon)
    extract_data(test_2016_data_path, test_2016_save_path, forecast_horizon=forecast_horizon)
print("Finish extracting sequence id")


# # ************************************************************************************************************
# # ********************************************* Process auxiliary ********************************************
# # ************************************************************************************************************
print("Start processing auxiliary data")
training_dir = "../splitted_dataset/training/solar_irradiance_with_auxiliary_data_training.txt"
save_dir = "../processed_dataset/training/auxiliary_training.txt"
full_training_processed_data = process_auxiliary_data(training_dir, save_dir)

validation_2011_dir = "../splitted_dataset/validation_2011/solar_irradiance_with_auxiliary_data_validation_2011.txt"
save_dir = "../processed_dataset/validation_2011/auxiliary_validation_2011.txt"
full_training_processed_data = process_auxiliary_data(validation_2011_dir, save_dir)

validation_2012_dir = "../splitted_dataset/validation_2012/solar_irradiance_with_auxiliary_data_validation_2012.txt"
save_dir = "../processed_dataset/validation_2012/auxiliary_validation_2012.txt"
full_training_processed_data = process_auxiliary_data(validation_2012_dir, save_dir)

test_2015_dir = "../splitted_dataset/test_2015/solar_irradiance_with_auxiliary_data_test_2015.txt"
save_dir = "../processed_dataset/test_2015/auxiliary_test_2015.txt"
full_training_processed_data = process_auxiliary_data(test_2015_dir, save_dir)

test_2016_dir = "../splitted_dataset/test_2016/solar_irradiance_with_auxiliary_data_test_2016.txt"
save_dir = "../processed_dataset/test_2016/auxiliary_test_2016.txt"
full_training_processed_data = process_auxiliary_data(test_2016_dir, save_dir)
print("Finish processing auxiliary data")

# ************************************************************************************************************
# ******************************************* Remove error auxiliary *****************************************
# ************************************************************************************************************
print("Start removing error data")
forecast_horizon_list = ["10m", "1h", "4h"]
training_data_path = "../processed_dataset/training/"
val_2011_data_path = "../processed_dataset/validation_2011/"
val_2012_data_path = "../processed_dataset/validation_2012/"
test_2015_data_path = "../processed_dataset/test_2015/"
test_2016_data_path = "../processed_dataset/test_2016/"

for forecast_horizon in forecast_horizon_list:
    remove_error_solar_power(training_data_path, "training", forecast_horizon=forecast_horizon)
    remove_error_solar_power(val_2011_data_path, "validation_2011", forecast_horizon=forecast_horizon)
    remove_error_solar_power(val_2012_data_path, "validation_2012", forecast_horizon=forecast_horizon)
    remove_error_solar_power(test_2015_data_path, "test_2015", forecast_horizon=forecast_horizon)
    remove_error_solar_power(test_2016_data_path, "test_2016", forecast_horizon=forecast_horizon)
print("Finish removing error data")
