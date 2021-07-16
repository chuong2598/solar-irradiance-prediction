import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import os
import collections
from tqdm.auto import tqdm
import cv2


# #############################################################################################################
# ############################################ Mask out and resize ############################################
# #############################################################################################################
def mask_out(x, y, center_x, center_y, radius):
    """
    Set pixel value to 0 if fall out a range of a circle with center = (center_x, center_y) and radius.
    """
    if ((x-center_x)**2 + (y - center_y)**2) <= radius**2:
        return 1
    return 0

def generate_mask_and_crop_coordinate(pro_image_path, raw_image_path):
    """
    Function to generate a mask and coordinated used to crop redundant part of an image
    """
    pro = plt.imread(pro_image_path)
    raw = plt.imread(raw_image_path)
    original_raw = raw.copy()
    pro = pro[:,:,2]
    
    non_black_index = np.where(pro != 0)
    axis_0_min_index =  np.min(non_black_index[1])
    axis_0_max_index = original_raw.shape[1]
    axis_1_min_index =  np.min(non_black_index[0])
    axis_1_max_index = np.max(non_black_index[0])
    
    radius = int((axis_0_max_index - ((axis_0_min_index)))/2) + 6
    center_x = axis_0_min_index+int((axis_0_max_index - (axis_0_min_index))/2) - 4
    center_y = axis_1_min_index+int((axis_1_max_index - (axis_1_min_index))/2) 
    
    y_list, x_list = np.mgrid[:original_raw.shape[0], :original_raw.shape[1]]
    coordinate = np.array(list(zip(x_list.flatten(), y_list.flatten()))) 
    mask = np.array([mask_out(coord[0], coord[1], center_x, center_y, radius) for coord in coordinate])
    mask = mask.reshape(original_raw.shape[0], original_raw.shape[1])

    full_mask = np.zeros((original_raw.shape[0], original_raw.shape[1], 3))
    full_mask[:,:,0] = full_mask[:,:,1] = full_mask[:,:,2] = mask
    
    masked_raw = np.uint8(full_mask * original_raw)
    non_black_index = np.where(masked_raw != 0)
    axis_0_min_index =  np.min(non_black_index[1])
    axis_0_max_index = masked_raw.shape[1]
    axis_1_min_index =  np.min(non_black_index[0])
    axis_1_max_index = np.max(non_black_index[0])
    crop_coordinate = (axis_0_min_index, axis_0_max_index, axis_1_min_index, axis_1_max_index)
    
    return full_mask, crop_coordinate
    
def maskout_and_resize(pro_image_path, raw_image_path, images_path, save_path, new_image_size=(128, 128)):
    """
    Function to mask out, crop, and resize image
    """
    full_mask, (axis_0_min_index, axis_0_max_index, axis_1_min_index, axis_1_max_index) = generate_mask_and_crop_coordinate(pro_image_path, raw_image_path)
    for i, path in enumerate(tqdm(sorted(os.listdir(images_path)))):
        if "DS" in path:
            continue
        for j, file_name in enumerate(sorted(os.listdir(os.path.join(images_path, path)))):
            if (j==0):
                os.makedirs("{}/{}".format(save_path, path), exist_ok=True)
            if (file_name.endswith(".jpg")):
                image = plt.imread(os.path.join(images_path, path, file_name))
                image = (image * full_mask).astype(np.uint8)
                image = image[axis_1_min_index: axis_1_max_index-1, axis_0_min_index:, ...]
                plt.imsave("{}/{}/{}".format(save_path, path, file_name), image)
                image = plt.imread("{}/{}/{}".format(save_path, path, file_name))
                image = cv2.resize(image, new_image_size, cv2.INTER_AREA)
                np.save("{}/{}/{}.npy".format(save_path, path, file_name), image)
                # plt.imsave("{}/{}/{}".format(save_path, path, file_name), image)



# #############################################################################################################
# ######################################### Extract sequence of images ########################################
# #############################################################################################################
def get_sequence_data(date_time, forecast_horizon="10m", time_interval=10):
    """
    Function to extract all sequences on a single date
    """
    assert forecast_horizon in ["10m", "1h", "4h"], ("Forecast horizon can be 10m, 1h, or 4h")
    
    index = np.arange(0, len(date_time))
    
    curr_date_time = date_time[:, [3,4]]
    curr_date_time[:, 0] = curr_date_time[:, 0] * 60
    curr_seconds = np.sum(curr_date_time, axis=1)
    
    slice_index = np.arange(0, 100, int(time_interval//10))
    
    if forecast_horizon == "10m":
        consecutive_seconds = np.array(list(zip(
                                            curr_seconds[slice_index[0]:], 
                                            curr_seconds[slice_index[1]:], 
                                            curr_seconds[slice_index[2]:], 
                                            curr_seconds[slice_index[3]:], 
                                            curr_seconds[slice_index[4]:],
                                            curr_seconds[slice_index[5]:],

                                            curr_seconds[slice_index[6]:],
                                                )))

        consecutive_index = np.array(list(zip(
                                            index[slice_index[0]:], 
                                            index[slice_index[1]:], 
                                            index[slice_index[2]:], 
                                            index[slice_index[3]:], 
                                            index[slice_index[4]:],
                                            index[slice_index[5]:],

                                            index[slice_index[6]:],
                                              )))
    
    elif forecast_horizon == "1h":
        consecutive_seconds = np.array(list(zip(
                                            curr_seconds[slice_index[0]:], 
                                            curr_seconds[slice_index[1]:],
                                            curr_seconds[slice_index[2]:], 
                                            curr_seconds[slice_index[3]:], 
                                            curr_seconds[slice_index[4]:],
                                            curr_seconds[slice_index[5]:],

                                            curr_seconds[slice_index[6]:],
                                            curr_seconds[slice_index[7]:],
                                            curr_seconds[slice_index[8]:],
                                            curr_seconds[slice_index[9]:],
                                            curr_seconds[slice_index[10]:],
                                            curr_seconds[slice_index[11]:],
                                                )))
        consecutive_index = np.array(list(zip(
                                            index[slice_index[0]:], 
                                            index[slice_index[1]:], 
                                            index[slice_index[2]:], 
                                            index[slice_index[3]:], 
                                            index[slice_index[4]:],
                                            index[slice_index[5]:], 

                                            index[slice_index[6]:], 
                                            index[slice_index[7]:], 
                                            index[slice_index[8]:],
                                            index[slice_index[9]:], 
                                            index[slice_index[10]:],
                                            index[slice_index[11]:],
                                            )))
        
    else:
        consecutive_seconds = np.array(list(zip(
                                            curr_seconds[slice_index[0]:], 
                                            curr_seconds[slice_index[1]:],
                                            curr_seconds[slice_index[2]:], 
                                            curr_seconds[slice_index[3]:], 
                                            curr_seconds[slice_index[4]:],
                                            curr_seconds[slice_index[5]:],

                                            curr_seconds[slice_index[6]:],
                                            curr_seconds[slice_index[7]:],
                                            curr_seconds[slice_index[8]:],
                                            curr_seconds[slice_index[9]:],
                                            curr_seconds[slice_index[10]:],
                                            curr_seconds[slice_index[11]:],

                                            curr_seconds[slice_index[12]:],
                                            curr_seconds[slice_index[13]:],
                                            curr_seconds[slice_index[14]:],
                                            curr_seconds[slice_index[15]:],
                                            curr_seconds[slice_index[16]:],
                                            curr_seconds[slice_index[17]:],

                                            curr_seconds[slice_index[18]:],
                                            curr_seconds[slice_index[19]:],
                                            curr_seconds[slice_index[20]:],
                                            curr_seconds[slice_index[21]:],
                                            curr_seconds[slice_index[22]:],
                                            curr_seconds[slice_index[23]:],
                                            
                                            curr_seconds[slice_index[24]:],
                                            curr_seconds[slice_index[25]:],
                                            curr_seconds[slice_index[26]:],
                                            curr_seconds[slice_index[27]:],
                                            curr_seconds[slice_index[28]:],
                                            curr_seconds[slice_index[29]:],
                                            )))
        consecutive_index = np.array(list(zip(
                                        index[slice_index[0]:], 
                                        index[slice_index[1]:], 
                                        index[slice_index[2]:], 
                                        index[slice_index[3]:], 
                                        index[slice_index[4]:],
                                        index[slice_index[5]:], 

                                        index[slice_index[6]:], 
                                        index[slice_index[7]:], 
                                        index[slice_index[8]:],
                                        index[slice_index[9]:], 
                                        index[slice_index[10]:],
                                        index[slice_index[11]:],

                                        index[slice_index[12]:],
                                        index[slice_index[13]:],
                                        index[slice_index[14]:],
                                        index[slice_index[15]:],
                                        index[slice_index[16]:],
                                        index[slice_index[17]:],

                                        index[slice_index[18]:],
                                        index[slice_index[19]:],
                                        index[slice_index[20]:],
                                        index[slice_index[21]:],
                                        index[slice_index[22]:],
                                        index[slice_index[23]:],
                                        
                                        index[slice_index[24]:],
                                        index[slice_index[25]:],
                                        index[slice_index[26]:],
                                        index[slice_index[27]:],
                                        index[slice_index[28]:],
                                        index[slice_index[29]:],
                                        )))
    # Return None if there are less than 5 images in one day => Not enough to form 1 sample
    if (len(consecutive_seconds) == 0):
        return None
    
    # Remove data if it is not consecutive data (second difference != 10)
    consecutive_diff = np.diff(consecutive_seconds, axis=1)
    indices = np.all(consecutive_diff == time_interval, axis = 1)
    indices = consecutive_index[indices]
    
    # Flatten indince to perform slicing
    flatten_indices = indices.flatten()
    # Get valid time sequence
    time_sequence = date_time[flatten_indices]
    time_sequence = time_sequence.reshape((indices.shape[0], indices.shape[1], date_time.shape[1]))
    return time_sequence

def id_to_datetime(str_id):
    year = int(str_id[:4])
    month = int(str_id[4:6])
    day = int(str_id[6:8])
    hour = int(str_id[8:10])
    minute = int(str_id[10:12])
    second = int(str_id[12:14])
    return [year, month, day, hour, minute, second]


def date_time_to_id(target_time_sequence):
    id_list = []
    for date_time in target_time_sequence:
        str_date_time = ""
        for element in date_time:
            str_element = str(element)
            if len(str_element) == 1:
                str_element = "0"+str_element
            str_date_time += str_element
        id_list.append(int(str_date_time))
    return id_list



# Extract indexing training data
def extract_data(images_path, save_path, forecast_horizon="10m", time_interval=10):
    """
    Function to extract sequence of image id
    images_path: path to dataset
    save_path: path to save extracted sequences
    time_interval: time interval betwen images in the sequence in minute (default: 10)
    forecast_horizon: the forecast horizon be be either 10m, 1h or 4h (correspond to 10 minutes, 1 hour and 4 hours).
    """
    sequence_id_lists = []
    for path in tqdm(sorted(os.listdir(images_path))):
        datetimes = []
        if(path == ".DS_Store"):
            continue
        skip_file = False
        for file_name in sorted(os.listdir(os.path.join(images_path, path))):
            if (file_name.endswith(".jpg")):
                image = plt.imread(os.path.join(images_path, path, file_name))
                datetimes.append(id_to_datetime(file_name))
                skip_file = False
        if (skip_file == True):
            continue
        if(len(datetimes) == 0):
            continue
        sequence_data = get_sequence_data(date_time=np.array(datetimes), time_interval=time_interval, forecast_horizon=forecast_horizon)
        # Continue if there is no sample extracted in the current day (ex: no sequence in 2011/25/05)
        if(np.all(sequence_data) == None):
            continue
        for i in range(len(sequence_data)):
            sequence_id_lists.append(date_time_to_id(sequence_data[i]))
    sequence_id_lists = np.array(sequence_id_lists)
    np.savetxt("./{}/sequence_id_{}.txt".format(save_path, forecast_horizon), sequence_id_lists, fmt='%s', delimiter=',')


# #############################################################################################################
# ########################################## Process Auxiliary Data ###########################################
# #############################################################################################################

def process_date(date):
    month, day, year = date.split("/")
    return np.array([year, month, day]).astype(object)

def process_time(time):
    minute, hour = time.split(":")
    return np.array([minute, hour]).astype(object)


def process_indice(date_time):
    year, month, date, hour, minute = date_time
    if len(month) < 2:
        month = "0"+month
    if len(month) < 2:
        month = "0"+month
    if len(date) < 2:
        date = "0"+date
    if len(hour) < 2:
        hour = "0"+hour
    if len(minute) < 2:
        minute = "0"+minute
    return year + month + date + hour + minute + "00"

def process_auxiliary_data(data_dir, save_dir):
    f = open(data_dir, "r")

    data = []
    columns = f.readline()

    for line in tqdm(f):
        if "DATE" not in line and len(line) != 0:
            if len(line.split(",")) == 11:
                data.append((line.strip().split(",")))
    data = np.array(data)
    dates = data[:,0]
    processed_date = np.array([process_date(date) for date in tqdm(dates)])

    times = data[:,1]
    processed_time = np.array([process_time(time) for time in tqdm(times)])

    processed_data = data[:,2:].astype(object)
    processed_data = np.c_[processed_date, processed_time, processed_data]

    date_times = processed_data[:, :5]
    processed_indices = np.array([process_indice(date_time) for date_time in date_times])

    processed_data = np.c_[processed_indices, processed_data]
    valid_data_indices = np.where(processed_data[:,5].astype(int) % 10 == 0)
    processed_data = processed_data[valid_data_indices]
    processed_data = processed_data.astype(str)
    np.savetxt(save_dir, processed_data, fmt="%s", delimiter=",")
    f.close()
    return processed_data


# #############################################################################################################
# ##################################### Remove error data Auxiliary Data ######################################
# #############################################################################################################

# !!Error: the auxilary data do not have 20121113143000
def remove_error_solar_power(data_path, dataset, forecast_horizon):
    solar_power_data = np.loadtxt(f"{data_path}/auxiliary_{dataset}.txt", delimiter=",")
    solar_power_data = solar_power_data[:,[0,6]]
    solar_power_dict = {}
    error_solar_image_id = []
    for row in solar_power_data:
        solar_power_dict[str(int(row[0]))] = row[1]
        # Only get data having solar power > 0
        if row[1] <= 0:
            error_solar_image_id.append((int(row[0])))

    sequence_id_list = np.loadtxt(f"{data_path}/sequence_id_{forecast_horizon}.txt", delimiter=",").astype(np.int)

    sequence_id_unique = np.unique(sequence_id_list.flatten()).astype(int)
    auxiliary_image_id = np.array(list(solar_power_dict.keys()))
    removed_indice = np.in1d(sequence_id_unique, auxiliary_image_id)
    removed_image_id = sequence_id_unique[~removed_indice]

    removed_image_id = np.concatenate([removed_image_id, error_solar_image_id])

    valid_sequence_id = []
    for i, sequence_id in enumerate(tqdm(sequence_id_list)):
        if np.any(np.in1d(sequence_id, removed_image_id)):
            # print(removed_image_id)
            # print(sequence_id[0])
            # print()
            continue
        valid_sequence_id.append(i)

    valid_sequence = sequence_id_list[valid_sequence_id].astype(int)

    np.savetxt(f"./{data_path}/sequence_id_auxiliary_{forecast_horizon}.txt", valid_sequence, fmt='%s', delimiter=',')
    
