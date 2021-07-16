import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import os
import collections
from tqdm.auto import tqdm
import pickle
import cv2
import torch




class SpatialTemoralLSTMCell(torch.nn.Module):
    
    def __init__(self, image_shape, in_channel, hidden_channels, kernel_size, stride=1):
        super(SpatialTemoralLSTMCell, self).__init__()
        """
        hidden_channels: Number of hidden features map 
        """
        self.hidden_channels = hidden_channels
        self.padding = kernel_size//2
        self.stride = stride
        
        # !!!!!!!!!!!!!!!! Change here !!!!!!!!!!!!!!!!
        bias_names = ["g_bias", "i_bias", "f_bias", "o_bias", "g_prime_bias", "i_prime_bias", "f_prime_bias"]
        for bias_name in bias_names:
            self.register_parameter(bias_name, torch.nn.Parameter(torch.rand(1)[0]))
        
        self.conv_x = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel, out_channels=self.hidden_channels*7, kernel_size=kernel_size, padding=self.padding, stride=self.stride),
            # Add layer norm to stablize and speed up the training processing
            torch.nn.LayerNorm([hidden_channels*7, image_shape[0], image_shape[1]])
        )
        
        self.conv_h_prev = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels*4, kernel_size=kernel_size, padding=self.padding, stride=self.stride),
            torch.nn.LayerNorm([hidden_channels*4, image_shape[0], image_shape[1]])
        )
        
        self.conv_m_prev = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels*3, kernel_size=kernel_size, padding=self.padding, stride=self.stride),
            torch.nn.LayerNorm([hidden_channels*3, image_shape[0], image_shape[1]])
        )
        
        self.conv_c = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=kernel_size, padding=self.padding, stride=self.stride),
            torch.nn.LayerNorm([hidden_channels, image_shape[0], image_shape[1]])
        )
        
        self.conv_m = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.hidden_channels, out_channels=self.hidden_channels, kernel_size=kernel_size, padding=self.padding, stride=self.stride),
            torch.nn.LayerNorm([hidden_channels, image_shape[0], image_shape[1]])
        )
        
        self.conv_c_m = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.hidden_channels*2, out_channels=self.hidden_channels, kernel_size=1, padding=0, stride=1)
        )
        
    def forward(self, x, h_prev, c_prev, m_prev):
        conv_x = self.conv_x(x)
        g_x, i_x, f_x, o_x, g_x_prime, i_x_prime, f_x_prime = torch.split(tensor=conv_x, split_size_or_sections=self.hidden_channels, dim=1)
        conv_h_prev = self.conv_h_prev(h_prev)
        g_h, i_h, f_h, o_h = torch.split(tensor=conv_h_prev, split_size_or_sections=self.hidden_channels, dim=1)
        g = torch.tanh(g_x + g_h + self.g_bias)
        i = torch.sigmoid(i_x + i_h + self.i_bias)
        f = torch.sigmoid(f_x + f_h + self.f_bias)
        c = f * c_prev + i * g
        
        conv_m_prev = self.conv_m_prev(m_prev)
        g_m_prime, i_m_prime, f_m_prime = torch.split(tensor=conv_m_prev, split_size_or_sections=self.hidden_channels, dim=1)
        g_prime = torch.tanh(g_x_prime + g_m_prime + self.g_prime_bias)
        i_prime = torch.sigmoid(i_x_prime + i_m_prime + self.i_prime_bias)
        f_prime = torch.sigmoid(f_x_prime + f_m_prime + self.f_prime_bias)
        m = f_prime * m_prev + i_prime * g_prime
        
        o_c = self.conv_c(c)
        o_m = self.conv_m(m)
        o = torch.sigmoid(o_x + o_h + o_c + o_m + self.o_bias)
        
        c_m_cat = torch.cat((c,m), dim=1)
        h = o * torch.tanh(self.conv_c_m(c_m_cat))
        return h, c, m

class PredRNN(torch.nn.Module):
    
    def __init__(self, nb_layers, image_shape, in_channel, hidden_layer_dim, kernel_size, stride=1):
        super(PredRNN, self).__init__()
        
        self.nb_layers = nb_layers
        self.hidden_layer_dim = hidden_layer_dim
        self.cell_list = []
        for i in range(nb_layers):
            if i == 0:
                new_cell = SpatialTemoralLSTMCell(image_shape=image_shape, in_channel=in_channel, 
                                                  hidden_channels=hidden_layer_dim, kernel_size=kernel_size, stride=stride)
            else:
                new_cell = SpatialTemoralLSTMCell(image_shape=image_shape, in_channel=hidden_layer_dim, 
                                                  hidden_channels=hidden_layer_dim, kernel_size=kernel_size, stride=stride)
            self.cell_list.append(new_cell)
            
        self.cell_list = torch.nn.ModuleList(self.cell_list)
        self.output_conv = torch.nn.Conv2d(in_channels=hidden_layer_dim, out_channels=in_channel, kernel_size=1, stride=1)
            
    
    def forward(self, batch_of_sequences, total_length, device="cuda"):
        batch, length, nb_channels, height, width = batch_of_sequences.shape
        
        h_list = []
        c_list = []
        for i in range(self.nb_layers):
            h_list.append(torch.zeros(batch, self.hidden_layer_dim, height, width, device=device))
            c_list.append(torch.zeros(batch, self.hidden_layer_dim, height, width, device=device))

        memory = torch.zeros(batch, self.hidden_layer_dim, height, width, device=device)

        prediction = []
        # Recurrent flow (For each timestep, perform vertical flow first)
        for t in range(total_length):
            if (t > length - 1 ):
                input_frame = prediction[-1]
            else:
                input_frame = batch_of_sequences[:, t]

            h_list[0], c_list[0], memory = self.cell_list[0](input_frame, h_list[0], c_list[0], memory)
            for layer in range(1, self.nb_layers):
                h_list[layer], c_list[layer], memory = self.cell_list[layer](h_list[layer-1], h_list[layer], c_list[layer], memory)
            # if (t == 0):
            #     continue
            timestep_prediction = torch.tanh(self.output_conv(h_list[-1]))
            prediction.append(timestep_prediction)

        prediction = torch.stack(prediction).permute(1,0,2,3,4)
        return prediction



class Alex_Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.alex_net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),

            torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(192),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),

            torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(384),

            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(256),
            
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(256),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.alex_net_1 = torch.nn.Sequential(
            torch.nn.Linear(256 * 3 * 3, 1024),
            torch.nn.ReLU(inplace=True),
        )

        self.hid1 =  torch.nn.Linear(in_features=1024, out_features=1024)
        self.drop_out = torch.nn.Dropout(p=0.3)
        self.hid2 =  torch.nn.Linear(in_features=1024, out_features=512)
        self.hid3 =  torch.nn.Linear(in_features=512, out_features=256)
        self.drop_out = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(in_features=256, out_features=1)

    def forward(self, image):
        alex_out = self.alex_net(image)
        alex_out = torch.flatten(alex_out, start_dim=1, end_dim=3)
        alex_out = self.alex_net_1(alex_out)
        hid = torch.relu(self.hid1(alex_out))
        hid = torch.relu(self.hid2(hid))
        hid = torch.relu(self.hid3(hid))
        hid = self.drop_out(hid)
        out = self.out(hid)
        return out



class Alex_Net_Auxiliary(torch.nn.Module):
    def __init__(self, auxiliary_indim):
        super().__init__()

        self.alex_net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),

            torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(192),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),

            torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(384),

            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(256),
            
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(256),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.alex_net_1 = torch.nn.Sequential(
            torch.nn.Linear(256 * 3 * 3, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=1024, out_features=1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=1024, out_features=512)
        )

        self.auxiliary_hid = torch.nn.Sequential(
            torch.nn.Linear(auxiliary_indim, 16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=16, out_features=32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=32, out_features=64)
        )

        self.hid = torch.nn.Sequential(
            torch.nn.Linear(576, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 64),
            torch.nn.ReLU(inplace=True)
            # torch.nn.Dropout(p=0.3)
        )
        self.out = torch.nn.Linear(64, 1)

    def get_features(self, image, auxilary_data):
        alex_out = self.alex_net(image)
        alex_out = torch.flatten(alex_out, start_dim=1, end_dim=3)
        alex_out = self.alex_net_1(alex_out)
        auxiliary_out = self.auxiliary_hid(auxilary_data)
        cat = torch.cat([alex_out, auxiliary_out], dim=1)
        hid = self.hid(cat)
        return hid

    def forward(self, image, auxilary_data):
        alex_out = self.alex_net(image)
        alex_out = torch.flatten(alex_out, start_dim=1, end_dim=3)
        alex_out = self.alex_net_1(alex_out)
        auxiliary_out = self.auxiliary_hid(auxilary_data)
        cat = torch.cat([alex_out, auxiliary_out], dim=1)
        hid = self.hid(cat)
        out = self.out(hid)
        return out



class Auxiliary_LSTM(torch.nn.Module):
    
    def __init__(self, nb_layers, in_dim, hidden_layer_dim, out_dim):
        super(Auxiliary_LSTM, self).__init__()
        self.nb_layers = nb_layers
        self.hidden_layer_dim = hidden_layer_dim
        self.cell_list = []
        for i in range(nb_layers):
            if i == 0:
                new_cell = torch.nn.LSTMCell(input_size=in_dim, hidden_size=hidden_layer_dim)
            else:
                new_cell = torch.nn.LSTMCell(input_size=hidden_layer_dim, hidden_size=hidden_layer_dim)
            self.cell_list.append(new_cell)
            
        self.cell_list = torch.nn.ModuleList(self.cell_list)
        self.out1 = torch.nn.Linear(in_features=hidden_layer_dim, out_features=hidden_layer_dim*2)
        self.out2 = torch.nn.Linear(in_features=hidden_layer_dim*2, out_features=hidden_layer_dim*4)
        self.out3 = torch.nn.Linear(in_features=hidden_layer_dim*4, out_features=out_dim)

    def increase_minute(self, current_datetime, amount_minute):
        total_minute = current_datetime[:,2]*60 + current_datetime[:,3] + amount_minute
        hours = total_minute // 60
        minutes = total_minute % 60
        new_date_time = current_datetime.clone()
        new_date_time[:, 2] = hours
        new_date_time[:, 3] = minutes
        return new_date_time

    def forward(self, x, total_length, device="cuda"):
        batch, in_length, in_dim = x.shape
        h_list = []
        c_list = []
        for i in range(self.nb_layers):
            h_list.append(torch.zeros(batch, self.hidden_layer_dim, device=device))
            c_list.append(torch.zeros(batch, self.hidden_layer_dim, device=device))

        prediction = []
        output_date_time_list = []
        # Recurrent flow (For each timestep, perform vertical flow first)
        for t in range(total_length):
            if (t > in_length - 1 ):
                prev_pred = prediction[-1]
                amount_minute = ((t+1)-in_length)*10
                input_date_time = self.increase_minute(x[:, -1, :4], amount_minute)
                output_date_time = self.increase_minute(x[:, -1, :4], amount_minute+10)
                input_frame = torch.cat([input_date_time, prev_pred], dim=1)
            else:
                output_date_time = self.increase_minute(x[:, t, :4], 10)
                input_frame = x[:, t]
            h_list[0], c_list[0] = self.cell_list[0](input_frame, (h_list[0], c_list[0]))
            for layer in range(1, self.nb_layers):
                h_list[layer], c_list[layer] = self.cell_list[layer](h_list[layer-1], (h_list[layer], c_list[layer]))
            # if (t == 0):
            #     continue
                
            out1 = torch.relu(self.out1(h_list[-1]))
            out2 = torch.relu(self.out2(out1))
            timestep_prediction = self.out3(out2)
            prediction.append(timestep_prediction)
            output_date_time_list.append(output_date_time.detach().cpu().numpy())
        
        auxiliary_prediction = torch.stack(prediction).permute(1,0,2)
        output_date_time_list = torch.tensor(np.array(output_date_time_list)).squeeze().permute(1,0,2).to(device=device)
        prediction = torch.cat([output_date_time_list, auxiliary_prediction], dim=2)
        return prediction




