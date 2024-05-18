import torch
import torch.nn as nn
import numpy as np
import pywt
import math
from layers.Invertible import RevIN
from layers.DishTS import DishTS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


#先移动平均再去噪  shift
class dropping_noise(nn.Module):
    """
    """
    def __init__(self,dec_lev):
        super(dropping_noise, self).__init__()
        self.dec_lev = dec_lev

    def forward(self, x):
        # padding on the both ends of time series
        def sgn(num):
            if (num > 0.0):
                return 1.0
            elif (num == 0.0):
                return 0.0
            else:
                return -1.0

        B, T, N = x.shape
        data = x.reshape(-1)
        data = data.T.tolist()  # 将np.ndarray()转为列表
        length0 = len(data)
        w = pywt.Wavelet('sym8')  # 选择sym8小波基
        #decLev = 15
        cdList= pywt.wavedec(data, w, level=self.dec_lev)  # 小波分解
        ca=cdList[0]
        del (cdList[0])
        cdList = list(reversed(cdList))


        Cd1 = np.array(cdList[0])
        abs_cd1 = np.abs(Cd1)
        median_cd1 = np.median(abs_cd1)

        sigma = (1.0 / 0.6745) * median_cd1
        lamda = sigma * math.sqrt(2.0 * math.log(float(length0), math.e))  # 固定阈值计算
        usecoeffs = []
        usecoeffs.append(ca)  # 向列表末尾添加对象
        # 软硬阈值折中的方法
        a = 0.85

        for i in range(self.dec_lev):
            for k in range(len(cdList[i])):
                if (abs(cdList[i][k]) >= lamda):
                    cdList[i][k] = sgn(cdList[i][k]) * (abs(cdList[i][k]) - a * lamda)
                else:
                    cdList[i][k] = 0.0

        for f in range(self.dec_lev):
            usecoeffs.append(cdList[self.dec_lev - f - 1])
        recoeffs = pywt.waverec(usecoeffs, w)  # 信号重构
        recoeffs = recoeffs.reshape(B,T,N)
        return recoeffs

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        #a=x.mean(1)
        #avg=torch.repeat_interleave(a,(self.kernel_size - 1) // 2,dim=0).view(x.shape[0],(self.kernel_size - 1) // 2,x.shape[2])
        #x = torch.cat([avg, x, avg], dim=1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size,dec_lev):
        super(series_decomp, self).__init__()
        self.drop = dropping_noise(dec_lev)
        self.moving_avg = moving_avg(kernel_size, stride=1)


    def forward(self, x):
        #无噪声趋势项分解
        trend = self.moving_avg(x)
        res = x-trend
        #moving_mean
        if (len(trend[0]) % 2 != 0):
            trend_no_noise = self.drop(trend[...,:, :-1])
        else:
            trend_no_noise = self.drop(trend)
        trend_no_noise=torch.tensor(trend_no_noise).cuda()
        trend_noise = trend - trend_no_noise
        #res
        if (len(res[0]) % 2 != 0):
            res_no_noise = self.drop(res[..., :, :-1])
        else:
            res_no_noise = self.drop(res)
        res_no_noise = torch.tensor(res_no_noise).cuda()
        res_noise = res - res_no_noise

        # numpy_x = x.cpu().detach().numpy()
        # numpy_trend = trend.cpu().detach().numpy()
        # numpy_trend_no_noise = trend_no_noise.cpu().detach().numpy()
        # a=numpy_x[:, :, -1][...,-1]
        # b=numpy_trend[:, :, -1][...,-1]
        # c = numpy_trend_no_noise[:, :, -1][..., -1]
        # np.save('input/slicex.npy', a)
        # np.save('input/sliceTrend.npy', b)
        # np.save('input/sliceTrend_no_noise.npy', c)

        return trend_noise.float(), trend_no_noise.float(),res_noise.float(),res_no_noise.float()


class Model(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.channels = configs.enc_in
        self.individual = configs.individual
        self.dec_lev = configs.dec_lev
        self.rev = RevIN(configs.enc_in) if configs.rev else None
        #self.dishts = DishTS(configs.enc_in,configs.seq_len) if configs.dishts else None
        self.dropout = nn.Dropout(configs.dropout)
        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size,self.dec_lev)
        if self.individual:
            self.Linear_Trend_Noise = nn.ModuleList()
            self.Linear_Trend_No_Noise = nn.ModuleList()
            self.Linear_Seasonal_Noise = nn.ModuleList()
            self.Linear_Seasonal_No_Noise = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Trend_Noise.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend_No_Noise.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Seasonal_Noise.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Seasonal_No_Noise.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Trend_Noise = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend_No_Noise = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal_Noise = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal_No_Noise = nn.Linear(self.seq_len, self.pred_len)
        #self._build()


    def forward(self, x):
        # x: [Batch, Input length, Channel]
        x = self.rev(x, 'norm') if self.rev else x
        #x = self.dishts(x,'forward') if self.dishts else x
        x = self.dropout(x)
        trend_noise_init, trend_no_noise_init, seasonal_noise_init,seasonal_no_noise_init = self.decompsition(x)
        trend_noise_init, trend_no_noise_init, seasonal_noise_init ,seasonal_no_noise_init= \
            trend_noise_init.permute(0, 2, 1), trend_no_noise_init.permute(0, 2, 1), seasonal_noise_init.permute(0, 2, 1),seasonal_no_noise_init.permute(0, 2, 1)


        if self.individual:
            trend_noise_output = torch.zeros([trend_noise_init.size(0), trend_noise_init.size(1), self.pred_len],
                                          dtype=trend_noise_init.dtype).to(trend_noise_init.device)
            trend_no_noise_output = torch.zeros([trend_no_noise_init.size(0), trend_no_noise_init.size(1), self.pred_len],
                                       dtype=trend_no_noise_init.dtype).to(trend_no_noise_init.device)
            seasonal_noise_output = torch.zeros([seasonal_noise_init.size(0), seasonal_noise_init.size(1), self.pred_len],
                                                dtype=seasonal_noise_init.dtype).to(seasonal_noise_init.device)
            seasonal_no_noise_output = torch.zeros([seasonal_no_noise_init.size(0), seasonal_no_noise_init.size(1), self.pred_len],
                                                dtype=seasonal_no_noise_init.dtype).to(seasonal_no_noise_init.device)
            for i in range(self.channels):
                trend_noise_output[:, i, :] = self.Linear_Trend_Noise[i](trend_noise_init[:, i, :])
                trend_no_noise_output[:, i, :] = self.Linear_Trend_No_Noise[i](trend_no_noise_init[:, i, :])
                seasonal_noise_output[:, i, :] = self.Linear_Seasonal_Noise[i](seasonal_noise_init[:, i, :])
                seasonal_no_noise_output[:, i, :] = self.Linear_Seasonal_No_Noise[i](seasonal_no_noise_init[:, i, :])
        else:
            trend_noise_output = self.Linear_Trend_Noise(trend_noise_init.float())
            trend_no_noise_output = self.Linear_Trend_No_Noise(trend_no_noise_init.float())
            seasonal_noise_output = self.Linear_Seasonal_Noise(seasonal_noise_init.float())
            seasonal_no_noise_output = self.Linear_Seasonal_No_Noise(seasonal_no_noise_init.float())

        x = (trend_noise_output + trend_no_noise_output + seasonal_noise_output + seasonal_no_noise_output).permute(0, 2, 1)
        x = self.rev(x, 'denorm') if self.rev else x
        #x = self.dishts(x, 'inverse') if self.dishts else x
        return x  # to [Batch, Output length, Channel]
