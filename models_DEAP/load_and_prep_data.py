# DEAP
import numpy as np
import matplotlib.pyplot as plt
import mne
import pickle
s01_raw_bdf = mne.io.read_raw_bdf(input_fname='/home/christoph/Desktop/Data_Thesis_Analyze/DEAP/data_original/s01.bdf', exclude=[33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47], preload=True)
data, times = s01_raw_bdf[:]
data_s01_32_channels_plus_status = np.concatenate((data[:32], np.expand_dims(data[-1], axis=0)), axis=0)

ticks = np.arange(0, 300000-200)
ticklabels = ['%d s' % tick for tick in (ticks / 512)]


In [55]: def identify_triggers(trigger_signal, estimated_trigger_distance, indicator_value):   
    ...:    
    ...:     # 1st version: define the timestamp when the signal is at zero again as "start of trigger"   
    ...:     triggers = [0]   
    ...:     ttl_found = False   
    ...:     ttl_samples_ctr = 0   
    ...:        
    ...:     for idx, data_point in enumerate(trigger_signal):   
    ...:         if triggers[-1] + int(0.9 * estimated_trigger_distance) <= idx and trigger_signal[idx] == indicator_value:   
    ...:             ttl_found = True   
    ...:             ttl_samples_ctr = ttl_samples_ctr + 1   
    ...:         else:   
    ...:             ttl_found = False   
    ...:         if ttl_samples_ctr > 0 and not ttl_found:   
    ...:             triggers.append(idx) # -1 as to change of index for old position; -41 as to offset-correciton   
    ...:             ttl_samples_ctr = 0   
    ...:            
    ...:     return triggers[1:]  
    ...:                                                                                                                                                                                     

In [56]: triggers_rating_screen = identify_triggers(data_s01_32_channels_plus_status[32,64000:], 10, 1)                                                                                      

In [57]: len(triggers_rating_screen)                                                                                                                                                         
Out[57]: 160


In [71]: min(['%d' % (triggers_rating_screen[x + 3] - triggers_rating_screen[x]) for x in range(0, len(triggers_rating_screen), 4)])                                                         
Out[71]: '10410'

In [72]: 10410 / 512                                                                                                                                                                         
Out[72]: 20.33203125

In [73]: 512 * 20                                                                                                                                                                            
Out[73]: 10240


I want to take the x+2, as this will be the "beginning of rating screen" plus then 512*20 // 20 seconds