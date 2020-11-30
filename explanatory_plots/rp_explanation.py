fig, axes = plt.subplots(2, 3, sharex=False, sharey=False)
axes[0, 0].plot(time, amplitude + signal_distortion, color='blue')
axes[1, 0].plot(time, amplitude + signal_distortion + trend, color='red')
axes[0, 0].set_xlim([0,30])
axes[1, 0].set_xlim([0,30])
axes[0, 0].set_ylim([-2,5])
axes[1, 0].set_ylim([-2,5])
axes[0, 0].set_ylabel('sin(time + distortion)')
axes[1, 0].set_ylabel('sin(time + distortion + trend)')
axes[0, 0].set_xlabel('time in s')
axes[1, 0].set_xlabel('time in s')
img0 = axes[0, 1].imshow(x_transformed_binary[0] * 5, cmap=plt.cm.binary, vmin=0, vmax=5)
img1 = axes[0, 2].imshow(x_transformed[0], cmap=plt.cm.plasma, vmin=0, vmax=5)
img2 = axes[1, 1].imshow(x_transformed_binary_with_trend[0] * 5, cmap=plt.cm.binary, vmin=0, vmax=5)
img3 = axes[1, 2].imshow(x_transformed_with_trend[0], cmap=plt.cm.plasma, vmin=0, vmax=5)
bar0 = plt.colorbar(img0, ax=axes[0, 1], ticks=[0, 5])
bar0.ax.set_yticklabels(['far', 'close'])  
bar1 = plt.colorbar(img1, ax=axes[0, 2])
bar1.set_label('distance')  
bar2 = plt.colorbar(img2, ax=axes[1, 1], ticks=[0, 5])
bar2.ax.set_yticklabels(['far', 'close']) 
bar3 = plt.colorbar(img3, ax=axes[1, 2])
bar3.set_label('distance')

axes[0, 1].set_yticks(ticks=[0, 500, 1000, 1500, 2000, 2500, 3000])
axes[0, 1].set_yticklabels(labels=['0', '5', '10', '15', '20', '25', '30'])
axes[0, 1].set_xticks(ticks=[0, 500, 1000, 1500, 2000, 2500, 3000])
axes[0, 1].set_xticklabels(labels=['0', '5', '10', '15', '20', '25', '30'])
axes[0, 1].set_xlim([0, 3000]) 
axes[0, 1].set_ylim([0, 3000])
axes[0, 1].set_ylabel('time in s')
axes[0, 1].set_xlabel('time in s')
axes[0, 2].set_yticks(ticks=[0, 500, 1000, 1500, 2000, 2500, 3000])
axes[0, 2].set_yticklabels(labels=['0', '5', '10', '15', '20', '25', '30'])
axes[0, 2].set_xticks(ticks=[0, 500, 1000, 1500, 2000, 2500, 3000])
axes[0, 2].set_xticklabels(labels=['0', '5', '10', '15', '20', '25', '30'])
axes[0, 2].set_xlim([0, 3000]) 
axes[0, 2].set_ylim([0, 3000])
axes[0, 2].set_ylabel('time in s')
axes[0, 2].set_xlabel('time in s')
axes[1, 1].set_yticks(ticks=[0, 500, 1000, 1500, 2000, 2500, 3000])
axes[1, 1].set_yticklabels(labels=['0', '5', '10', '15', '20', '25', '30'])
axes[1, 1].set_xticks(ticks=[0, 500, 1000, 1500, 2000, 2500, 3000])
axes[1, 1].set_xticklabels(labels=['0', '5', '10', '15', '20', '25', '30'])
axes[1, 1].set_xlim([0, 3000]) 
axes[1, 1].set_ylim([0, 3000])
axes[1, 1].set_ylabel('time in s')
axes[1, 1].set_xlabel('time in s')
axes[1, 2].set_yticks(ticks=[0, 500, 1000, 1500, 2000, 2500, 3000])
axes[1, 2].set_yticklabels(labels=['0', '5', '10', '15', '20', '25', '30'])
axes[1, 2].set_xticks(ticks=[0, 500, 1000, 1500, 2000, 2500, 3000])
axes[1, 2].set_xticklabels(labels=['0', '5', '10', '15', '20', '25', '30'])
axes[1, 2].set_xlim([0, 3000]) 
axes[1, 2].set_ylim([0, 3000])
axes[1, 2].set_ylabel('time in s')
axes[1, 2].set_xlabel('time in s')
fig.set_size_inches(12, 6)
plt.tight_layout()
plt.savefig('/home/christoph/Desktop/Thesis_Plots/RP_Explanation_Plots/urgently_needed_example/rpexplanation.png', dpi=100)

#
trend = np.arange(0, 3, 0.001)
In [2]: import numpy as np                                                                                                                                                                                        

In [3]: import matplotlib.pyplot as plt                                                                                                                                                                           

In [4]: from pyts.image import RecurrencePlot                                                                                                                                                                     

In [5]: rp = RecurrencePlot()                                                                                                                                                                                     

In [6]: rp_binary = RecurrencePlot(threshold='distance', percentage=20)                                                                                                                                           

In [7]: time = np.arange(0, 30, 0.01)                                                                                                                                                                             

In [8]: amplitude = np.sin(time)                                                                                                                                                                                  

In [9]: signal_distortion = np.random.randn(3000) * 0.3 

In [36]: x_transformed_binary_with_trend = rp_binary.fit_transform((amplitude + signal_distortion + trend).reshape(1, -1))                                                                                        

In [37]: x_transformed_binary = rp_binary.fit_transform((amplitude + signal_distortion).reshape(1, -1))   
In [38]: x_transformed_with_trend = rp.fit_transform((amplitude + signal_distortion + trend).reshape(1, -1))                                                                                                      

In [39]: x_transformed = rp.fit_transform((amplitude + signal_distortion).reshape(1, -1))   