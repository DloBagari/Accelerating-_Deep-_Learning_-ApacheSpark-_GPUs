'''
Monitor GPU use
'''

from __future__ import print_function
from __future__ import division

import numpy as np

import matplotlib
import os
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import time
import datetime
from scipy.interpolate import spline
import sys



def get_current_milliseconds():
    
    return(int(round(time.time() * 1000)))


def get_current_time_in_seconds():
    
    return(time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime()))

def get_current_time_in_miliseconds():
    
    return(get_current_time_in_seconds() + '-' + str(datetime.datetime.now().microsecond))




def generate_plot(gpu_log_filepath, max_history_size, graph_filepath):
   
    history_size = 0
    number_of_gpus = -1
    gpu_utilization = []
    gpu_memory = []
    gpu_utilization_one_timestep = []
    gpu_memory_one_timestep = []
    for line_number, line in enumerate(reversed(open(gpu_log_filepath).readlines())): 
        if history_size > max_history_size:
            break

        line = line.split(',')

        if line[0].startswith('util') or len(gpu_utilization_one_timestep) == number_of_gpus:
            if number_of_gpus == -1 and len(gpu_utilization_one_timestep) > 0:

                 number_of_gpus = len(gpu_utilization_one_timestep)
            if len(gpu_utilization_one_timestep) == number_of_gpus:

                gpu_utilization.append(list(reversed(gpu_utilization_one_timestep)))
                gpu_memory.append(list(reversed(gpu_memory_one_timestep))) 
                
                history_size += 1

            else: 
                pass

              

            gpu_utilization_one_timestep = []

            gpu_memory_one_timestep = []
        if line[0].startswith('util'): continue

        try:
            current_gpu_utilization = int(line[0].strip().replace(' %', ''))
            current_gpu_memory = int(line[1].strip().replace(' MiB', ''))
            current_gpu_memory =int( (current_gpu_memory / 4037) *100)


        except:
            print('line: {0}'.format(line))
            print('line_number: {0}'.format(line_number))
            1/0
        gpu_utilization_one_timestep.append(current_gpu_utilization)
        gpu_memory_one_timestep.append(current_gpu_memory)


    
    gpu_utilization = np.array(list(reversed(gpu_utilization))) 
    gpu_memory = np.array(list(reversed(gpu_memory)))

    
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    values = []
    values2 = []
    for i in gpu_utilization:
        for j in i:
            values.append(j)
    for i in gpu_memory:
        for j in i:
            values2.append(j)

    x =np.array(range(len(values)))
    y =np.array(values)
    xx =np.linspace(x.min(), x.max(), 300)


    yy = spline(x, y, xx)

    xm =np.array(range(len(values2)))
    ym =np.array(values2)
    xxm =np.linspace(xm.min(), xm.max(), 300)


    yym = spline(xm, ym, xxm)
    

    
    ax.plot(xxm,yym, linewidth=1.3, antialiased=True, color= "blue")
    ax.fill_between(xxm, yym, alpha=0.6, zorder = 5, antialiased=True, color= "blue")

    ax.plot(xx,yy, linewidth=1.3, antialiased=True, color="#A81450" )
    ax.fill_between(xx, yy, alpha=0.8, zorder = 5, antialiased=True, color="#A81450")
    
    ax.set_title('GPU_1 utilization,used memory over time ({0})'.format(get_current_time_in_miliseconds()))
    ax.set_xlabel('5 seconds window')
    ax.set_ylabel('GPU_1 utilization and memory (%)')
    gpu_utilization_mean_per_gpu = np.mean(gpu_utilization, axis=0)
    gpu_memory_mean_per_gpu = np.mean(gpu_memory, axis=0)
    avgs = []
    
    avgs.extend(['Memory (avg:{1})'.format(2, np.round(gpu_memory_mean, 1)) for gpu_number, gpu_memory_mean in zip(range(gpu_memory.shape[1]), gpu_memory_mean_per_gpu)])
    avgs.extend(['Utilization (avg:{1})'.format(2, np.round(gpu_utilization_mean, 1)) for gpu_number, gpu_utilization_mean in zip(range(gpu_utilization.shape[1]), gpu_utilization_mean_per_gpu)])
    lgd = ax.legend(avgs, loc='center right', bbox_to_anchor=(1.45, 0.5))

    plt.tight_layout()
    plt.grid(alpha=0.8)
    plt.savefig(graph_filepath, dpi=300, format='png', bbox_inches='tight')

    plt.close()



def main():
    '''
    This is the main function
    '''
    # Parameters
    gpu_log_filepath = 'gpu_utilization1.log'  
    max_history_size = 100
    max_history_sizes =[500] 
    for max_history_size in max_history_sizes:
        graph_filepath = 'gpu1_utillization_{0}.png'.format(max_history_size)
        generate_plot(gpu_log_filepath, max_history_size, graph_filepath)
if __name__ == "__main__":
    main()
