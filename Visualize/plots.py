########################################################################################################################
"""
Description :  Contains methods for all the plots used in the paper TX-Ray.
Python version : 3.7.3
author Vageesh Saxena
"""
########################################################################################################################

########################################### Importing Libraries ########################################################
import os
import sys
import heapq
import operator
import pickle
from collections import Counter, OrderedDict, defaultdict

import numpy as np
import pandas as pd

import plotly
import plotly.express as px
import plotly.graph_objects as go

# Custom Libraries
sys.path.append('../../Evaluation_metrics/')
from measures import find_shared_neurons
########################################################################################################################

def mass_activation_plot(unsup_data, zero_shot_data, sup_data, data_dict):
    """
    :param unsup_data: Unsupervised data(dtype:pandas dataframe)
    :param zero_shot_data: Zero shot data(dtype:pandas dataframe)
    :param sup_data: Supervised data(dtype:pandas dataframe)
    :param data_dict: dictionary containing input instructions(dtype:dict)
    Plots the mass activation plot and save it in data_dict["visualize"]["plot_directory"]
    """
    
    if not os.path.exists(data_dict["visualize"]["plot_directory"]):
        os.makedirs(data_dict["visualize"]["plot_directory"])
        
    zero_shot_neurons = list(zero_shot_data['max_activation_index'].unique())
    unsup_neurons = list(unsup_data['max_activation_index'].unique())
    sup_neurons = list(sup_data['max_activation_index'].unique())
    
    zero_shot_mass_dict, unsup_mass_dict, sup_mass_dict = ({} for i in range(3))
    
    for neuron in unsup_neurons:
        temp = unsup_data[unsup_data['max_activation_index']==neuron]
        unsup_mass_dict[neuron] = sum(temp['max_activations'])
    for neuron in zero_shot_neurons:
        temp = zero_shot_data[zero_shot_data['max_activation_index']==neuron]
        zero_shot_mass_dict[neuron] = sum(temp['max_activations'])
    for neuron in sup_neurons:
        temp = sup_data[sup_data['max_activation_index']==neuron]
        sup_mass_dict[neuron] = sum(temp['max_activations'])
        
    sup = [value[1] for value in sorted(sup_mass_dict.items(), key=operator.itemgetter(1), reverse=True)]
    unsup = [value[1] for value in sorted(unsup_mass_dict.items(), key=operator.itemgetter(1), reverse=True)]
    zshot = [value[1] for value in sorted(zero_shot_mass_dict.items(), key=operator.itemgetter(1), reverse=True)] 
    
    fig = go.Figure()
    fig.add_trace(go.Bar(y=sup, name="sup", marker_color=data_dict['visualize']['viz_colors']['sup_color']))
    fig.add_trace(go.Bar(y=unsup, name="unsup", marker_color=data_dict['visualize']['viz_colors']['unsup_color']))
    fig.add_trace(go.Bar(y=zshot, name="zshot", marker_color=data_dict['visualize']['viz_colors']['zero_shot_color']))
    
    fig.update_layout(barmode='relative', 
                    title_text='Mass activations for neurons',
                    xaxis_title="Neurons",
                    yaxis_title="Log mass Activations",
                    yaxis_type="log",
                    xaxis = go.XAxis(showticklabels=False),
                    yaxis = go.YAxis(showticklabels=False)
                    )

    # fig.write_image(os.path.join(data_dict["visualize"]["plot_directory"], "mass_activation_plot.pdf"))
    plotly.offline.plot(fig, filename = os.path.join(data_dict["visualize"]["plot_directory"], "mass_activation_plot.pdf"),
                        auto_open=False)
    fig.show()

def freq_analysis_plot(sup_data, unsup_data_epoch1, unsup_data_epoch49, data_dict):
    """
    :param unsup_data_epoch1: Unsupervised data for 1 epoch(dtype:pandas dataframe)
    :param unsup_data_epoch49: Unsupervised data for 1 epoch(dtype:pandas dataframe)
    :param sup_data: Supervised data(dtype:pandas dataframe)
    :param data_dict: dictionary containing input instructions(dtype:dict)
    Plots the frequency analysis plot and save it in data_dict["visualize"]["plot_directory"]
    """
    
    sup = sup_data['POS'].to_dict()
    sup_freq = Counter(sup.values())
    sup_freq = dict(OrderedDict(sup_freq.most_common()))

    unsup1 = unsup_data_epoch1['POS'].to_dict()
    unsup1_freq = Counter(unsup1.values())
    unsup1_freq = dict(OrderedDict(unsup1_freq.most_common()))

    unsup49 = unsup_data_epoch49['POS'].to_dict()
    unsup49_freq = Counter(unsup49.values())
    unsup49_freq = dict(OrderedDict(unsup49_freq.most_common()))

    unsup1_dataframe = pd.DataFrame.from_dict(unsup1_freq, orient='index', columns=['unsup-1'])
    unsup49_dataframe = pd.DataFrame.from_dict(unsup49_freq, orient='index', columns=['unsup-49'])
    sup_dataframe = pd.DataFrame.from_dict(sup_freq, orient='index', columns=['sup'])

    unsup1_pos = list(unsup1_dataframe.index)
    unsup1_pos_mass_activation = []
    for pos in unsup1_pos:
        temp = unsup_data_epoch1[unsup_data_epoch1['POS']==pos]
        unsup1_pos_mass_activation.append(temp['max_activations'].sum())

    sup_pos = list(sup_dataframe.index)
    sup_pos_mass_activation = []
    for pos in sup_pos:
        temp = sup_data[sup_data['POS']==pos]
        sup_pos_mass_activation.append(temp['max_activations'].sum())

    unsup49_pos = list(unsup49_dataframe.index)
    unsup49_pos_mass_activation = []
    for pos in unsup49_pos:
        temp = unsup_data_epoch49[unsup_data_epoch49['POS']==pos]
        unsup49_pos_mass_activation.append(temp['max_activations'].sum())

    unsup1_dataframe['unsup1-mass_activation'] = unsup1_pos_mass_activation
    unsup49_dataframe['unsup49-mass_activation'] = unsup49_pos_mass_activation
    sup_dataframe['sup-mass_activation'] = sup_pos_mass_activation

    df = unsup1_dataframe.join(sup_dataframe)
    df_ = df.join(unsup49_dataframe)

    df_.sort_values(['unsup-1'],inplace=True,ascending=False)
    df_['unsup corpus POS freq. %'] = df_['unsup-1'].apply(lambda x:x/df_['unsup-1'].sum())
    df_['unsup epoch 1 act. mass %'] = df_['unsup1-mass_activation'].apply(lambda x:x/df_['unsup1-mass_activation'].sum())
    df_['unsup epoch 49 act. mass %'] = df_['unsup49-mass_activation'].apply(lambda x:x/df_['unsup49-mass_activation'].sum())

    plot_dict = df_[['unsup corpus POS freq. %','unsup epoch 1 act. mass %','unsup epoch 49 act. mass %']].to_dict()

    fig = go.Figure()
    fig.add_trace(go.Bar(x= list(plot_dict['unsup corpus POS freq. %'].keys()) ,
                         y= list(plot_dict['unsup corpus POS freq. %'].values()), 
                         name="unsup POS freq. %", marker_color='black'))
    fig.add_trace(go.Bar(x= list(plot_dict['unsup epoch 1 act. mass %'].keys()) ,
                         y= list(plot_dict['unsup epoch 1 act. mass %'].values()), 
                         name="unsup epoch 1 act. mass %", marker_color='gray'))
    fig.add_trace(go.Bar(x= list(plot_dict['unsup epoch 49 act. mass %'].keys()) ,
                         y= list(plot_dict['unsup epoch 49 act. mass %'].values()), 
                         name="unsup epoch 49 act. mass %", marker_color=data_dict['visualize']['viz_colors']['unsup_epoch_49']))

    fig.update_layout(barmode='relative', 
                        title_text='% POS activations vs. % POS frequencies',
                        xaxis_title="POS tags",
                        yaxis_title="POS %",
                        )

    # fig.write_image(os.path.join(data_dict["visualize"]["plot_directory"], "mass_activation_plot.pdf"))
    plotly.offline.plot(fig, filename = os.path.join(data_dict["visualize"]["plot_directory"], "freq_activation_plot.pdf"),
                        auto_open=False)
    fig.show()

def hellinger_length_plot(hellinger_stats, filename):
    """
    :param hellinger_stats: path to the savd file for the hellinger statistics from calculate_hellinger_distance function 
    :param filename: file name with directory where the results are to be stored(dtype:str)
    Description: Plots a scatter plot between number of features activated for every neuron vs hellinger distance between 
                the two models
    """

    with open(hellinger_stats, 'rb') as handle:
        hellinger_dict = pickle.load(handle)

    num_token_list, distance_list = ([] for i in range(2))
    for activation,(distance,num_tokens) in hellinger_dict.items():   
        num_token_list.append(num_tokens)
        distance_list.append(distance)

    fig = px.scatter(x= num_token_list ,y= distance_list)
    
    plot_title = str(len(hellinger_dict)) + " neurons activated"
    fig.update_layout(barmode='relative', 
                        title_text=plot_title,
                        xaxis_title="Log Hellinger length",
                        yaxis_title="Hellinger distance",
                        xaxis_type="log",
                        xaxis = go.XAxis(showticklabels=False),
                        yaxis = go.YAxis(showticklabels=False)
                        )
    
    plotly.offline.plot(fig, filename = filename,auto_open=False)
    fig.show()

    
def length_shift_token_plot(model1, model2, modelname1, modelname2, color1, color2, y_axis_label_model1, 
                            y_axis_label_model2, data_dict, filename):
    """
    :param model1:data from trained model 1(dtype:dataframe)
    :param model2:data from trained model 2(dtype:dataframe)
    :param color1:color for model 1(dtype:str)
    :param color2:color for model 2(dtype:str)
    :param modelname1:model1 label(dtype:str)
    :param modelname2:model2 label(dtype:str)
    :param y_axis_label_model1:Y axis label annotation for model1(dtype:str)
    :param y_axis_label_model2:Y axis label annotation for model2(dtype:str)
    :param data_dict: dictionary containing input instructions(dtype:dict)
    :param filename: pickled file name and directory to store the results
    """
        
    fig = go.Figure()
    # Collecting number of tokens in each neurons for both the model
    model1_tokens_dict, model2_tokens_dict = ({} for i in range(2))
    for neuron in list(model1['max_activation_index'].unique()):
        model1_data = model1[model1['max_activation_index'] == neuron]
        model1_tokens_dict[neuron] = model1_data['inputs'].nunique()
    for neuron in list(model2['max_activation_index'].unique()):
        model2_data = model2[model2['max_activation_index'] == neuron]
        model2_tokens_dict[neuron] = model2_data['inputs'].nunique()

    model1_token_list, model1_y_list = ([] for i in range(2))
    model2_token_list, model2_y_list = ([] for i in range(2))
    # plotting scatter plot
    for neuron in range(data_dict['models']['pretrained_lm']['nhid']):
        if neuron in list(model1['max_activation_index'].unique()):
            model1_token_list.append(model1_tokens_dict[neuron])
            model1_y_list.append(y_axis_label_model1)
            
        if neuron in list(model2['max_activation_index'].unique()):
            model2_token_list.append(model2_tokens_dict[neuron])
            model2_y_list.append(y_axis_label_model2)
    
    fig.add_trace(go.Scatter(x=model1_token_list,y= model1_y_list, mode='markers', name=modelname1 ,
                             marker_color=color1))
    fig.add_trace(go.Scatter(x= model2_token_list, y=model2_y_list, mode='markers', name=modelname2,
                             marker_color=color2))
    
    model1_neurons = list(model1['max_activation_index'].unique())
    model2_neurons = list(model2['max_activation_index'].unique())
    shared_neurons = find_shared_neurons(model1_neurons, model2_neurons)
    for neuron in shared_neurons:
        if model1_tokens_dict[neuron] > model2_tokens_dict[neuron]:
            color_ = data_dict['visualize']['viz_colors']['length_reduced']
        elif model1_tokens_dict[neuron] == model2_tokens_dict[neuron]:
            color_ = 'black'
        else:
            color_ = data_dict['visualize']['viz_colors']['length_increased']
        x_,y_ = [model1_tokens_dict[neuron],model2_tokens_dict[neuron]],[y_axis_label_model1 ,y_axis_label_model2]
        fig.add_trace(go.Scatter(x= x_, y=y_, mode='lines', marker_color=color_, name=" "))

    title_text = "Length of " + str(len(shared_neurons)) + " alive neurons"
    fig.update_layout(showlegend=False, title_text=title_text, xaxis_type="log", 
                      xaxis_title="Log number of tokens activated")
    
    plotly.offline.plot(fig, filename = filename, auto_open=False)
    fig.show()

def length_shift_pos_plot(model1, model2, modelname1, modelname2, color1, color2, y_axis_label_model1, 
                            y_axis_label_model2, data_dict, filename):
    """
    :param model1:data from trained model 1(dtype:dataframe)
    :param model2:data from trained model 2(dtype:dataframe)
    :param color1:color for model 1(dtype:str)
    :param color2:color for model 2(dtype:str)
    :param modelname1:model1 label(dtype:str)
    :param modelname2:model2 label(dtype:str)
    :param y_axis_label_model1:Y axis label annotation for model1(dtype:str)
    :param y_axis_label_model2:Y axis label annotation for model2(dtype:str)
    :param data_dict: dictionary containing input instructions(dtype:dict)
    :param filename: pickled file name and directory to store the results
    """
        
    fig = go.Figure()
    # Collecting number of pos in each neurons for both the model
    model1_pos_dict, model2_pos_dict = ({} for i in range(2))
    for neuron in list(model1['max_activation_index'].unique()):
        model1_data = model1[model1['max_activation_index'] == neuron]
        model1_pos_dict[neuron] = model1_data['POS'].nunique()
    for neuron in list(model2['max_activation_index'].unique()):
        model2_data = model2[model2['max_activation_index'] == neuron]
        model2_pos_dict[neuron] = model2_data['POS'].nunique()

    model1_token_list, model1_y_list = ([] for i in range(2))
    model2_token_list, model2_y_list = ([] for i in range(2))
    # plotting scatter plot
    for neuron in range(data_dict['models']['pretrained_lm']['nhid']):
        if neuron in list(model1['max_activation_index'].unique()):
            model1_token_list.append(model1_pos_dict[neuron])
            model1_y_list.append(y_axis_label_model1)
            
        if neuron in list(model2['max_activation_index'].unique()):
            model2_token_list.append(model2_pos_dict[neuron])
            model2_y_list.append(y_axis_label_model2)
    
    fig.add_trace(go.Scatter(x=model1_token_list,y= model1_y_list, mode='markers', name=modelname1 ,
                             marker_color=color1))
    fig.add_trace(go.Scatter(x= model2_token_list, y=model2_y_list, mode='markers', name=modelname2,
                             marker_color=color2))
    
    model1_neurons = list(model1['max_activation_index'].unique())
    model2_neurons = list(model2['max_activation_index'].unique())
    shared_neurons = find_shared_neurons(model1_neurons, model2_neurons)
    for neuron in shared_neurons:
        if model1_pos_dict[neuron] > model2_pos_dict[neuron]:
            color_ = data_dict['visualize']['viz_colors']['length_reduced']
        elif model1_pos_dict[neuron] == model2_pos_dict[neuron]:
            color_ = 'black'
        else:
            color_ = data_dict['visualize']['viz_colors']['length_increased']
        x_,y_ = [model1_pos_dict[neuron],model2_pos_dict[neuron]],[y_axis_label_model1 ,y_axis_label_model2]
        fig.add_trace(go.Scatter(x= x_, y=y_, mode='lines', marker_color=color_, name=" "))

    title_text = "Length of " + str(len(shared_neurons)) + " alive neurons"
    fig.update_layout(showlegend=False, title_text=title_text, 
                      xaxis_title="number of POS activated")
    
    plotly.offline.plot(fig, filename = filename, auto_open=False)
    fig.show()

def choose_top_pos_from_data(df):
    """
    :param df: dataframe(dtype:pandas dataframe)
    :returns: a dict with the top three pos tags associated with a token in the entire dataset.
    """
    counter_dict = {}
    unique_tokens = df['inputs'].unique()
    for token in unique_tokens:
        temp = df[df['inputs']==token]
        temp_pos = list(temp['POS'])
        temp_pos = [tag.strip() for tag in temp_pos]
        tags = Counter(temp_pos)
        most_common_tags = tags.most_common(3)
        most_common_tags = [tags[0] for tags in most_common_tags]
        counter_dict[token] = most_common_tags
    return counter_dict



def plot_top_10_hellinger_neurons(hellinger_stats, model1_data, model2_data, color1, color2, modelname1, modelname2, 
                                  data_dict, foldername, n_tokens=0, process_data_flag=False):
    """
    :param hellinger_stats: path to the savd file for the hellinger statistics from calculate_hellinger_distance function 
    :param model1_data:data from trained model 1(dtype:dataframe)
    :param model2_data:data from trained model 2(dtype:dataframe)
    :param color1:color for model 1(dtype:str)
    :param color2:color for model 2(dtype:str)
    :param modelname1:model1 label(dtype:str)
    :param modelname2:model2 label(dtype:str)
    :param data_dict: dictionary containing input instructions(dtype:dict)
    :param foldername: pickled file name and directory to store the results
    :param n_tokens: number of tokens you want to plot(dtype:int)
    :param process_data_flag: True if the pickle files need to be generated, False if you want to load the pickle 
                              files.
    :Description: Generates the plot for the top 10 neurons with highest hellinger distances in hellinger_stats
    """
    # removing the whitespaces
    model1_data['POS'] = model1_data['POS'].apply(lambda x:x.replace(" ",""))
    model2_data['POS'] = model2_data['POS'].apply(lambda x:x.replace(" ",""))
    
    # Getting all the POS tags activated
    model1_pos = list(model1_data['POS'].unique())
    model1_pos = list(model2_data['POS'].unique())
    all_pos = set(model1_pos + model1_pos)
    # all_pos = [pos.strip() for pos in all_pos]
    
    # loading the Hellinger distance dictionary
    with open(hellinger_stats, 'rb') as handle:
        hellinger_dict = pickle.load(handle)
        
    top_10_neurons = heapq.nlargest(10, hellinger_dict, key=hellinger_dict.get)
    for neuron in top_10_neurons:
        path = os.path.join(data_dict["visualize"]["plot_directory"],foldername,"top_10",str(neuron))
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        model1_data_temp = model1_data[model1_data['max_activation_index']==neuron]
        model2_data_temp = model2_data[model2_data['max_activation_index']==neuron]
                
        # Getting the pos stats from all the dictionaries
        model1_pos_dict = dict(Counter(model1_data_temp['POS']))
        model2_pos_dict = dict(Counter(model2_data_temp['POS']))
        # Creating dataframe from the dictionaries
        model1_pos = pd.DataFrame.from_dict(model1_pos_dict, orient='index', columns=[modelname1])
        model2_pos = pd.DataFrame.from_dict(model2_pos_dict, orient='index', columns=[modelname2])
        # Normalizing the statistics
        model1_pos[modelname1] = model1_pos[modelname1].apply(lambda x: x/model1_pos[modelname1].sum())
        model2_pos[modelname2] = model2_pos[modelname2].apply(lambda x: x/model2_pos[modelname2].sum())
        # Merging dataframe
        data = [model1_pos[modelname1], model2_pos[modelname2]]
        df = pd.concat(data,axis=1)
        # Again converting the dataframe to dictionary for further computations.
        all_pos_stats = df.to_dict()
        
        # Getting all the pos stats into a dictionary
        for viz_data in all_pos_stats.keys():
            for tags in all_pos:
                if tags not in all_pos_stats[viz_data].keys():
                    all_pos_stats[viz_data][tags] = None
            
        # Converting pos stats to a dataframe
        # all_pos_stats = pd.DataFrame.from_dict(all_pos_stats)
        
        if process_data_flag == True:
            # Getting the data.
            model1_neurondata = model1_data[model1_data['max_activation_index']==neuron]
            model1_neurondata['POS'] = model1_neurondata['POS'].apply(lambda x: x.strip())
            model2_neurondata = model2_data[model2_data['max_activation_index']==neuron]
            model2_neurondata['POS'] = model2_neurondata['POS'].apply(lambda x: x.strip())
            
            # Converting the other pos tags to the top three ones
            model1_top_pos = choose_top_pos_from_data(model1_neurondata)
            model2_top_pos = choose_top_pos_from_data(model2_neurondata)
            
            model1_tokens = list(model1_neurondata['inputs'])
            model1_pos = list(model1_neurondata['POS'])
            model2_tokens = list(model2_neurondata['inputs'])
            model2_pos = list(model2_neurondata['POS'])

            for index, pos in enumerate(model1_pos):
                if pos not in model1_top_pos[model1_tokens[index]]:
                    model1_pos[index] = model1_top_pos[model1_tokens[index]][0]
            for index, pos in enumerate(model2_pos):
                if pos not in model2_top_pos[model2_tokens[index]]:
                    model2_pos[index] = model2_top_pos[model2_tokens[index]][0]
                    
            model1_neurondata['POS'] = model1_pos
            model2_neurondata['POS'] = model2_pos
            
            # Getting all the unique tokens
            model1_unique_tokens = model1_neurondata["inputs"].unique()
            model2_unique_tokens = model2_neurondata["inputs"].unique()
            
            model1_dict,model2_dict = ({} for i in range(2))
            
            # Generating model1 visualization
            # Getting mean for all the unique tokens
            for tokens in model1_unique_tokens:
                temp_df = model1_neurondata[model1_neurondata["inputs"] == tokens]
                pos = list(temp_df["POS"].unique())
                activation_temp = []
                for unique_pos in pos:
                    activation_temp.append(temp_df[temp_df['POS']==unique_pos]["max_activations"].mean())
                model1_dict[tokens] = {"POS":pos, "activation":activation_temp}
            
            # Getting the top 20 activation tokens
            model1_top_20 = {}
            temp_activations, temp_tokens = ([] for i in range(2))
            for key, value in model1_dict.items():
                for index in range(len(value['POS'])):
                    temp_tokens.append(key)
                    temp_activations.append(value['activation'][index])      
            model1_top_20_activation_index = sorted(range(len(temp_activations)), key=lambda x: temp_activations[x])[-n_tokens:]
            for indexes in model1_top_20_activation_index:
                model1_top_20[temp_tokens[indexes]] = model1_dict[temp_tokens[indexes]]
            
            # Flipping the dictionary to get it in the order of {pos-tags:list(tuple(token,mean_activations))}
            model1_token_dict = defaultdict(list)
            for token,stats in model1_top_20.items():
                for index,value in enumerate(stats['POS']):
                    model1_token_dict[stats['POS'][index]].append((token,stats['activation'][index]))
            
            # Adding the null features for the tags not present
            for tags in all_pos:
                if tags not in model1_token_dict.keys():
                    model1_token_dict[tags].append((' ',0.0))

            # Sorting dict on the basis of the names
            sorted_model1_dict = {}
            for key in sorted(model1_token_dict.keys()):
                sorted_model1_dict[key] = model1_token_dict[key]
                
            with open(os.path.join(path,'model1_data.pickle'), 'wb') as handle:
                pickle.dump(sorted_model1_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            # Generating model2 visualization
            # Getting mean for all the unique tokens
            for tokens in model2_unique_tokens:
                temp_df = model2_neurondata[model2_neurondata["inputs"] == tokens]
                pos = list(temp_df["POS"].unique())
                activation_temp = []
                for unique_pos in pos:
                    activation_temp.append(temp_df[temp_df['POS']==unique_pos]["max_activations"].mean())
                model2_dict[tokens] = {"POS":pos, "activation":activation_temp}
            
            # Getting the top 20 activation tokens
            model2_top_20 = {}
            temp_activations, temp_tokens = ([] for i in range(2))
            for key, value in model2_dict.items():
                for index in range(len(value['POS'])):
                    temp_tokens.append(key)
                    temp_activations.append(value['activation'][index])      
            model2_top_20_activation_index = sorted(range(len(temp_activations)), key=lambda x: temp_activations[x])[-n_tokens:]
            for indexes in model2_top_20_activation_index:
                model2_top_20[temp_tokens[indexes]] = model2_dict[temp_tokens[indexes]]
            
            # Flipping the dictionary to get it in the order of {pos-tags:list(tuple(token,mean_activations))}
            model2_token_dict = defaultdict(list)
            for token,stats in model2_top_20.items():
                for index,value in enumerate(stats['POS']):
                    model2_token_dict[stats['POS'][index]].append((token,stats['activation'][index]))
            
            # Adding the null features for the tags not present
            for tags in all_pos:
                if tags not in model2_token_dict.keys():
                    model2_token_dict[tags].append((' ',0.0))

            # Sorting dict on the basis of the names
            sorted_model2_dict = {}
            for key in sorted(model2_token_dict.keys()):
                sorted_model2_dict[key] = model2_token_dict[key]
                
            with open(os.path.join(path,'model2_data.pickle'), 'wb') as handle:
                pickle.dump(sorted_model2_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        else:
            # loading the dictionary
            with open(os.path.join(path,'model1_data.pickle'), 'rb') as handle:
                sorted_model1_dict = pickle.load(handle)
            with open(os.path.join(path,'model2_data.pickle'), 'rb') as handle:
                sorted_model2_dict = pickle.load(handle)
                
        fig = go.Figure()
        # Plotting the bar plot
        fig.add_trace(go.Bar(x=list(all_pos_stats[modelname1].keys()), y=list(all_pos_stats[modelname1].values()), 
                             name=modelname1, marker_color=color1, opacity=0.6))
        fig.add_trace(go.Bar(x=list(all_pos_stats[modelname2].keys()), y=list(all_pos_stats[modelname2].values()), 
                             name=modelname2, marker_color=color2, opacity=0.6))

        # Plotting the tokens on the bar plot
        pos_model1 = list(sorted_model1_dict.keys())
        values_model1 = list(sorted_model1_dict.values())

        pos_model2 = list(sorted_model2_dict.keys())
        values_model2 = list(sorted_model2_dict.values())
        model1_value = [[(value[0],np.nan) if value[1]==0.0 else (value[0],value[1]) for value in pairs] for pairs in values_model1]
        model2_value = [[(value[0],np.nan) if value[1]==0.0 else (value[0],value[1]) for value in pairs] for pairs in values_model2]

        model1_token = [[value[0] for value in pairs] for pairs in model1_value]
        model1_activations = [[value[1] for value in pairs] for pairs in model1_value]

        model2_token = [[value[0] for value in pairs] for pairs in model2_value]
        model2_activations = [[value[1] for value in pairs] for pairs in model2_value]

        pos_model1_list, activation_model1_list, token_model1_list = ([] for i in range(3))
        for index in range(len(pos_model1)):
            for activation_list_index, activation in enumerate(model1_activations[index]):
                if activation >= 0.0:
                	pos_model1_list.append(pos_model1[index])
                	activation_model1_list.append(activation)
                	token_model1_list.append(model1_token[index][activation_list_index])
        fig.add_trace(go.Scatter(x=pos_model1_list, y=activation_model1_list, text=token_model1_list, 
                                 mode='markers+text', marker_color=color1, name=modelname1, 
                                 textfont={'color':color1}))

        pos_model2_list, activation_model2_list, token_model2_list = ([] for i in range(3))
        for index in range(len(pos_model2)):
            for activation_list_index, activation in enumerate(model2_activations[index]):
            	if activation >= 0.0:
                	pos_model2_list.append(pos_model2[index])
                	activation_model2_list.append(activation)
                	token_model2_list.append(model2_token[index][activation_list_index])
        fig.add_trace(go.Scatter(x=pos_model2_list, y=activation_model2_list, text=token_model2_list, 
                                 mode='markers+text', marker_color=color2, name=modelname2, 
                                 textfont={'color':color2}))
        
        fig.update_layout(title_text='Hellinger plot for ' + str(neuron) + "-neuron" ,
                    xaxis_title="POS-tags",
                    yaxis_title="Activation",
                    xaxis = go.XAxis(showticklabels=True),
                    yaxis = go.YAxis(showticklabels=True)
                    )
        
        plotly.offline.plot(fig, filename = os.path.join(path,str(neuron)+".pdf"), auto_open=False)
        fig.show()
        
def plot_least_10_hellinger_neurons(hellinger_stats, model1_data, model2_data, color1, color2, modelname1, modelname2, 
                                  data_dict, foldername, n_tokens=0, process_data_flag=False):
    """
    :param hellinger_stats: path to the savd file for the hellinger statistics from calculate_hellinger_distance function 
    :param model1_data:data from trained model 1(dtype:dataframe)
    :param model2_data:data from trained model 2(dtype:dataframe)
    :param color1:color for model 1(dtype:str)
    :param color2:color for model 2(dtype:str)
    :param modelname1:model1 label(dtype:str)
    :param modelname2:model2 label(dtype:str)
    :param data_dict: dictionary containing input instructions(dtype:dict)
    :param foldername: pickled file name and directory to store the results
    :param n_tokens: number of tokens you want to plot(dtype:int)
    :param process_data_flag: True if the pickle files need to be generated, False if you want to load the pickle 
                              files.
    :Description: Generates the plot for the least 10 neurons with highest hellinger distances in hellinger_stats
    """
    # removing the whitespaces
    model1_data['POS'] = model1_data['POS'].apply(lambda x:x.replace(" ",""))
    model2_data['POS'] = model2_data['POS'].apply(lambda x:x.replace(" ",""))
    
    # Getting all the POS tags activated
    model1_pos = list(model1_data['POS'].unique())
    model1_pos = list(model2_data['POS'].unique())
    all_pos = set(model1_pos + model1_pos)
    # all_pos = [pos.strip() for pos in all_pos]
    
    # loading the Hellinger distance dictionary
    with open(hellinger_stats, 'rb') as handle:
        hellinger_dict = pickle.load(handle)
        
    least_10_neurons = heapq.nsmallest(10, hellinger_dict, key=hellinger_dict.get)
    for neuron in least_10_neurons:
        path = os.path.join(data_dict["visualize"]["plot_directory"],foldername,"least_10",str(neuron))
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        model1_data_temp = model1_data[model1_data['max_activation_index']==neuron]
        model2_data_temp = model2_data[model2_data['max_activation_index']==neuron]
                
        # Getting the pos stats from all the dictionaries
        model1_pos_dict = dict(Counter(model1_data_temp['POS']))
        model2_pos_dict = dict(Counter(model2_data_temp['POS']))
        # Creating dataframe from the dictionaries
        model1_pos = pd.DataFrame.from_dict(model1_pos_dict, orient='index', columns=[modelname1])
        model2_pos = pd.DataFrame.from_dict(model2_pos_dict, orient='index', columns=[modelname2])
        # Normalizing the statistics
        model1_pos[modelname1] = model1_pos[modelname1].apply(lambda x: x/model1_pos[modelname1].sum())
        model2_pos[modelname2] = model2_pos[modelname2].apply(lambda x: x/model2_pos[modelname2].sum())
        # Merging dataframe
        data = [model1_pos[modelname1], model2_pos[modelname2]]
        df = pd.concat(data,axis=1)
        # Again converting the dataframe to dictionary for further computations.
        all_pos_stats = df.to_dict()
        
        # Getting all the pos stats into a dictionary
        for viz_data in all_pos_stats.keys():
            for tags in all_pos:
                if tags not in all_pos_stats[viz_data].keys():
                    all_pos_stats[viz_data][tags] = None
            
        # Converting pos stats to a dataframe
        # all_pos_stats = pd.DataFrame.from_dict(all_pos_stats)
        
        if process_data_flag == True:
            # Getting the data.
            model1_neurondata = model1_data[model1_data['max_activation_index']==neuron]
            model1_neurondata['POS'] = model1_neurondata['POS'].apply(lambda x: x.strip())
            model2_neurondata = model2_data[model2_data['max_activation_index']==neuron]
            model2_neurondata['POS'] = model2_neurondata['POS'].apply(lambda x: x.strip())
            
            # Converting the other pos tags to the least three ones
            model1_least_pos = choose_top_pos_from_data(model1_neurondata)
            model2_least_pos = choose_top_pos_from_data(model2_neurondata)
            
            model1_tokens = list(model1_neurondata['inputs'])
            model1_pos = list(model1_neurondata['POS'])
            model2_tokens = list(model2_neurondata['inputs'])
            model2_pos = list(model2_neurondata['POS'])

            for index, pos in enumerate(model1_pos):
                if pos not in model1_least_pos[model1_tokens[index]]:
                    model1_pos[index] = model1_least_pos[model1_tokens[index]][0]
            for index, pos in enumerate(model2_pos):
                if pos not in model2_least_pos[model2_tokens[index]]:
                    model2_pos[index] = model2_least_pos[model2_tokens[index]][0]
                    
            model1_neurondata['POS'] = model1_pos
            model2_neurondata['POS'] = model2_pos
            
            # Getting all the unique tokens
            model1_unique_tokens = model1_neurondata["inputs"].unique()
            model2_unique_tokens = model2_neurondata["inputs"].unique()
            
            model1_dict,model2_dict = ({} for i in range(2))
            
            # Generating model1 visualization
            # Getting mean for all the unique tokens
            for tokens in model1_unique_tokens:
                temp_df = model1_neurondata[model1_neurondata["inputs"] == tokens]
                pos = list(temp_df["POS"].unique())
                activation_temp = []
                for unique_pos in pos:
                    activation_temp.append(temp_df[temp_df['POS']==unique_pos]["max_activations"].mean())
                model1_dict[tokens] = {"POS":pos, "activation":activation_temp}
            
            # Getting the least 20 activation tokens
            model1_least_20 = {}
            temp_activations, temp_tokens = ([] for i in range(2))
            for key, value in model1_dict.items():
                for index in range(len(value['POS'])):
                    temp_tokens.append(key)
                    temp_activations.append(value['activation'][index])      
            model1_least_20_activation_index = sorted(range(len(temp_activations)), key=lambda x: temp_activations[x])[-n_tokens:]
            for indexes in model1_least_20_activation_index:
                model1_least_20[temp_tokens[indexes]] = model1_dict[temp_tokens[indexes]]
            
            # Flipping the dictionary to get it in the order of {pos-tags:list(tuple(token,mean_activations))}
            model1_token_dict = defaultdict(list)
            for token,stats in model1_least_20.items():
                for index,value in enumerate(stats['POS']):
                    model1_token_dict[stats['POS'][index]].append((token,stats['activation'][index]))
            
            # Adding the null features for the tags not present
            for tags in all_pos:
                if tags not in model1_token_dict.keys():
                    model1_token_dict[tags].append((' ',0.0))

            # Sorting dict on the basis of the names
            sorted_model1_dict = {}
            for key in sorted(model1_token_dict.keys()):
                sorted_model1_dict[key] = model1_token_dict[key]
                
            with open(os.path.join(path,'model1_data.pickle'), 'wb') as handle:
                pickle.dump(sorted_model1_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            # Generating model2 visualization
            # Getting mean for all the unique tokens
            for tokens in model2_unique_tokens:
                temp_df = model2_neurondata[model2_neurondata["inputs"] == tokens]
                pos = list(temp_df["POS"].unique())
                activation_temp = []
                for unique_pos in pos:
                    activation_temp.append(temp_df[temp_df['POS']==unique_pos]["max_activations"].mean())
                model2_dict[tokens] = {"POS":pos, "activation":activation_temp}
            
            # Getting the least 20 activation tokens
            model2_least_20 = {}
            temp_activations, temp_tokens = ([] for i in range(2))
            for key, value in model2_dict.items():
                for index in range(len(value['POS'])):
                    temp_tokens.append(key)
                    temp_activations.append(value['activation'][index])      
            model2_least_20_activation_index = sorted(range(len(temp_activations)), key=lambda x: temp_activations[x])[-n_tokens:]
            for indexes in model2_least_20_activation_index:
                model2_least_20[temp_tokens[indexes]] = model2_dict[temp_tokens[indexes]]
            
            # Flipping the dictionary to get it in the order of {pos-tags:list(tuple(token,mean_activations))}
            model2_token_dict = defaultdict(list)
            for token,stats in model2_least_20.items():
                for index,value in enumerate(stats['POS']):
                    model2_token_dict[stats['POS'][index]].append((token,stats['activation'][index]))
            
            # Adding the null features for the tags not present
            for tags in all_pos:
                if tags not in model2_token_dict.keys():
                    model2_token_dict[tags].append((' ',0.0))

            # Sorting dict on the basis of the names
            sorted_model2_dict = {}
            for key in sorted(model2_token_dict.keys()):
                sorted_model2_dict[key] = model2_token_dict[key]
                
            with open(os.path.join(path,'model2_data.pickle'), 'wb') as handle:
                pickle.dump(sorted_model2_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        else:
            # loading the dictionary
            with open(os.path.join(path,'model1_data.pickle'), 'rb') as handle:
                sorted_model1_dict = pickle.load(handle)
            with open(os.path.join(path,'model2_data.pickle'), 'rb') as handle:
                sorted_model2_dict = pickle.load(handle)
                
        fig = go.Figure()
        # Plotting the bar plot
        fig.add_trace(go.Bar(x=list(all_pos_stats[modelname1].keys()), y=list(all_pos_stats[modelname1].values()), 
                             name=modelname1, marker_color=color1, opacity=0.6))
        fig.add_trace(go.Bar(x=list(all_pos_stats[modelname2].keys()), y=list(all_pos_stats[modelname2].values()), 
                             name=modelname2, marker_color=color2, opacity=0.6))

        # Plotting the tokens on the bar plot
        pos_model1 = list(sorted_model1_dict.keys())
        values_model1 = list(sorted_model1_dict.values())

        pos_model2 = list(sorted_model2_dict.keys())
        values_model2 = list(sorted_model2_dict.values())
        model1_value = [[(value[0],np.nan) if value[1]==0.0 else (value[0],value[1]) for value in pairs] for pairs in values_model1]
        model2_value = [[(value[0],np.nan) if value[1]==0.0 else (value[0],value[1]) for value in pairs] for pairs in values_model2]

        model1_token = [[value[0] for value in pairs] for pairs in model1_value]
        model1_activations = [[value[1] for value in pairs] for pairs in model1_value]

        model2_token = [[value[0] for value in pairs] for pairs in model2_value]
        model2_activations = [[value[1] for value in pairs] for pairs in model2_value]

        pos_model1_list, activation_model1_list, token_model1_list = ([] for i in range(3))
        for index in range(len(pos_model1)):
            for activation_list_index, activation in enumerate(model1_activations[index]):
                pos_model1_list.append(pos_model1[index])
                activation_model1_list.append(activation)
                token_model1_list.append(model1_token[index][activation_list_index])
        fig.add_trace(go.Scatter(x=pos_model1_list, y=activation_model1_list, text=token_model1_list, 
                                 mode='markers+text', marker_color=color1, name=modelname1, 
                                 textfont={'color':color1}))

        pos_model2_list, activation_model2_list, token_model2_list = ([] for i in range(3))
        for index in range(len(pos_model2)):
            for activation_list_index, activation in enumerate(model2_activations[index]):
                pos_model2_list.append(pos_model2[index])
                activation_model2_list.append(activation)
                token_model2_list.append(model2_token[index][activation_list_index])
        fig.add_trace(go.Scatter(x=pos_model2_list, y=activation_model2_list, text=token_model2_list, 
                                 mode='markers+text', marker_color=color2, name=modelname2, 
                                 textfont={'color':color2}))
        
        fig.update_layout(title_text='Hellinger plot for ' + str(neuron) + "-neuron" ,
                    xaxis_title="POS-tags",
                    yaxis_title="Activation",
                    xaxis = go.XAxis(showticklabels=True),
                    yaxis = go.YAxis(showticklabels=True)
                    )
        
        plotly.offline.plot(fig, filename = os.path.join(path,str(neuron)+".pdf"), auto_open=False)
        fig.show()
