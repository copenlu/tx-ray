# TX-Ray: Quantifying and Explaining Model-Knowledge Transfer in (Un-)Supervised NLP ([paper](https://arxiv.org/abs/1912.00982))

TX-Ray is a method to better understand how models learn by visualizing (inspecting), quantifying and analyzing model neuron knowledge building and adaptation. It can be used to compare model knowledge during training epochs, fine-tuning and between architectures. TX-Ray applies the principle of ''visualizing what input features a neuron prefers'', commonly known as activation maximization, and adapts the method for use in NLP -- i.e. to visualize which words each neuron prefers. It does so by aggregating maximal layer activations and connecting them to the discrete input (or output) features that produced the maximum. The resulting feature activation distributions can be used to quantify knowledge change (forgetting and transfer) using distribution distance measures, which works without requiring probing tasks. One can also visualize which features each neuron prefers in a learning stage and explore semantics learned by neurons, which is especially interesting for discovering what knwoledge abstraction (task solving building blocks) self-supervised models learn, which unlike probing tasks/ diagnostic classifiers etc. allow to discover unforseen learning effects.

With TX-Ray, we provide a set of visualizations and measures and that support both *overview and detail* analysis. There are plots that guide users by allowing them to find interesting neurons and then visualize which knowledge these neurons abstract and how it changes during learning.

This repository currently shows how to use TX-Ray for RNNs, using all or only the most active activations for analyis. In the [paper](https://arxiv.org/abs/1912.00982), we focus on the most active neurons -- max-1 activations.

# Dataset
1. [Wikitext-2](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip)
2. [IMDB movies review dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

# Usage example
1. open [Examples/LSTM/LSTM.ipynb](Examples/LSTM/LSTM.ipynb) -- Note `github does not render plotly plots`, but jupyter does
1. download_data.sh downloads the data from the source and unzips it. 
2. prepare_data.py takes the IMDB data from raw files and converts in into a .csv file that contains features (sec 2.1 in the paper).
3. The example pre-trains an LSTM based sequence encoder on the Wikitext-2 dataset and than fine tunes the encoder using a classifier on the IMDB movies review dataset -- classic transfer learning setup. The folder Examples/LSTM/ contains lstm_lm.py and lstm_classifier.py files to train the sequence encoder and classifier models respectively. These models can also be trained by utilizing the example Jupyter Notebook LSTM.ipynb and setting train_model_flag to True. In the notebook, a dictionary are used to configure model settings.
4. To generate the visualizations, the data has to be aggregated from the trained models for every instance. This can be done by setting collect_data_flag to True value.

## overview (pick neurons and model-level insights) plot types
*Note: plots are in plotly (interactive)*, *not in matplotlib* as in the paper. We chose plotly as backend, as it integrates with [Weights and Biases](https://www.wandb.com/) and Facebooks [visdom](https://github.com/facebookresearch/visdom)
+ calculate_hellinger_distance computes Hellinger distance between the shared neurons for two models or the pre-trained vs. fine tuned training stage. The results are later saved to the directory provided. Turn on the calculate_distance_flag to True value to compute the distance.
+ hellinger_length_plot plots a scatter plot between Hellinger length to the Hellinger distance for every neuron from calculate_hellinger_distance. Hellinger length is the number of unique features activated between 2 two neuron distributions -- i.e. the union of distribition d_1 OR distribition d_2 features, e.g. d_2={I. like}, d_2={I, cookies} -> Hellinger length 3 because d_1 OR d_2 = {I, like, cookies}.
<p align="center">
  <img src="/Examples/LSTM/plots/hellinger_length.png" width=630 title="hover text">

+ length_shift_pos_plot and length_shift_token_plot generates a scattered-line plot showing how the lenght of a neuron (the number of its unique features), changes during learning or between models -- e.g. for POS tags or unique token featues. Neuron shortening indicates specialization, lengthening indicates a more general response.
<p align="center">
  <img src="/Examples/LSTM/plots/length_shift_plot.png" width=630 title="hover text">


+ freq_analysis_plot compares the original POS tag frequencies (of IMDB) to the %POS activations mass at the early and late stages of training of encoder pretraining. 
<p align="center">
  <img src="/Examples/LSTM/plots/act_freq.png" width="630" title="hover text">
  
+ mass_activation_plot plots the normalized **activation mass** for all the neurons in the unsupervised, zero-shot, and supervised case.
<p align="center">
  <img src="/Examples/LSTM/plots/mass_activation.png" width="630" title="hover text">
  
## detail (zoom in to see what knowledge neurons abstract) plot types  
+ plot_top_10_hellinger_neurons and plot_least_10_hellinger_neurons generate visualizations for the neurons with most and least 10 Hellinger distances between training stage or model (saved as neuron distribitions).
<p align="center">
  <img src="/Examples/LSTM/plots/top10.png" width="630" title="hover text">
</p>
<p align="center">
  <img src="/Examples/LSTM/plots/least10.png" width="630" title="hover text">
</p>

# Paper and bibtex Reference (*bibtex currently outdated* -- proper version to appear after UAI 2020)
[TX-Ray: Quantifying and Explaining Model-Knowledge Transfer in (Un-)Supervised NLP](https://arxiv.org/abs/1912.00982), Nils Rethmeier, Vageesh Saxena, and Isabelle Augenstein
```
@inproceedings{rethmeier2019txray,
  title={TX-Ray: Quantifying and Explaining Model-Knowledge Transfer in (Un-)Supervised NLP},
  author={Nils Rethmeier and Vageesh Kumar Saxena and Isabelle Augenstein},
  booktitle = {Proceedings of the Thirty-Sixth Conference on Uncertainty in Artificial
               Intelligence, {UAI} 2020, Toronto, Canada, August 03-06, 2020},
  publisher = {{AUAI} Press},
  year      = {2020},
  url       = {https://arxiv.org/pdf/1912.00982.pdf},
}
```
