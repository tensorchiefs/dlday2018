## Using LSTM for Modelling Wind Plant Power Performance
Braulio Barahona, Lilach Goren Huber (ZHAW)

Modelling the power performance of wind power plants has several applications, 
from improving maintenance operations to predictions of power production. Given the large 
amount of data that is collected by the control units at the turbine and the plant level, 
the application of deep learning is very attractive. Although the wind plant itself operates 
essentially according to design, it does so under environmental conditions that constantly change. 
Moreover, its operation might also be influenced by conditions of the power network. Making it hard to 
accurately model the power output. Here, we describe the application of Long Short-Term Memory (LSTM)
artificial neural networks (ANNs) to model power output and operational parameters at the turbine and 
at the plant level. LSTMs, as well as other recurrent ANNs, offer the possibility to capture time 
dependencies, thus we analyze the capability of deep architectures at different time scales. 
We present a workflow to select the network architecture and to train it. The workflow comprehends 
common steps applied when fitting a statistical model to data and expert knowledge of the system 
operation. In order to demonstrate the performance of the model we use an open access data set of 
a real wind farm in operation. The performance of our models is compared also to common industry 
practices for predicting the power output.

## Towards automatic claims assessment for car insurance
Teresa Kubacka, Mara Nägelin (Swiss Mobiliar)

In recent years, deep learning has undergone a dramatic development. As a technology, deep learning has become mature enough so that it is increasingly more common for companies, whose core business lies outside of a strictly tech domain, to use deep-learning-based tools to support internal processes. Main benefits are focused around optimization of cost, time and resources, decision-making support, transfer of expert knowledge and better user experience.   
In car insurance, every claim is typically accompanied with a set of images documenting the extent of the damage, and structured information related to the car parts, which have undergone reparation or replacement. Due to large amount of cases processed each year, the abundance of data makes car claim assessment a good candidate for process optimization using deep learning techniques. 
In this poster we discuss a POC of a machine learning pipeline for automatic claims assessment realized using image recognition based on Tensorflow. We show how object detection and instance segmentation can help with different parts of the claims assessment process. We present quick wins as well as the most challenging aspects of using machine-learning based solution for this purpose. We discuss ways of dealing with complexity present in the dataset and the importance of human labelers with the domain expertise for the output of the model.

## Transductive Label Augmentation for Improved Deep Network Learning
Ismail Elezi (ZHAW)

Abstract: A major impediment to the application of deep learning to real-world problems is the scarcity of labeled data. Small training sets are in fact of no use to deep networks as, due to the large number of trainable parameters, they will very likely be subject to overfitting phenomena. On the other hand, the increment of the training set size through further manual or semi-automatic labellings can be costly, if not possible at times. Thus, the standard techniques to address this issue are transfer learning and data augmentation, which consists of applying some sort of "transformation" to existing labeled instances to let the training set grow in size. Although this approach works well in applications such as image classification, where it is relatively simple to design suitable transformation operators, it is not obvious how to apply it in more structured scenarios. Motivated by the observation that in virtually all application domains it is easy to obtain unlabeled data, in this paper we take a different perspective and propose a **label augmentation** approach. We start from a small, curated labeled dataset and let the labels propagate through a larger set of unlabeled data using graph transduction techniques. This allows us to naturally use (second-order) similarity information which resides in the data, a source of information which is typically neglected by standard augmentation techniques. In particular, we show that by using known game theoretic transductive processes we can create larger and accurate enough labeled datasets which use results in better trained neural networks. Preliminary experiments are reported which demonstrate a consistent improvement over standard image classification datasets.

## Rediscovering ICA properties in neural networks applied to mixtures of signals problems
Matthias Hermann, Michael Grunwald and Thomas Gnädig (HTWG-Konstanz,IOS)

Despite their great success, there is still no comprehensive theoretical 
understanding of learning with Deep Neural Networks (DNNs) and their 
internal characteristics.  Previous works proposed to analyze neural 
networks from a information theoretical point of view (Tishby and 
Zaslavsky, 2015; Schwartz-Ziv and Tishby, 2017).
This work follows this idea and rediscovers properties similar to 
Independent Component Analysis (ICA) in neural networks.
While neural networks are capable of classifying (e.g. cross-entropy) or 
separating source signals (e.g. mean-squared-error) through optimization 
a good interpretation on how this is working is often missing. We 
believe that in order to accomplish these tasks, neural networks need to 
perform something that is similar to ICA during training phase. To proof 
this behavior we generate synthetic mixtures of signals with a static 
mixture matrix and learn classifying them into (0) mixed signal and (1) 
unmixed signal. Throughout experiments we show that neural networks with 
non-linear activation functions indeed learn concepts that correspond to 
unmixing matrices and hence are able to generalize to both seen and 
unseen signal mixtures. Further results show that neural networks 
maximize kurtosis (non-Gaussianity) in deeper layers which further 
relates the training of neural networks to the objective used in ICA.

## Arthritis Net - Automated bone erosion scoring for rheumatoid arthritis with deep convolutional neural networks
Janick Rohrbach (ZHAW)

Rheumatoid arthritis can cause irreversible damage to the joints. The severity of those bone erosions is scored by using x-ray images. This is usually done by a trained rheumatologist or radiologist and takes several minutes per patient. 
This poster shows an automated method to score the joints in x-ray images with deep convolutional neural networks. We take a classification and a regression approach on x-ray images of joints from the left hand. In the classification task, we predict the Ratingen-score on a discrete integer scale from 0 to 5. 
The model achieves class average validation and test accuracies of 42% and 43% respectively. The class average accuracies for predictions that are off by no more than 1 class are 82% for the validation set and 83% for the test set. 
The regression model predicts the continuous percentage of bone erosion between 0% and 100% with a validation and test mean squared error of 72.8 and 97.6 respectively. The mean absolute error is 3.1 for the validation set and 3.5 for the test set. 
An automated scoring of bone erosion could help rheumatologists to spend less time with the scoring and have more time with the patient.

## Sound Classification with CNN on Low Power Embedded platforms
Simon Vogel (ZHAW) 

In this paper, we explored different low power embedded platforms concerning their efficiency by analyzing the clock-cycles needed to perform a given CNN-classification task. By using the new CMSIS Neural-Net-Library, we implemented the CNN on a Cortex-M7 microcontroller, which offers DSP like performance and optimizations. For comparison purposes, we implemented the net on a standard Cortex-M0+ microcontroller and on the GAP8 processor, which uses 8 RISC-V cores to perform efficient signal processing and neural net tasks. The results showed that between the Cortex-M0+, M7 and GAP8 an efficiency increase of factor 100 can be achieved. This leads to reduction in energy consumption of the same order of magnitude which is crucial for battery powered hearing instruments.
The CNN performs audio scene classification by analyzing 32x32 mel-spectrogram data, using two convolutional and four fully connected layers. During the training phase, a large dataset providing data from 6 different scene-classes was used. Our best CNN with over 1 million parameters was able to classify audio samples with an accuracy of 92%. To allow real-time classification on a lower power embedded system, the net size was reduced to 6000 parameters what decreased the accuracy to 86%. This accuracy is still sufficient for the intended use in hearing instruments. Audio samples of 3 seconds length are classified within 16.5 milliseconds by the Cortex-M7.
For further research, the benefits and implementation of RNN networks like LSTM structures could be analyzed on embedded systems.


## Are you serious? Probabilistic Modelling in Deep Neural Networks
Elvis Murina, Vasily Tolkachev, Beate Sick, Oliver Dürr (ZHAW,HTWG,UZH)

Deep convolutional neural networks show outstanding performance in image-based phenotype classification given that all existing phenotypes are presented during the training of the network. However, in real-world high-content screening (HCS) experiments, it is often impossible to know all phenotypes in advance. Moreover, novel phenotype discovery itself can be an HCS outcome of interest. This aspect of HCS is not yet covered by classical deep learning approaches. When presenting an image with a novel phenotype to a trained network, it fails to indicate a novelty discovery but assigns the image to a wrong phenotype. To tackle this problem and address the need for novelty detection, we use a recently developed Bayesian approach for deep neural networks called Monte Carlo (MC) dropout to define different uncertainty measures for each phenotype prediction. With real HCS data, we show that these uncertainty measures allow us to identify novel or unclear phenotypes. In addition, we also found that the MC dropout method results in a significant improvement of classification accuracy. The proposed procedure used in our HCS case study can be easily transferred to any existing network architecture and will be beneficial in terms of accuracy and novelty detection.

## Stroke detection using convolutional neural networks.
Lisa Herzog, Elvis Murina, Oliver Dürr, Susanne Wegener Beate Sick (UZH,USZ,HTWG,ZHAW)

We apply deep learning approaches to magnetic resonance images of stroke and TIA patients. We show how to take the special three-dimensional structure of the data into account in order to improve the model performance. We further utilize MC dropout methods during test time for probabilistic predictions and corresponding confidence measures. For reliable patient-level predictions, we evaluate how to combine the image-based prediction values by considering the uncertainty measurements.
