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
