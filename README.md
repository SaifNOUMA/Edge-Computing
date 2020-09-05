# Edge-Computing

> ## Introduction
>* Deep neural networks are the state of the art methods for many learning tasks thanks to their potential to derive better features at each network layer. However, the increased efficiency of additional layers in a deep neural network comes at the cost of additional latency and energy consumption in feed forward inference. Thus, it becomes more challenging to deploy them in the edge with limited resources.
Therefore, large-scale DL models are generally deployed in the cloud while end devices merely send input data to the cloud and then wait for the DL inference results. Specifically, it cannot guarantee the delay requirement for real-time services such as real-time object recognition with strict demands.
To address these issues, DL applications tend to resort to edge computing. In fact, the use of optimization techniques, distributed DNNs and collaborative inference between IoT devices and the cloud becomes a promising solution.

> ## Proposed Work:
>### 1. Proposed Architecture
>In order to reduce the computation time of DL inference, we add early exits branches at
different stages of the network. Hence, this latter enable the inference to exit early from these
additional branches based on a confidence criteria. For instance, as depicted in figure below, an
edge device could give primary inference results at an early stage if the confidence criteria is
satisfied. Otherwise, further computation should take place on the cloud or on the edge server.
![GitHub Logo](/images/distributed_cnn.png)
>### 2. Distributed Model Inference
>During inference, the predictions will be made by the earlier exits until the main branch (last exit point) is reached. As a result, the forecast will be returned by the last exit in the worst case. To deal with the communication between different nodes, we employ a distributed workflow to take charge of the message activity and the synchronization as depicted in figure below:
![GitHub Logo](/images/communication_workflow.png)
