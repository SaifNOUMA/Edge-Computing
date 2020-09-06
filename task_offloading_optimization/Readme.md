This Folder presents an IoT application that use ns3-gym :

Definition :
    - The goal of this project is to implement an offloading Task and Resource Management in the IoT Environment.


    - Simplification of the project:
        For simplification:
             I have used as metrics (Execution Time) and the application that will run on each IoT Devices is simply the send of packets to a local node.

            Normally, the metrics to evaluate the decisions are both the execution time and the energy consumption.

            

Description of the files:
    - sim.cc: 
        It contains the simulation of the IoT scenario : 
        * IoT Devices where each one have a bunch of Tasks and would like to minimize the Latency and  The energy consumption.

        * Edge Server who have the capability to execute the tasks more faster.
        
        * The communication between the IoT devices and the Edge Server is based on Wifi.
        
    - agent.py:
        Contains the RL agent that will decide where each task will be executed (Edge Server or Locally)

        The agent will run the simulation several iterations in order to find out the optimal decisions.

    
Run the Simulation:
    - python agent.py
    - ./waf --run "offloading-v0 --N=4 --M=2"