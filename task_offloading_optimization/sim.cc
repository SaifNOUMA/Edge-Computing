/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */


#include <fstream>
#include<string>
#include <stdlib.h>   
#include "ns3/ipv4.h"
#include "ns3/core-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include "ns3/mobility-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/netanim-module.h"
#include "ns3/config-store-module.h"
#include "ns3/wifi-radio-energy-model-helper.h"
#include "ns3/opengym-module.h"
#include "ns3/gnuplot.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("offloading-v0");



// ********************* Global Variables ****************************
const uint32_t MA = 5000;                       // Maximum Number of Tasks        
bool ipRecvTos = true;                          // Type of Service
bool ipRecvTtl = true;                          // Time to Live

uint32_t N = 1;                                 // Number of Users
uint32_t M = 3 ;                                // Number of Tasks

uint32_t Decisions[MA] ;                   // Decisions For Task Offloading
float    ExecTime[MA] ;                   // Execution Time For Tasks
uint32_t ok = 0 ;                               // Notification For The Completion for All Tasks
uint32_t FirstShot = 0 ;                        // Notification for The First Iteration
float    SumPast ; 

Ptr<Node> wifiApNode = CreateObject<Node> ();   // Access Point Node
Ptr<Node> edgeServer = CreateObject<Node> ();   // Edge Server Node
PointToPointHelper pointToPoint; 
NetDeviceContainer p2pDevices;                  // Card Devices For ES & AP
Ipv4InterfaceContainer p2pInterfaces;           // Interfaces For Communications ES & AP

NodeContainer wifiStaNodes;                     // Users' Nodes For Outside Communication
Ipv4InterfaceContainer staWifiInterfaces;       // Interfaces For Base Stations
Ipv4InterfaceContainer apWifiInterfaces;        // Interfaces For The Access Point


NodeContainer LocNodes;                        // Nodes For Local Computation
NetDeviceContainer LocDevices[MA];             // NetDevices For Local Card Devices
Ipv4InterfaceContainer LocInterfaces[MA];      // Interfaces For Local Communications

Ptr<OpenGymInterface> openGymInterface;
int done = 0 ;                                 // Variable to indicate the case of Game Over    
Gnuplot2dDataset dataset;                      // Variable For The Plot Generation
uint32_t iteration = 0 ;                       // Cursor For The Number of Iterations
uint32_t nb_iterations = 300 ;                 // Number of Iterations For a Single Episode


std::string fileNameWithNoExtension = "cost_function";
std::string graphicsFileName        = fileNameWithNoExtension + ".png";
std::string plotFileName            = fileNameWithNoExtension + ".plt";
std::string plotTitle               = "Execution Time";
std::string dataTitle               = "Execution Time";
Gnuplot plot (graphicsFileName);               // Instantiate the plot and set its title.


// Abstraction Of The Functions
void ReceivePacket (Ptr<Socket> socket);
static void SendPacket (Ptr<Socket> socket, uint32_t pktSize, uint32_t pktCount, Time pktInterval);
void SendResult  (Ptr<Node> source, Ptr<Node> destination, Ipv4Address ipDest, uint32_t data_size, Time delay);
void SendTask  (Ptr<Node> source, Ptr<Node> destination, Ipv4Address ipDest, uint32_t data_size, Time delay);
void Execute (Time time); 



// ***************************************** Implementation of Functions ************************************************

void ReceivePacket (Ptr<Socket> socket)
{
  Ptr<Node> destination = socket->GetNode();                          // Pointer To Destination Node
  Ipv4Address ipDst = destination->GetObject<Ipv4> ()->GetAddress (1,0).GetLocal(); // Ipv4 Address Of The Destination Node


  Address addr ;
  Ptr<Packet> packet = socket->RecvFrom (2000, 30, addr);
  Ipv4Address ipSrc = InetSocketAddress::ConvertFrom(addr).GetIpv4(); // Ipv4 Address Of The Source Node



  if (p2pInterfaces.GetAddress (1) == ipDst){
    // In Case That The Destination Is The Edge Server

    Ptr<Node> Dest ;
    Ipv4Address ipDest;
    // Determiner The Id Of The IoT Device Who Sends The sockets
    for (int i = 0 ; i < (int) N ; i++){
      if (ipSrc == staWifiInterfaces.GetAddress(i)){
          Dest = wifiStaNodes.Get (i);
          ipDest = staWifiInterfaces.GetAddress (i);
          break;
      }
    }
    SendResult(edgeServer, Dest, ipDest, 1000, Seconds(0.5));     // Send The Result With a Delay of 0.5 Seconds

  }


  if (destination->GetId() >= N + 2 && destination->GetId() < 2 * N + 2){
    // In Case That The Destination Is a Local Node
    uint32_t id = destination->GetId() - N - 2;       // Id Of The IoT Device That Sends a request.
    Ptr<Node> Dest = wifiStaNodes.Get (id);           // Node of the IoT Device.
    Ptr<Node> Src  = LocNodes.Get (id);               // Local Node From which we will send the result.
    Ipv4Address ipDst = staWifiInterfaces.GetAddress (id); 
  
    SendResult(Src , Dest , ipDst , 100, Seconds(0)) ; // Send The Result To The IoT Device with a Delay of 0 Seconds.

  }


  
  if (destination->GetId() >= 2 && destination->GetId() < N + 2){
    // Save The Execution Time For The Task Completion in The ExecTime Array.
    
    uint32_t index = destination->GetId() - 2 ;
    uint32_t Ref = (uint32_t) Simulator::Now().GetSeconds();
    
    Ref = (Ref - 1) % M ; 
    ExecTime[M*index+Ref] = Simulator::Now().GetSeconds(); 
    ok ++ ;                                           // Increment The Number Of The Completed Tasks
  }


  if (ok == N * M){
    //  If All The Tasks Are Completed.
    std::cout <<  "Episode : " << iteration << std::endl ;


    if (FirstShot == 0){
      FirstShot ++ ; 

      float sum = 0.0 ; 
      for(uint32_t i = 0 ; i < N * M ; i ++)
        sum += ExecTime[i] - (float) ((uint32_t) ExecTime[i]);
      
      SumPast = sum ; 

    }
    else {
      openGymInterface->NotifyCurrentState();
    }

    if (iteration == nb_iterations) {
      done = 1 ;
    
      std::cout << "End of Simulation" << std::endl; 
      plot.AddDataset (dataset);
      // Open the plot file.
      std::ofstream plotFile (plotFileName.c_str());
      // Write the plot file.
      plot.GenerateOutput (plotFile);
      // Close the plot file.
      plotFile.close ();
    }

    ok = 0;             // Reset The Number of Completed Tasks For The Future Tasks

    float temp = (float) ((uint32_t) Simulator::Now().GetSeconds() + 1 ) ; 
    float temp1= Simulator::Now().GetSeconds() ; 
    Time time = Seconds( temp - temp1) ;

    Execute(time);

  }
}


static void SendPacket (Ptr<Socket> socket, uint32_t pktSize,
                        uint32_t pktCount, Time pktInterval )
{
  if (pktCount > 0)
    {

      Ptr<Packet> packet= Create<Packet> (pktSize) ;
      socket->Send(packet);
      Simulator::Schedule (pktInterval, &SendPacket,
                           socket, pktSize,pktCount - 1, pktInterval);
    }
  else
    {
      socket->Close ();
    }
}



void SendResult  (Ptr<Node> source, Ptr<Node> destination, Ipv4Address ipDest, uint32_t data_size, Time delay){
    // SendResult Will Send A Signal From The Server or LocalNode To The IoT Device For Task's Result
  
  TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
  Ptr<Socket> socket = Socket::CreateSocket (source , tid);

  //Receiver socket on Edge Server
  Ptr<Socket> recvSink = Socket::CreateSocket (destination, tid);
  InetSocketAddress local = InetSocketAddress (Ipv4Address::GetAny (), 80);
  recvSink->SetIpRecvTos (ipRecvTos);
  recvSink->SetIpRecvTtl (ipRecvTtl);
  recvSink->Bind (local);
  recvSink->SetRecvCallback (MakeCallback (&ReceivePacket));

  InetSocketAddress remote = InetSocketAddress (ipDest, 80);
  socket->Connect(remote);

  Time interPacketInterval = Seconds (1.0);
  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (),
                              delay, &SendPacket,
                  socket, data_size , 1, interPacketInterval);
}



void SendTask (Ptr<Node> source, Ptr<Node> destination, Ipv4Address ipDest, uint32_t data_size, Time delay){
  // SendTask Will Send A Signal From The IoT Device To The Server or LocalNode For Task Execution

  TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
  Ptr<Socket> socket = Socket::CreateSocket (source , tid);

  //Receiver socket on Edge Server
  Ptr<Socket> recvSink = Socket::CreateSocket (destination, tid);
  InetSocketAddress local = InetSocketAddress (Ipv4Address::GetAny (), 80);
  recvSink->SetIpRecvTos (ipRecvTos);
  recvSink->SetIpRecvTtl (ipRecvTtl);
  recvSink->Bind (local);
  recvSink->SetRecvCallback (MakeCallback (&ReceivePacket));

  InetSocketAddress remote = InetSocketAddress (ipDest, 80);
  socket->Connect(remote);

  Time interPacketInterval = Seconds (1.0);
  Simulator::ScheduleWithContext (socket->GetNode ()->GetId (),
                              delay, &SendPacket,
                  socket, data_size , 1, interPacketInterval);
}



void Execute(Time time){
  // Main Function That Launch The Beginning Of The Tasks

  for (int i = 0 ; i < (int) (N * M) ; i++){

    int num = i / M;
    int task = i % M;
    

    if (Decisions[i]){
        Ptr<Node> source = wifiStaNodes.Get(num);
        Ptr<Node> dest   = edgeServer;
        Ipv4Address ipv4Dest = p2pInterfaces.GetAddress (1);
        SendTask(source, dest, ipv4Dest, 2000, Seconds(task) + time);

    }
    else{
        Ptr<Node> source = wifiStaNodes.Get(num);
        Ptr<Node> dest   = LocNodes.Get (num);
        Ipv4Address ipv4Dest = LocInterfaces[num].GetAddress (1);

        SendTask(source, dest, ipv4Dest, 1000, Seconds(task) + time);
    }
  }
}



// ****************************** Gym Interface ****************************** // 

Ptr<OpenGymSpace> MyGetObservationSpace(void)
{
  uint32_t Ntasks = N * M;
  int low = 0;
  int high = 1;
  std::vector<uint32_t> shape = {Ntasks,};
  std::string dtype = TypeNameGet<uint32_t> ();
  Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (low, high, shape, dtype);

  return space;
}



Ptr<OpenGymSpace> MyGetActionSpace(void)
{
  Ptr<OpenGymDiscreteSpace> space = CreateObject<OpenGymDiscreteSpace> ( N * M + 1 );
  return space;
}


bool MyGetGameOver(void)
{
  bool isGameOver = ( done==1 ) ;

  return isGameOver;
}


Ptr<OpenGymDataContainer> MyGetObservation(void)
{
  std::vector<uint32_t> shape = {N * M,};
  Ptr<OpenGymBoxContainer<uint32_t> > box = CreateObject<OpenGymBoxContainer<uint32_t> >(shape);
  for (uint32_t i = 0 ; i < N * M ; i ++ ){
    box->AddValue(Decisions[i]);
  }

  return box;
}



bool MyExecuteActions (Ptr<OpenGymDataContainer> action){
  
  Ptr<OpenGymDiscreteContainer> value = DynamicCast<OpenGymDiscreteContainer>(action);
  uint32_t index = value->GetValue();

  if (index != N * M) 
    Decisions[index] = 1 - Decisions[index];

  return true;
}


float MyGetReward(void)
{
    float reward ; 
    float sum = 0.0 ; 
    for(uint32_t i = 0 ; i < N * M ; i ++)
      sum += ExecTime[i] - (float) ((uint32_t) ExecTime[i]);


    if ( sum < SumPast ) {
       // && abs(sum - SumPast) > 0.1){
      reward = + 1.0;
    }
    else
    {
        if (sum > SumPast ) { 
          //&& abs(sum - SumPast) > 0.1){
          reward = - 1.0;
        }

        else{
          if (sum > 0.8){
            reward = - 1.0 ;
          }
          else
          {
            reward = + 2.0 ;
          }     
        }
    }

    SumPast = sum ;
    iteration ++ ; 
    dataset.Add(iteration, sum) ;       // Add A New Point To The Plot

    return reward ; 

}






// ********************************* Main Function ***********************************
int
main (int argc, char *argv[])
{

  // Parameters of the environment
  uint32_t simSeed = 1;
  uint32_t openGymPort = 5555;
  bool     tracing = false;

  Packet::EnablePrinting ();
  Packet::EnableChecking ();

  CommandLine cmd;
  cmd.AddValue ("tracing", "Enable PCAP tracing", tracing);
  cmd.AddValue ("N", "Number of Users 'wiFi STA devices'", N);
  cmd.AddValue ("M", "Number of Tasks per User", M);
  cmd.Parse (argc,argv);



  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (simSeed);


  /********************** Point to Point Communication Between AP and ES *********************/
  pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("500Mbps"));
  pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));
  p2pDevices = pointToPoint.Install (NodeContainer (wifiApNode, edgeServer));


  /********************** WiFi Communication Between AP and Users's Nodes *********************/
  wifiStaNodes.Create (N);
  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  YansWifiPhyHelper phy = YansWifiPhyHelper::Default ();
  phy.SetChannel (channel.Create ());
  WifiHelper wifi;
  wifi.SetRemoteStationManager ("ns3::AarfWifiManager");
  WifiMacHelper mac;
  Ssid ssid = Ssid ("ns-3-ssid");
  mac.SetType ("ns3::StaWifiMac",
		  	  "Ssid", SsidValue (ssid),
		  	  "ActiveProbing", BooleanValue (false));
  NetDeviceContainer staDevices;
  staDevices = wifi.Install (phy, mac, wifiStaNodes);
  mac.SetType ("ns3::ApWifiMac",
		  	  "Ssid", SsidValue (ssid));
  NetDeviceContainer apDevices;
  apDevices = wifi.Install (phy, mac, wifiApNode);



  // Connections Between "Base Stations Nodes" & Local Nodes
  LocNodes.Create (N);
  for (uint32_t i = 0 ; i < N ; i ++){
	  LocDevices[i] = pointToPoint.Install( NodeContainer(wifiStaNodes. Get(i), LocNodes. Get(i)) );
  }



  // *************************** Mobility Model *********************************
  MobilityHelper mobility;
  mobility.SetPositionAllocator ("ns3::GridPositionAllocator",
		  	  	  	  	  	  	  "MinX", DoubleValue (0.0),
								  "MinY", DoubleValue (0.0),
								  "DeltaX", DoubleValue (5.0),
								  "DeltaY", DoubleValue (10.0),
								  "GridWidth", UintegerValue (3),
								  "LayoutType", StringValue ("RowFirst"));
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiStaNodes);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiApNode);




  /*************************** Internet Configuration *****************************/
  InternetStackHelper stack;
  stack.Install (edgeServer);
  stack.Install (wifiApNode);
  stack.Install (wifiStaNodes);
  stack.Install(LocNodes);

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");
  p2pInterfaces = address.Assign (p2pDevices);
  address.SetBase ("10.1.2.0", "255.255.255.0");
  staWifiInterfaces = address.Assign (staDevices);
  apWifiInterfaces = address.Assign (apDevices);
  for (uint32_t i = 0 ; i < N ; i ++) {
	  std::string ipaddress = "10." + std::to_string(i + 1) + ".3.0" ;
	  address.SetBase (ns3::Ipv4Address (ipaddress.c_str()), "255.255.255.0");
	  LocInterfaces[i] = address.Assign (LocDevices[i]);
  }

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();



  // ******************* Initialization For The Execution Time Figure *******************
  plot.SetTitle (plotTitle);
  plot.SetTerminal ("png");
  // Set the labels for each axis.
  plot.SetLegend ("Iteration", "Execution Time");
  // Set the range for the x axis.
  std::string range = "set xrange [0:+" + std::to_string(nb_iterations) + "]" ;
  plot.AppendExtra (range);

  // Instantiate the dataset, set its title, and make the points be
  // plotted along with connecting lines.
  dataset.SetTitle (dataTitle);
  dataset.SetStyle (Gnuplot2dDataset::LINES_POINTS);




  // *************************** Call The Application **************************
  std::fill_n(ExecTime , N * M , 0) ;
  std::fill_n(Decisions, N * M , 1) ;

  Execute(Seconds(1.0));   




  // OpenGym Env
  openGymInterface = CreateObject<OpenGymInterface> (openGymPort);
  openGymInterface->SetGetActionSpaceCb( MakeCallback (&MyGetActionSpace) );
  openGymInterface->SetGetObservationSpaceCb( MakeCallback (&MyGetObservationSpace) );
  openGymInterface->SetGetGameOverCb( MakeCallback (&MyGetGameOver) );
  openGymInterface->SetGetObservationCb( MakeCallback (&MyGetObservation) );
  openGymInterface->SetGetRewardCb( MakeCallback (&MyGetReward) );
  openGymInterface->SetExecuteActionsCb( MakeCallback (&MyExecuteActions) );




  // NETAMIN: Visualization Tool for Network Simulation
  AnimationInterface anim("offloading-v0.xml");
  anim.EnablePacketMetadata (true);
  anim.EnableIpv4L3ProtocolCounters (Seconds (0), Seconds (10));
  // Edge Server
  anim.SetConstantPosition(edgeServer, ((N+1) / 2) * 10.0, 40.0);
  anim.UpdateNodeColor(edgeServer, 0, 0, 255);
  anim.UpdateNodeDescription(edgeServer, "Edge Server");
  // WiFi Stations: IoT Devices
  for (int i = 0 ; i < (int) N ; i ++){
	  anim.SetConstantPosition(wifiStaNodes.Get (i), (i + 1)*10.0, 20.0);
	  anim.UpdateNodeColor(wifiStaNodes.Get (i), 0, 255, 0);
	  anim.UpdateNodeDescription(wifiStaNodes.Get (i), "N°" + std::to_string(i+1));
	  // Local Execution for IoT Devices
	  anim.SetConstantPosition(LocNodes.Get (i), (i + 1)*10.0, 10.0);
	  anim.UpdateNodeColor(LocNodes.Get (i), 0, 255, 255);
	  anim.UpdateNodeDescription(LocNodes.Get (i), "Loc N°" + std::to_string(i+1));
  }
  // WiFi Access Point
  anim.SetConstantPosition(wifiApNode, ((N+1)/2)*10.0, 30.0);
  anim.UpdateNodeColor(wifiApNode, 255, 0, 0);
  anim.UpdateNodeDescription(wifiApNode, "Access Point");



  //  Launch The Simunlation
  Simulator::Stop (Seconds (1000000.0));
  Simulator::Run ();

  openGymInterface->NotifySimulationEnd();

  Simulator::Destroy ();


  return 0;
}
