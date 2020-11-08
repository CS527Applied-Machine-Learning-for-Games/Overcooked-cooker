using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using Newtonsoft.Json;
using System.Net.Sockets;
using System.Threading;
using System.Text;
using UnityEngine;
using Object = UnityEngine.Object;
//using System.Web.Script.Serialization;
using System.Runtime.Serialization;
//using ZeroMQ;
//using NetMQ;
//using NetMQ.Sockets;

namespace Overcooked_Socket
{
    internal class Main : MonoBehaviour
    {
        #region private members 	
        private TcpClient socketConnection;
        private Thread clientReceiveThread;
        private PlayerControls playerControl2 = getPlayer()[0];
        private PlayerControls playerControl1 = getPlayer()[1];
        private ServerIngredientContainer[] ContainerControls = getContainer();
        private MoveAction actionmove;


        public struct PlayerInfo
        {
            public string name;
            public struct Position
            {
                public double x;
                public double y;
                public double z;
                public double a;
            };
            public Position p;
            public bool isCarry;
            public string carry;
        }
        public struct ContainerInfo
        {
            public string name;
            public struct Position
            {
                public double x;
                public double y;
                public double z;
            };
            public Position p;
            public bool hasIngredient;
            public string ingredient;
        }
        #endregion
        public void Start()
        {
            Logger.Clear();
            Logger.Log("Loaded.");
            //ArrayList contents = StationUtil.GetIngredientContainerContents(gameObject);
            //Logger.Log($"contentSize:{contents.Count}");
            //foreach (String content in contents)
            //{
            //    Logger.Log($"contents:{content}");
            //}
            ConnectToTcpServer();
        }
        public static string GetNewOrder()
        {
            ArrayList orders = OrderUtil.GetOrders(ObjectUtil.GetFlowController());

            //int index = 1;

            string rtn = "";
            rtn += ($"{orders.Count},");

            foreach (string order in orders)
            {
                rtn += ($"{order},");
            }
           // rtn = rtn.Substring(0, rtn.Length - 1);

            return rtn;
        }
        private static PlayerControls[] getPlayer()
        {
            PlayerControls[] playerControls = GameObject.FindObjectsOfType<PlayerControls>();
           
            return playerControls;
        }
        private static ServerIngredientContainer[] getContainer()
        {
            ServerIngredientContainer[] containers = Object.FindObjectsOfType<ServerIngredientContainer>();
            return containers;
        }
        private Vector3 getPlayerPosition(PlayerControls player)
        {
            Vector3 playerPos = PlayerUtil.GetChefPosition(player);
            return playerPos;
        }
        private Vector3 getContainerPosition(ServerIngredientContainer container)
        {
            Vector3 containerPos = ContainerUtil.GetContainerPosition(container);
            return containerPos;
        }

        private double getPlayerAngle(PlayerControls player)
        {
            return Convert.ToDouble(PlayerUtil.GetChefAngles(player));
        }
        private PlayerInfo getPlayerInfo(PlayerControls playerControl, Dictionary<String, String> containerMap)
        {
            PlayerInfo playerInfo = new PlayerInfo();
            playerInfo.name = playerControl.name;
            playerInfo.p = new PlayerInfo.Position();
            Vector3 playerPos = getPlayerPosition(playerControl);
            playerInfo.p.x = Math.Round(playerPos.x, 2);
            playerInfo.p.y = Math.Round(playerPos.y, 2);
            playerInfo.p.z = Math.Round(playerPos.z, 2);
            playerInfo.p.a = Math.Round(getPlayerAngle(playerControl), 2);
            playerInfo.isCarry = PlayerUtil.IsCarrying(playerControl);
            if (playerInfo.isCarry == false)
            {
                playerInfo.carry = "None";
            }
            else
            {
                String playerCarry = PlayerUtil.GetCarrying(playerControl);
                foreach (var dic in containerMap)
                {
                    if(dic.Key == playerCarry)
                    {
                        if(dic.Value == "")
                        {
                            playerCarry = playerCarry + "+" + "None";
                        }
                        else
                        {
                            playerCarry = playerCarry + "+" + dic.Value;
                        }                    
                        break;
                    }
                }
                playerInfo.carry = playerCarry;
            }
            return playerInfo;
        }
        private Dictionary<String, String> getContainerMap(ContainerInfo[] cInfo)
        {
            Dictionary<String, String> containerMap = new Dictionary<String, String>();
            foreach(ContainerInfo c in cInfo)
            {
                containerMap.Add(c.name, c.ingredient);
            }
            return containerMap;
        }
        private ContainerInfo getContainerInfo(ServerIngredientContainer container)
        {
            ContainerInfo cInfo = new ContainerInfo();
            cInfo.name = container.name;
            cInfo.p = new ContainerInfo.Position();
            Vector3 cPos = getContainerPosition(container);
            cInfo.p.x = Math.Round(cPos.x, 2);
            cInfo.p.y = Math.Round(cPos.y, 2);
            cInfo.p.z = Math.Round(cPos.z, 2);
            cInfo.hasIngredient = ContainerUtil.HasIngredient(container);
            if (cInfo.hasIngredient == false)
            {
                cInfo.ingredient = "None";
            }
            else
            {
                cInfo.ingredient = ContainerUtil.GetIngredient(container);
            }
            return cInfo;
        }

        private string getPlayerInfoString(PlayerInfo playerInfo)
        {
            string msg = "";
            msg += $"{playerInfo.name},";
            msg += $"{playerInfo.p.x},{playerInfo.p.y},{playerInfo.p.z},{playerInfo.p.a},";
            msg += $"{playerInfo.isCarry},";
            msg += $"{playerInfo.carry},";
            return msg;
        }
        private string getContainerInfostring(ContainerInfo container)
        {
            string msg = "";
            msg += $"{container.name},";
            msg += $"{container.p.x},{container.p.y},{container.p.z},";
            return msg;
        }
        private void Reply()
        {
            ContainerInfo[] containerInfos = new ContainerInfo[ContainerControls.Length];
            for(int i = 0; i < ContainerControls.Length; i++)
            {
                containerInfos[i] = getContainerInfo(ContainerControls[i]);
            }
            Dictionary<String, String> containerMap = getContainerMap(containerInfos);
            PlayerInfo player2 = getPlayerInfo(playerControl1, containerMap);
            PlayerInfo player1 = getPlayerInfo(playerControl2, containerMap);
            string result = getPlayerInfoString(player2);
            result += getPlayerInfoString(player1);
            result += GetNewOrder();
            result += $"{containerInfos.Length},";
            foreach (var containerInfo in containerInfos)
            {
                result += getContainerInfostring(containerInfo);
            }
            //Logger.Log(msg);
            //Logger.LogWorkstations();
            Logger.LogContainer();
            Send(result);
            // Logger.Log(result);
        }
        public void Update()
        {

            actionmove.Update();

            if (Input.GetKeyDown(KeyCode.M))
            {
                Loader.Unload();
            }
        }
        private void ConnectToTcpServer()
        {
            
            try
            {
                clientReceiveThread = new Thread(new ThreadStart(ListenForData));
                clientReceiveThread.IsBackground = true;
                clientReceiveThread.Start();
                Logger.Log("server start!");
          
            }
            catch (Exception e)
            {
                Logger.Log("On client connect exception " + e);
            }

        }
        /// <summary> 	
        /// Runs in background clientReceiveThread; Listens for incomming data. 	
        /// </summary>     
        private void ListenForData()
        {
       
            try
            {
                Logger.Log("listening on port 7777");
                socketConnection = new TcpClient("localhost", 7777);
                NetworkStream stream = socketConnection.GetStream();

                Byte[] bytes = new Byte[1024];
                
                while (true)
                {
                    // Get a stream object for reading 				
                    


                        int length;
                        // Read incomming stream into byte arrary. 	
                        
                        
                        while ((length = stream.Read(bytes, 0, bytes.Length)) != 0)
                        {
                            var incommingData = new byte[length];
                            Array.Copy(bytes, 0, incommingData, 0, length);
                            // Convert byte array to string message. 						
                            string serverMessage = Encoding.ASCII.GetString(incommingData);
                            Logger.Log("server message received as: " + serverMessage);
                        // process data

                            if (serverMessage.StartsWith("action"))
                            {
                                string[] info = serverMessage.Split(' ');

                                if (info[1].Equals("move"))
                                {
                                    int playerId = Int32.Parse(info[2]);
                                    float targetX = float.Parse(info[5]);
                                    float targetZ = float.Parse(info[6]);
                                    Logger.Log("id: " + playerId + " x: " + targetX + " z: " + targetZ);



                                    PlayerControls player = playerControl1;
                                    if (playerId == 1)
                                    {
                                        player = playerControl2;
                                    }
                                    actionmove = new MoveAction(player, new Vector3(targetX, 0, targetZ));
                                    //action.Update();
                                    Logger.Log("player moved!");
                                } else if (info[1].Equals("pickdrop"))
                                {
                                    int playerId = Int32.Parse(info[2]);
                                    PlayerControls player = playerControl1;
                                    if (playerId == 1)
                                    {
                                        player = playerControl2;
                                    }
                                    PickDropAction actionpickdrop = new PickDropAction(player, false);
                                    actionpickdrop.Update();
                                    Logger.Log("player pick or drop!");
                                }
                        }

                        Reply();
                        }
                        
                    
                 
                }
               
            }
            catch (Exception e)
            {

                Logger.Log("exception: " + e);
            }
           
        }
        /// <summary> 	
        /// Send message to server using socket connection. 	
        /// </summary> 	
        private void Send(string clientMessage)
        {
            if (socketConnection == null)
            {
                return;
            }
            try
            {
                // Get a stream object for writing. 			
                NetworkStream stream = socketConnection.GetStream();
                if (stream.CanWrite)
                {
                    // Convert string message to byte array.                 
                    byte[] clientMessageAsByteArray = Encoding.ASCII.GetBytes(clientMessage);
                    // Write byte array to socketConnection stream.                 
                    stream.Write(clientMessageAsByteArray, 0, clientMessageAsByteArray.Length);
                    //Logger.Log("Client sent his message - should be received by server");
                }
            }
            catch (SocketException socketException)
            {
                Logger.Log("Socket exception: " + socketException);
            }
        }
    }
}
