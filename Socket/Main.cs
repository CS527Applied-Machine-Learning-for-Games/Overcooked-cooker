﻿using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using Object = UnityEngine.Object;
//using System.Web.Script.Serialization;
using System.Runtime.Serialization;
//using ZeroMQ;
//using NetMQ;
//using NetMQ.Sockets;

namespace Overcooked_Socket {
    internal class Main : MonoBehaviour {
        #region private members 	
        private TcpClient socketConnection;
        private Thread clientReceiveThread;
        private PlayerControls playerControl2 = getPlayer () [0];
        private PlayerControls playerControl1 = getPlayer () [1];
        private PickupItemSpawner[] pickupItemControls = getPickupItem ();
        private ClientWorkstation[] workStations = getClientWorkstation ();
        private MoveAction actionmove;
        private PickDropAction actionpickdrop;
        private ChopAction actionchop;
        private TurnAction actionturn;
        public struct PlayerInfo {
            public string name;
            public struct Position {
                public double x;
                public double y;
                public double z;
                public double a;
            };
            public Position p;
            public bool isCarry;
            public string carry;
        }
        public struct ContainerInfo {
            public string name;
            public string id;
            public struct Position {
                public double x;
                public double z;
            };
            public Position p;
            public bool hasIngredient;
            public string ingredient;
        }
        public struct FoodInfo {
            public string name;
            public struct Position {
                public double x;
                public double z;
            };
            public Position p;
        }
        #endregion
        public void Start () {
            //Logger.Clear();
            //Logger.Log("Loaded.");
            ConnectToTcpServer ();
        }
        public static string GetNewOrder () {
            ArrayList orders = OrderUtil.GetOrders (ObjectUtil.GetFlowController ());

            //int index = 1;

            string rtn = "";
            rtn += ($"{orders.Count},");

            foreach (string order in orders) {
                rtn += ($"{order},");
            }
            // rtn = rtn.Substring(0, rtn.Length - 1);

            return rtn;
        }
        private static PlayerControls[] getPlayer () {
            PlayerControls[] playerControls = GameObject.FindObjectsOfType<PlayerControls> ();

            return playerControls;
        }
        private static ServerIngredientContainer[] getContainer () {
            ServerIngredientContainer[] containers = Object.FindObjectsOfType<ServerIngredientContainer> ();
            return containers;
        }
        private static PickupItemSpawner[] getPickupItem () {
            PickupItemSpawner[] pickupItems = Object.FindObjectsOfType<PickupItemSpawner> ();
            return pickupItems;
        }
        private static ClientCatchableItem[] getFoodItem () {
            ClientCatchableItem[] foodItems = Object.FindObjectsOfType<ClientCatchableItem> ();
            return foodItems;
        }
        private static ClientFlammable[] getClientFlammable () {
            ClientFlammable[] fire = Object.FindObjectsOfType<ClientFlammable> ();
            return fire;
        }
        private static ClientWorkstation[] getClientWorkstation () {
            ClientWorkstation[] workstations = Object.FindObjectsOfType<ClientWorkstation> ();
            return workstations;
        }
        private Vector3 getPlayerPosition (PlayerControls player) {
            Vector3 playerPos = PlayerUtil.GetChefPosition (player);
            return playerPos;
        }
        private Vector3 getContainerPosition (ServerIngredientContainer container) {
            Vector3 containerPos = ContainerUtil.GetContainerPosition (container);
            return containerPos;
        }
        private Vector3 getFoodPosition (ClientCatchableItem food) {
            Vector3 foodPos = food.transform.position;
            return foodPos;
        }

        private double getPlayerAngle (PlayerControls player) {
            return Convert.ToDouble (PlayerUtil.GetChefAngles (player));
        }
        private PlayerInfo getPlayerInfo (PlayerControls playerControl, Dictionary<String, String> containerMap) {
            PlayerInfo playerInfo = new PlayerInfo ();
            playerInfo.name = playerControl.name;
            playerInfo.p = new PlayerInfo.Position ();
            Vector3 playerPos = getPlayerPosition (playerControl);
            playerInfo.p.x = Math.Round (playerPos.x, 2);
            playerInfo.p.y = Math.Round (playerPos.y, 2);
            playerInfo.p.z = Math.Round (playerPos.z, 2);
            playerInfo.p.a = Math.Round (getPlayerAngle (playerControl), 2);
            playerInfo.isCarry = PlayerUtil.IsCarrying (playerControl);
            if (playerInfo.isCarry == false) {
                playerInfo.carry = "None";
            } else {
                String result = "";
                GameObject playerCarry = PlayerUtil.GetCarrying (playerControl);
                //Logger.Log($"player holding: {playerCarry.name}, {playerCarry.GetInstanceID().ToString()}");
                bool flag = false;
                foreach (var dic in containerMap) {
                    if (dic.Key.Equals (playerCarry.GetInstanceID ().ToString ())) {
                        flag = true;
                        if (dic.Value == "") {
                            result = playerCarry.name + "+" + "None";
                        } else {
                            result = playerCarry.name + "+" + dic.Value;
                        }
                        break;
                    }
                }
                if (flag == false) {
                    result = playerCarry.name;
                }
                playerInfo.carry = result;
            }
            return playerInfo;
        }
        private Dictionary<String, String> getContainerMap (ContainerInfo[] cInfo) {
            Dictionary<String, String> containerMap = new Dictionary<String, String> ();
            foreach (ContainerInfo c in cInfo) {
                containerMap.Add (c.id, c.ingredient);
            }
            return containerMap;
        }
        private Dictionary<String, float> getPotProgressMap () {
            Dictionary<String, float> potProgressMap = new Dictionary<String, float> ();
            ClientCookingHandler[] cookingHandler = Object.FindObjectsOfType<ClientCookingHandler> ();
            foreach (var item in cookingHandler) {
                potProgressMap.Add (item.name, item.GetCookingProgress ());
            }
            return potProgressMap;
        }
        private bool hasFire (ClientFlammable[] flammables) {
            bool flag = false;
            foreach (var flammable in flammables) {
                if (flammable.OnFire ()) {
                    flag = true;
                    break;
                }
            }
            return flag;
        }
        private String getScore () {
            ScoreUIController s = Object.FindObjectOfType<ScoreUIController> ();
            //Logger.Log($"name: {s.name}");
            var score = ReflectionUtil.GetValue (s, "m_totalScore");
            return score.ToString ();
        }
        private ContainerInfo getContainerInfo (ServerIngredientContainer container) {
            ContainerInfo cInfo = new ContainerInfo ();
            cInfo.name = container.name;
            cInfo.id = container.gameObject.GetInstanceID ().ToString ();
            cInfo.p = new ContainerInfo.Position ();
            Vector3 cPos = getContainerPosition (container);
            cInfo.p.x = Math.Round (cPos.x, 2);
            cInfo.p.z = Math.Round (cPos.z, 2);
            cInfo.hasIngredient = ContainerUtil.HasIngredient (container);
            if (cInfo.hasIngredient == false) {
                cInfo.ingredient = "None";
            } else {
                cInfo.ingredient = ContainerUtil.GetIngredient (container);
            }
            return cInfo;
        }
        private FoodInfo getFoodInfo (ClientCatchableItem food) {
            FoodInfo fInfo = new FoodInfo ();
            fInfo.name = food.name;
            fInfo.p = new FoodInfo.Position ();
            Vector3 fPos = getFoodPosition (food);
            fInfo.p.x = Math.Round (fPos.x, 2);
            fInfo.p.z = Math.Round (fPos.z, 2);
            return fInfo;
        }
        private string getPlayerInfoString (PlayerInfo playerInfo) {
            string msg = "";
            msg += $"{playerInfo.name},";
            msg += $"{playerInfo.p.x},{playerInfo.p.y},{playerInfo.p.z},{playerInfo.p.a},";
            msg += $"{playerInfo.isCarry},";
            msg += $"{playerInfo.carry},";
            return msg;
        }
        private string getContainerInfostring (ContainerInfo container) {
            string msg = "";
            msg += $"{container.name},";
            msg += $"{container.ingredient},";
            msg += $"{container.p.x},{container.p.z},";
            return msg;
        }
        private string getPickupItemInfostring (PickupItemSpawner pickupItem) {
            string msg = "";
            msg += $"{pickupItem.m_itemPrefab.name},";
            msg += $"{Math.Round(pickupItem.transform.position.x, 2)},{Math.Round(pickupItem.transform.position.y, 2)},{Math.Round(pickupItem.transform.position.z, 2)},";
            return msg;
        }
        private string getFoodInfostring (FoodInfo food) {
            string msg = "";
            msg += $"{food.name},";
            msg += $"{food.p.x},{food.p.z},";
            return msg;
        }
        private string getChopProgress (ClientWorkstation[] workstations) {
            float[] result = new float[workstations.Length];
            for (int i = 0; i < workstations.Length; i++) {
                var item = workstations[i];
                if (item.IsBeingUsed ()) {
                    ClientWorkableItem workableItem = (ClientWorkableItem) ReflectionUtil.GetValue (item, "m_item");
                    result[i] = workableItem.GetProgress ();
                } else {
                    result[i] = 0.0F;
                }
            }
            string s = "";
            foreach (float progress in result) {
                s += $"{progress},";
            }
            return s;
        }

        private void Reply () {
            ServerIngredientContainer[] ContainerControls = getContainer ();
            ClientFlammable[] flammableControls = getClientFlammable ();
            ClientCatchableItem[] foodControls = getFoodItem ();
            FoodInfo[] foodInfos = new FoodInfo[foodControls.Length];
            if (foodControls.Length != 0) {
                for (int i = 0; i < foodControls.Length; i++) {
                    foodInfos[i] = getFoodInfo (foodControls[i]);
                    //Logger.Log($"{containerInfos[i].id}");
                }
            }
            ContainerInfo[] containerInfos = new ContainerInfo[ContainerControls.Length];
            for (int i = 0; i < ContainerControls.Length; i++) {
                containerInfos[i] = getContainerInfo (ContainerControls[i]);
                //Logger.Log($"{containerInfos[i].id}");
            }
            Dictionary<String, String> containerMap = getContainerMap (containerInfos);
            Dictionary<String, float> potProgressMap = getPotProgressMap ();
            PlayerInfo player2 = getPlayerInfo (playerControl1, containerMap);
            PlayerInfo player1 = getPlayerInfo (playerControl2, containerMap);
            //Logger.Log("current z: " + player1.p.z);
            string result = getPlayerInfoString (player2);
            result += getPlayerInfoString (player1);
            result += GetNewOrder ();
            result += $"{containerInfos.Length},";
            foreach (var containerInfo in containerInfos) {
                result += getContainerInfostring (containerInfo);
            }
            result += $"{foodInfos.Length},";
            foreach (var food in foodInfos) {
                result += getFoodInfostring (food);
            }
            //result += $"{pickupItemControls.Length},";
            //foreach (var pickup in pickupItemControls)
            //{
            //    result += getPickupItemInfostring(pickup);
            //}
            result += $"{potProgressMap.Count},";
            foreach (var containerInfo in containerInfos) {
                if (potProgressMap.ContainsKey (containerInfo.name)) {
                    result += $"{potProgressMap[containerInfo.name]},";
                }
            }
            result += $"{hasFire(flammableControls)},";
            result += $"{getScore()},";
            result += $"{workStations.Length},";
            result += $"{getChopProgress(workStations)}";
            Send (result);
            // Logger.Log(result);
            //Logger.LogContainer();
        }
        public void Update () {

            // Logger.Log("Pickdrop Updaing");

            // actionpickdrop.Update();
            // Logger.Log("Chop Updaing");
            // actionchop.Update();
            // Logger.Log("Move Updaing");
            //actionmove.Update();

            if (Input.GetKeyDown (KeyCode.M)) {
                Loader.Unload ();
            }
        }
        private void ConnectToTcpServer () {

            try {
                clientReceiveThread = new Thread (new ThreadStart (ListenForData));
                clientReceiveThread.IsBackground = true;
                clientReceiveThread.Start ();
                //Logger.Log("server start!");

            } catch (Exception e) {
                print (e);
                // Logger.Log("On client connect exception " + e);
            }

        }
        /// <summary> 	
        /// Runs in background clientReceiveThread; Listens for incomming data. 	
        /// </summary>     
        private void ListenForData () {

            try {
                //Logger.Log("listening on port 7777");
                socketConnection = new TcpClient ("localhost", 7777);
                NetworkStream stream = socketConnection.GetStream ();

                Byte[] bytes = new Byte[1024];

                while (true) {
                    // Get a stream object for reading 				

                    int length;
                    // Read incomming stream into byte arrary. 	

                    while ((length = stream.Read (bytes, 0, bytes.Length)) != 0) {
                        var incommingData = new byte[length];
                        Array.Copy (bytes, 0, incommingData, 0, length);
                        // Convert byte array to string message. 						
                        string serverMessage = Encoding.ASCII.GetString (incommingData);
                        //Logger.Log("server message received as: " + serverMessage);
                        // process data

                        if (serverMessage.StartsWith ("action")) {
                            string[] info = serverMessage.Split (' ');

                            if (info[1].Equals ("move")) {
                                int playerId = Int32.Parse (info[2]);
                                float targetX = float.Parse (info[5]);
                                float targetZ = float.Parse (info[6]);
                                //Logger.Log("id: " + playerId + " x: " + targetX + " z: " + targetZ);

                                PlayerControls player = playerControl1;
                                if (playerId == 1) {
                                    player = playerControl2;
                                }
                                actionmove = new MoveAction (player, new Vector3 (targetX, 0, targetZ));
                                while (!actionmove.Update ()) {
                                    actionmove.Update ();
                                }
                                //actionmove.Update();
                                //Logger.Log("player moved!");
                            } else if (info[1].Equals ("pickdrop")) {
                                int playerId = Int32.Parse (info[2]);
                                PlayerControls player = playerControl1;
                                if (playerId == 1) {
                                    player = playerControl2;
                                }
                                actionpickdrop = new PickDropAction (player, false);
                                actionpickdrop.Update ();

                                //Logger.Log("player pick or drop!");
                            } else if (info[1].Equals ("chop")) {
                                int playerId = Int32.Parse (info[2]);
                                PlayerControls player = playerControl1;
                                if (playerId == 1) {
                                    player = playerControl2;
                                }
                                //Logger.Log("chop action start");
                                actionchop = new ChopAction (player);
                                actionchop.Update ();

                                // Logger.Log("player chop!");
                            } else if (info[1].Equals ("turn")) {
                                //Logger.Log("turn action received");
                                int playerId = Int32.Parse (info[2]);
                                int direction = Int32.Parse (info[3]);
                                PlayerControls player = playerControl1;
                                if (playerId == 1) {
                                    player = playerControl2;
                                }
                                actionturn = new TurnAction (player, direction);
                                while (!actionturn.Update ()) {
                                    actionturn.Update ();
                                }
                            }

                        }
                        if (!serverMessage.StartsWith ("action")) {
                            Reply ();
                        }

                    }
                }

            } catch (Exception e) {
                print (e);
                //Logger.Log("exception: " + e);
            }

        }
        /// <summary> 	
        /// Send message to server using socket connection. 	
        /// </summary> 	
        private void Send (string clientMessage) {
            if (socketConnection == null) {
                return;
            }
            try {
                // Get a stream object for writing. 			
                NetworkStream stream = socketConnection.GetStream ();
                if (stream.CanWrite) {
                    // Convert string message to byte array.                 
                    byte[] clientMessageAsByteArray = Encoding.ASCII.GetBytes (clientMessage);
                    // Write byte array to socketConnection stream.                 
                    stream.Write (clientMessageAsByteArray, 0, clientMessageAsByteArray.Length);
                    //Logger.Log("Client sent his message - should be received by server");
                }
            } catch (SocketException socketException) {
                print (socketException);
                //Logger.Log("Socket exception: " + socketException);
            }
        }
    }
}