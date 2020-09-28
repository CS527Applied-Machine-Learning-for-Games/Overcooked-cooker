using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Text;
using UnityEngine;
using ZeroMQ;
using NetMQ;
using NetMQ.Sockets;

namespace Overcooked_Socket
{
    internal class Main : MonoBehaviour
    {
        #region private members 	
        private TcpClient socketConnection;
        private Thread clientReceiveThread;
        private Vector3 playerPos0;
        private Vector3 playerPos1;
        #endregion
        public void Start()
        {
            Logger.Clear();
            Logger.Log("Loaded.");
            ConnectToTcpServer();
        }
        public static string GetNewOrder()
        {
            ArrayList orders = OrderUtil.GetOrders(ObjectUtil.GetFlowController());

            int index = 1;

            string rtn = "Order:";

            foreach (string order in orders)
            {
                rtn += ($"#{index++}: {order} ");
            }

            return rtn;
        }
        private void Reply()
        {
            playerPos0 = PlayerUtil.GetChefPosition(0);
            playerPos1 = PlayerUtil.GetChefPosition(1);
            string msg = "";
            msg += "";
            msg += $"Player 0: {Logger.FormatPosition(playerPos0)}, a={Convert.ToDouble(PlayerUtil.GetChefAngles(0)).ToString("0.00")} ";
            msg += $"Player 1: {Logger.FormatPosition(playerPos1)}, a={Convert.ToDouble(PlayerUtil.GetChefAngles(1)).ToString("0.00")} ";
            msg += GetNewOrder();
            //Logger.Log(msg);
            Send(msg);
        }
        public void Update()
        {
            if (Input.GetKeyDown(KeyCode.Keypad9))
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
                socketConnection = new TcpClient("localhost", 5555);
                Byte[] bytes = new Byte[1024];
                while (true)
                {
                    // Get a stream object for reading 				
                    using (NetworkStream stream = socketConnection.GetStream())
                    {
                        int length;
                        // Read incomming stream into byte arrary. 					
                        while ((length = stream.Read(bytes, 0, bytes.Length)) != 0)
                        {
                            var incommingData = new byte[length];
                            Array.Copy(bytes, 0, incommingData, 0, length);
                            // Convert byte array to string message. 						
                            string serverMessage = Encoding.ASCII.GetString(incommingData);
                            //Logger.Log("server message received as: " + serverMessage);
                            Reply();
                        }
                    }
                }
            }
            catch (SocketException socketException)
            {
                Logger.Log("Socket exception: " + socketException);
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
