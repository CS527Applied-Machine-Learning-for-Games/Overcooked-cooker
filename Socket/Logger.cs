//using System;
//using System.Collections;
//using System.IO;
//using UnityEngine;
//using UnityEngine.Networking;
//using Object = UnityEngine.Object;

//namespace Overcooked_Socket
//{
//    internal static class Logger
//    {

//        private readonly static string PATH = "D:\\Output1.txt";

//        public static void Log(String message)
//        {
//            String str = File.ReadAllText(PATH);
//            File.WriteAllText(PATH, str + message + "\n");
//        }

//        public static String FormatPosition(Vector3 location)
//        {
//            return "x=" + Math.Round(location.x, 2) + ", y=" + Math.Round(location.y, 2) + ", z=" + Math.Round(location.z, 2);
//        }
//        public static String FormatCarry(PlayerControls player)
//        {
//            return "isCarry=" + PlayerUtil.IsCarrying(player) + ", carry=" + PlayerUtil.GetCarrying(player);
//        }

//        public static void Clear()
//        {
//            File.WriteAllText(PATH, "");
//        }
//        public static String GetCarry(ClientPlate plate)
//        {
//            ClientPlayerAttachmentCarrier clientCarrier =
//                (ClientPlayerAttachmentCarrier)ReflectionUtil.GetValue(plate, "m_clientCarrier");

//            IClientAttachment[] carriedObjects =
//                (IClientAttachment[])ReflectionUtil.GetValue(clientCarrier, "m_carriedObjects");

//            // Logger.Log($"Number of carried objects: {carriedObjects.Length}");
//            for (int i = 0; i < carriedObjects.Length; i++)
//            {

//                // Logger.Log($"    Carried object is null: {carriedObjects[i] == null}");
//                // Logger.Log($"    CarriedObject Type: {carriedObjects[i].GetType()}");
//                // Logger.Log($"    Carried game object is null: {carriedObjects[i].AccessGameObject() == null}");

//                if (carriedObjects[i] != null)
//                {
//                    if (carriedObjects[i].AccessGameObject() != null)
//                    {
//                        // Logger.Log($"    Carried game object type: {carriedObjects[i].AccessGameObject()}");
//                        return carriedObjects[i].AccessGameObject().name;
//                    }
//                }
//            }
//            return "";
//        }

//        static void func1()
//        {
//            ClientCatchableItem[] items = Object.FindObjectsOfType<ClientCatchableItem>();
//            Logger.Log($"ItemContainer container length: {items.Length}");
//            foreach (var item in items)
//            {

//                //Logger.Log($"IngredientDisposalBehaviour name : {((IngredientAssembledNode)item.GetOrderComposition()).m_ingriedientOrderNode.name}");

//                Logger.Log($"Component name: {item.name}");

//                Logger.Log($"Plate position: {Logger.FormatPosition(item.transform.position)}");

//                //Logger.Log($"tag: {item.GetInstanceID()}");
//                //Logger.Log($"ingerdient:{ContainerUtil.GetIngredient(item)}");
//            }
//            //ClientIngredientSpray[] containers = Object.FindObjectsOfType<ClientIngredientSpray>();
//            //if (containers == null)
//            //{
//            //    Logger.Log("Fail to call IngredientCatcher !");
//            //    return;
//            //}
//            //else
//            //{
//            //    Logger.Log($"container : {containers.Length}");
//            //    foreach (var item in containers)
//            //    {
//            //        Logger.Log($"Component name: {item.name}");
//            //        Logger.Log($"Plate location: {Logger.FormatPosition(item.transform.position)}");
//            //        //Logger.Log($"ingerdient:{ContainerUtil.GetIngredient(item)}");
//            //    }
//            //}
//        }
//        static void func3()
//        {
//            ServerIngredientContainer[] containers = Object.FindObjectsOfType<ServerIngredientContainer>();
            
//            Logger.Log($"ItemContainer container length: {containers.Length}");
//            foreach (var item in containers)
//            {
                
//                //Logger.Log($"IngredientDisposalBehaviour name : {((IngredientAssembledNode)item.GetOrderComposition()).m_ingriedientOrderNode.name}");
                    
//                Logger.Log($"Component name: {item.name}");
                    
//                Logger.Log($"Plate position: {Logger.FormatPosition(item.transform.position)}");

//                Logger.Log($"tag: {item.GetInstanceID()}");
//                //Logger.Log($"ingerdient:{ContainerUtil.GetIngredient(item)}");
//            }
           
//        }

//        public static void LogContainer()
//        {
//            func1();
//            //func3();
//        }
//        public static void LogWorkstations()
//        {
//            Logger.Log("Before method call");

//            //  ClientWorkstation[] workstations = Object.FindObjectsOfType<ClientWorkstation>();
//            //ClientCookingHandler[] cookingHandler = Object.FindObjectsOfType<ClientCookingHandler>();
//            ClientFlammable[] fire = Object.FindObjectsOfType<ClientFlammable>();


//            Logger.Log($"size:{fire.Length}");

//            foreach (var item in fire)
//            {
//                Logger.Log($"Component name: {item.name}");
//                Logger.Log($"Plate location: {Logger.FormatPosition(item.transform.position)}");
//                Logger.Log($"isFire: {item.OnFire()}");
//               // Logger.Log($"IsBurning:{item.IsBurning()}");
//                //Logger.Log($"CookedOrderState:{item.GetCookedOrderState()}");
//                //Logger.Log($"CookedOrderState:{item.GetCookingProgress()}");

//                // Logger.Log($"pick: {ReflectionUtil.CanPickupItem(item)}");
//                //ClientWorkableItem workableItem = (ClientWorkableItem)ReflectionUtil.GetValue(item, "m_item");
//                //if(workableItem == null)
//                //{
//                //    Logger.Log("blank");

//                //}
//                //else
//                //{
//                //    Logger.Log(workableItem.name);
//                //    Logger.Log($"meiqiewan:{workableItem.GetProgress()}");
//                //}
               
//            }
//            Logger.Log("After method call");
//        }

//    }
//}
