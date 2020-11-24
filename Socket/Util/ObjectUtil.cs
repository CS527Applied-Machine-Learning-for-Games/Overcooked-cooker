using UnityEngine;

namespace Overcooked_Socket {
    internal static class ObjectUtil {

        public static ClientKitchenFlowControllerBase GetFlowController () {
            return Object.FindObjectOfType<ClientKitchenFlowControllerBase> ();
        }

        public static PlayerControls GetBotControls () {
            // Logger.Log($"PlayerControl instances count: {Object.FindObjectsOfType<PlayerControls>().Length}");
            return Object.FindObjectsOfType<PlayerControls> () [0];
        }
    }
}