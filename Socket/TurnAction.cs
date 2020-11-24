using System;
using UnityEngine;
namespace Overcooked_Socket {

    internal class TurnAction : Action {

        private readonly PlayerControls player;
        //private readonly ClientWorkstation workstation;
        private readonly int offset = 1;
        private Vector3 destination;
        private int direction;
        private float targetX;
        private float targetZ;
        public TurnAction (PlayerControls player, int direction) {
            this.player = player;

            Vector3 current = PlayerUtil.GetChefPosition (player);
            targetX = current.x;
            targetZ = current.z;
            if (direction == 0) { //turn up
                targetZ = targetZ + offset;

            } else if (direction == 1) { //turn right
                targetX = targetX + offset;

            } else if (direction == 2) { //turn down
                targetZ = targetZ - offset;

            } else if (direction == 3) { //turn left
                targetX = targetX - offset;
            }
            this.destination = new Vector3 (targetX, 0, targetZ);

        }

        public bool Update () {
            float facingDif = PlayerUtil.GetAngleFacingDiff (player, destination);

            // Logger.Log($"Facing diff: {facingDif}");

            if (facingDif > 30) {
                Keyboard.Input input = PlayerUtil.GetInputFacing (player, destination);
                // Logger.Log($"Input: {input}");
                Keyboard.Get ().SendDown (input);
                return false;
            }
            Keyboard.Get ().StopXMovement ();
            Keyboard.Get ().StopZMovement ();

            return true;
        }

        public void End () {
            Keyboard.Get ().SendUp (Keyboard.Input.CHOP_THROW);
        }

    }

}