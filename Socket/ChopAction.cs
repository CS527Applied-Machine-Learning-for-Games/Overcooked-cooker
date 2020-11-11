namespace Overcooked_Socket
{

    internal class ChopAction : Action {

        private readonly PlayerControls player;
        //private readonly ClientWorkstation workstation;

        public ChopAction(PlayerControls player) {
            this.player = player;
            //this.workstation = workstation;
            
            Keyboard.Get().SendDown(Keyboard.Input.CHOP_THROW);
            
            //Logger.Log("ChopAction instantiated");
        }

        public bool Update() {
            //return StationUtil.HasFinishedChopping(workstation);
           // Logger.Log("Chop update");
            if (Keyboard.Get().IsKeyDown(Keyboard.Input.CHOP_THROW))
            {
                //Logger.Log("Chop release");
                Keyboard.Get().SendUp(Keyboard.Input.CHOP_THROW);
            }
            
            return true;
        }

        public void End() {
            Keyboard.Get().SendUp(Keyboard.Input.CHOP_THROW);
        }

    }

}
