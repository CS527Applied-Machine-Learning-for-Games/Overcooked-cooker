# Overcooked-cooker

This is a repo for developing AI agent to play with overcooked.


# Inject command

smi.exe inject -p Overcooked2 -a "path/to/OvercookedAI-Mono.dll" -n Overcooked_Socket -c Loader -m Init


#### To do:

##### C# and Python side

- ~~TestEnv get chef angle/orientation~~ Embedded in getchefpos()

##### Model side

- Fine-tune Behavior cloning model and apply it to initialize the RL model 
- Evalutation


##### Others
- All the objects that can be held by the chef (used in encoding)