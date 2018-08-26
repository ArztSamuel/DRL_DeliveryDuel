# ML-Agents Glossary

 * **Academy** - Unity Component which controls timing, reset, and 
 training/inference settings of the environment. 
 * **Action** - The carrying-out of a decision on the part of an 
 agent within the environment.
 * **Agent** - Unity Component which produces observations and 
 takes actions in the environment. Agents actions are determined 
 by decisions produced by a linked Brain.
 * **Brain** - Unity Component which makes decisions for the agents 
 linked to it.
 * **Decision** - The specification produced by a Brain for an action 
 to be carried out given an observation. 
 * **Editor** - The Unity Editor, which may include any pane 
 (e.g. Hierarchy, Scene, Inspector). 
 * **Environment** - The Unity scene which contains Agents, Academy, 
 and Brains.
 * **FixedUpdate** - Unity method called each time the the game engine 
 is stepped. ML-Agents logic should be placed here.
 * **Frame** - An instance of rendering the main camera for the 
 display. Corresponds to each `Update` call of the game engine.
 * **Observation** - Partial information describing the state of the 
 environment available to a given agent. (e.g. Vector, Visual, Text)
 * **Policy** - Function for producing decisions from observations.
 * **Reward** - Signal provided at every step used to indicate 
 desirability of an agent’s action within the current state 
 of the environment.
 * **State** - The underlying properties of the environment 
 (including all agents within it) at a given time.
 * **Step** - Corresponds to each `FixedUpdate` call of the game engine. 
 Is the smallest atomic change to the state possible.
 * **Update** - Unity function called each time a frame is rendered. 
 ML-Agents logic should not be placed here.
 * **External Coordinator** - ML-Agents class responsible for 
 communication with outside processes (in this case, the Python API).
 * **Trainer** - Python class which is responsible for training a given 
 external brain. Contains TensorFlow graph which makes decisions 
 for external brain.
