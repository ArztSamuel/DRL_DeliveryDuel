# Training ML-Agents

ML-Agents conducts training using an external Python training process. During training, this external process communicates with the Academy object in the Unity scene to generate a block of agent experiences. These experiences become the training set for a neural network used to optimize the agent's policy (which is essentially a mathematical function mapping observations to actions). In reinforcement learning, the neural network optimizes the policy by maximizing the expected rewards. In imitation learning, the neural network optimizes the policy to achieve the smallest difference between the actions chosen by the agent trainee and the actions chosen by the expert in the same situation. 

The output of the training process is a model file containing the optimized policy. This model file is a TensorFlow data graph containing the mathematical operations and the optimized weights selected during the training process. You can use the generated model file with the Internal Brain type in your Unity project to decide the best course of action for an agent. 

Use the Python program, `learn.py` to train your agents. This program can be found in the `python` directory of the ML-Agents SDK. The [configuration file](#training-config-file), `trainer_config.yaml` specifies the hyperparameters used during training. You can edit this file with a text editor to add a specific configuration for each brain.

For a broader overview of reinforcement learning, imitation learning and the ML-Agents training process, see [ML-Agents Overview](ML-Agents-Overview.md).

## Training with Learn.py

Use the Python `Learn.py` program to train agents. `Learn.py` supports training with [reinforcement learning](Background-Machine-Learning.md#reinforcement-learning), [curriculum learning](Training-Curriculum-Learning.md), and [behavioral cloning imitation learning](Training-Imitation-Learning.md).

Run `Learn.py` from the command line to launch the training process. Use the command line patterns and the `trainer_config.yaml` file to control training options.

The basic command for training is:

    python3 learn.py <env_file_path> --run-id=<run-identifier> --train

where `<env_file_path>` is the path to your Unity executable containing the agents to be trained and `<run-identifier>` is an optional identifier you can use to identify the results of individual training runs.

For example, suppose you have a project in Unity named "CatsOnBicyclesCatsOnBicycles" which contains agents ready to train. To perform the training:

1. Build the project, making sure that you only include the training scene.
2. Open a terminal or console window.
3. Navigate to the ml-agents `python` folder.
4. Run the following to launch the training process using the path to the Unity environment you built in step 1:

        python3 learn.py ../../projects/Cats/CatsOnBicycles.app --run-id=cob_1 --train

During a training session, the training program prints out and saves updates at regular intervals (specified by the `summary_freq` option). The saved statistics are grouped by the `run-id` value so you should assign a unique id to each training run if you plan to view the statistics. You can view these statistics using TensorBoard during or after training by running the following command (from the ML-Agents python directory):

    tensorboard --logdir=summaries

And then opening the URL: [localhost:6006](http://localhost:6006).
 
When training is finished, you can find the saved model in the `python/models` folder under the assigned run-id — in the cats example, the path to the model would be `python/models/cob_1/CatsOnBicycles_cob_1.bytes`.

While this example used the default training hyperparameters, you can edit the [training_config.yaml file](#training-config-file) with a text editor to set different values. 

### Command line training options

In addition to passing the path of the Unity executable containing your training environment, you can set the following command line options when invoking `learn.py`:

* `--curriculum=<file>` – Specify a curriculum JSON file for defining the lessons for curriculum training. See [Curriculum Training](Training-Curriculum-Learning.md) for more information.
* `--keep-checkpoints=<n>` – Specify the maximum number of model checkpoints to keep. Checkpoints are saved after the number of steps specified by the `save-freq` option. Once the maximum number of checkpoints has been reached, the oldest checkpoint is deleted when saving a new checkpoint. Defaults to 5.
* `--lesson=<n>` – Specify which lesson to start with when performing curriculum training. Defaults to 0.
* `--load` – If set, the training code loads an already trained model to initialize the neural network before training. The learning code looks for the model in `python/models/<run-id>/` (which is also where it saves models at the end of training). When not set (the default), the neural network weights are randomly initialized and an existing model is not loaded.
* `--run-id=<path>` – Specifies an identifier for each training run. This identifier is used to name the subdirectories in which the trained model and summary statistics are saved as well as the saved model itself. The default id is "ppo". If you use TensorBoard to view the training statistics, always set a unique run-id for each training run. (The statistics for all runs with the same id are combined as if they were produced by a the same session.)
* `--save-freq=<n>` Specifies how often (in  steps) to save the model during training. Defaults to 50000.
* `--seed=<n>` – Specifies a number to use as a seed for the random number generator used by the training code.
* `--slow` – Specify this option to run the Unity environment at normal, game speed. The `--slow` mode uses the **Time Scale** and **Target Frame Rate** specified in the Academy's **Inference Configuration**. By default, training runs using the speeds specified in your Academy's **Training Configuration**. See [Academy Properties](Learning-Environment-Design-Academy.md#academy-properties).
* `--train` – Specifies whether to train model or only run in inference mode. When training, **always** use the `--train` option.
* `--worker-id=<n>` – When you are running more than one training environment at the same time, assign each a unique worker-id number. The worker-id is added to the communication port opened between the current instance of learn.py and the ExternalCommunicator object in the Unity environment. Defaults to 0.
* `--docker-target-name=<dt>` – The Docker Volume on which to store curriculum, executable and model files. See [Using Docker](Using-Docker.md).

### Training config file

The training config file, `trainer_config.yaml` specifies the training method, the hyperparameters, and a few additional values to use during training. The file is divided into sections. The **default** section defines the default values for all the available settings. You can also add new sections to override these defaults to train specific Brains. Name each of these override sections after the GameObject containing the Brain component that should use these settings. (This GameObject will be a child of the Academy in your scene.) Sections for the example environments are included in the provided config file. `Learn.py` finds the config file by name and looks for it in the same directory as itself.

| ** Setting ** | **Description** | **Applies To Trainer**|
| :--                | :--                       | :--                                  |
| batch_size | The number of experiences in each iteration of gradient descent.| PPO, BC |
| batches_per_epoch | In imitation learning, the number of batches of training examples to collect before training the model.| BC |
| beta | The strength of entropy regularization.| PPO, BC |
| brain_to_imitate | For imitation learning, the name of the GameObject containing the Brain component to imitate. | BC |
| buffer_size | The number of experiences to collect before updating the policy model. | PPO, BC |
| epsilon | Influences how rapidly the policy can evolve during training.| PPO, BC |
| gamma | The reward discount rate for the Generalized Advantage Estimator (GAE).  | PPO  |
| hidden_units | The number of units in the hidden layers of the neural network. | PPO, BC |
| lambd | The regularization parameter. | PPO  |
| learning_rate | The initial learning rate for gradient descent. | PPO,  BC |
| max_steps | The maximum number of simulation steps to run during a training session. | PPO, BC |
| memory_size | The size of the memory an agent must keep. Used for training with a recurrent neural network. See [Using Recurrent Neural Networks](Feature-Memory.md). | PPO, BC |
| normalize | Whether to automatically normalize observations. | PPO, BC |
| num_epoch | The number of passes to make through the experience buffer when performing gradient descent optimization. | PPO, BC |
| num_layers | The number of hidden layers in the neural network. | PPO, BC |
| sequence_length | Defines how long the sequences of experiences must be while training. Only used for training with a recurrent neural network. See [Using Recurrent Neural Networks](Feature-Memory.md). | PPO, BC |
| summary_freq | How often, in steps, to save training statistics. This determines the number of data points shown by TensorBoard. | PPO, BC |
| time_horizon | How many steps of experience to collect per-agent before adding it to the experience buffer. | PPO, BC |
| trainer | The type of training to perform: "ppo" or "imitation".| PPO, BC |
| use_recurrent | Train using a recurrent neural network. See [Using Recurrent Neural Networks](Feature-Memory.md).| PPO, BC |
|| PPO = Proximal Policy Optimization, BC = Behavioral Cloning (Imitation)) ||

For specific advice on setting hyperparameters based on the type of training you are conducting, see:

* [Training with PPO](Training-PPO.md)
* [Using Recurrent Neural Networks](Feature-Memory.md)
* [Training with Curriculum Learning](Training-Curriculum-Learning.md)
* [Training with Imitation Learning](Training-Imitation-Learning.md)

You can also compare the [example environments](Learning-Environment-Examples.md) to the corresponding sections of the `trainer-config.yaml` file for each example to see how the hyperparameters and other configuration variables have been changed from the defaults.
