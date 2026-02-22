# Hydra Configuration Files

We can run different trainings by modifying the hyperparameters. All hyperparameters are defined inside the `/config` folder. They are organized by categories to facilitate their use.

In the code, these hyperparameters are read, instead of using typical argparse arguments, using a library called [Hydra](https://hydra.cc/docs/intro/). Hydra allows dynamically creating a hierarchical configuration through composition and modifying it through configuration files and the command line.

### Modifying Hyperparameters with Hydra

#### Selecting Configuration Files

In certain folders, we can see that there is more than one configuration file. One of the strengths of Hydra is being able to have different configurations for the same category.
For example, inside `/config/model` we have two configuration files for two different architectures:

AlexNet

```yaml
name: alexnet
n_classes: ${dataset.n_classes}
normalization: True
```

ResNet

```yaml
name: resnet18
n_classes: ${dataset.n_classes}
normalization: False
```

Using Hydra, we can easily maintain different configurations for each of the architectures. To select which architecture we want to use, we have two ways:

The first will be to run the following command:

```bash
python main.py model=resnet
```

The second will be to modify the configuration files used by default. This information is contained in the configuration file `config/config.yaml`:

```yaml
defaults:
  - dataset: cifar10
  - model: alexnet
  - optimizer: sgd
  - optimizer/loss: cross_entropy
  - optimizer/scheduler: step
  - training: pc
  - _self_
```

#### Modifying Individual Parameters

If we modify one of the files, for example, `/config/training/pc.yaml` from:

```yaml
epochs: 5
batch_size: 64
n_workers: 14
device: "cpu"
gpu_id: 0
seed: 42
print_freq: 100
```

to

```yaml
epochs: 10
batch_size: 64
n_workers: 14
device: "cpu"
gpu_id: 0
seed: 42
print_freq: 100
```

we will be modifying the number of epochs we train our model.
Similarly, if instead of modifying the file, we run

```bash
python main.py training.epochs=10
```

we will be overriding the default value and also training for 10 epochs.

#### Automatic Execution of Multiple Runs

One of the main advantages of Hydra over ArgParse arguments, besides the organization of configurations, is the possibility of automatically defining sweepers over hyperparameters.

If one wants to evaluate the influence of the number of epochs 5, 10, 25, on the model's performance, we can train the model three times by modifying that parameter. However, Hydra gives us the possibility to modify our `config/config.yaml` file and add:

```yaml
hydra:
  sweeper:
    params:
      training.epochs: 5, 10, 25
```

If additionally we run `main.py` using the `-m` argument:

```bash
python main.py -m
```

We will observe how 3 runs are automatically executed one after the other.

If in addition to the number of epochs, we want to measure the impact of the learning rate, we can modify `config/config.yaml` as follows:

```yaml
hydra:
  sweeper:
    params:
      training.epochs: 5, 10, 25
      optimizer.lr: 0.001, 0.01, 0.1
```

and when running `python main.py -m` we will see how we are automatically executing 9 runs. Each of these runs will have its own entry in MLFlow, greatly simplifying the search for optimal hyperparameters.

We can even define sweepers over complete configurations or ranges.

```yaml
hydra:
  sweeper:
    params:
      model: alexnet, testnet
      training.epochs: range(1,10)  
      optimizer.lr: 0.001, 0.01, 0.1
```

> **Note**: This way of using sweepers is the manual form. Hydra has functionalities to integrate more complex algorithms for optimal hyperparameter search such as [Nevergrad](https://hydra.cc/docs/plugins/nevergrad_sweeper/) or [Optuna](https://hydra.cc/docs/plugins/optuna_sweeper/).