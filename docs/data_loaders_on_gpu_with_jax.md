---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "PUFGZggH49zp"}

## Introduction to Data Loaders on GPU with JAX

+++ {"id": "3ia4PKEV5Dr8"}

This tutorial explores different data loading strategies for using **JAX** on a single [**GPU**](https://jax.readthedocs.io/en/latest/glossary.html#term-GPU). While JAX doesn't include a built-in data loader, it seamlessly integrates with popular data loading libraries, including:
*   [**PyTorch DataLoader**](https://github.com/pytorch/data)
*   [**TensorFlow Datasets (TFDS)**](https://github.com/tensorflow/datasets)
*   [**Grain**](https://github.com/google/grain)
*   [**Hugging Face**](https://huggingface.co/docs/datasets/en/use_with_jax#data-loading)

You'll see how to use each of these libraries to efficiently load data for a simple image classification task using the MNIST dataset.

+++ {"id": "-rsMgVtO6asW"}

**Import JAX API**

```{code-cell}
:id: tDJNQ6V-Dg5g

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random, device_put
```

+++ {"id": "TsFdlkSZKp9S"}

**Checking GPU Availability for JAX**

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: N3sqvaF3KJw1
outputId: 491a4cee-45cd-4827-8e88-22ffe49ef354
---
jax.devices()
```

+++ {"id": "qyJ_WTghDnIc"}

**Setting Hyperparameters and Initializing Parameters**

You'll define hyperparameters for your model and data loading, including layer sizes, learning rate, batch size, and the data directory. You'll also initialize the weights and biases for a fully-connected neural network.

```{code-cell}
:id: qLNOSloFDka_

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Function to initialize network parameters for all layers based on defined sizes
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

layer_sizes = [784, 512, 512, 10]  # Layers of the network
step_size = 0.01                   # Learning rate
num_epochs = 8                     # Number of training epochs
batch_size = 128                   # Batch size for training
n_targets = 10                     # Number of classes (digits 0-9)
num_pixels = 28 * 28               # Each MNIST image is 28x28 pixels
data_dir = '/tmp/mnist_dataset'    # Directory for storing the dataset

# Initialize network parameters using the defined layer sizes and a random seed
params = init_network_params(layer_sizes, random.PRNGKey(0))
```

+++ {"id": "rHLdqeI7D2WZ"}

**Model Prediction with Auto-Batching**

In this section, you'll define the `predict` function for your neural network. This function computes the output of the network for a single input image.

To efficiently process multiple images simultaneously, you'll use [`vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax.vmap), which allows you to vectorize the `predict` function and apply it across a batch of inputs. This technique, called auto-batching, improves computational efficiency by leveraging hardware acceleration.

```{code-cell}
:id: bKIYPSkvD1QV

from jax.scipy.special import logsumexp

def relu(x):
  return jnp.maximum(0, x)

def predict(params, image):
  # per-example predictions
  activations = image
  for w, b in params[:-1]:
    outputs = jnp.dot(w, activations) + b
    activations = relu(outputs)

  final_w, final_b = params[-1]
  logits = jnp.dot(final_w, activations) + final_b
  return logits - logsumexp(logits)

# Make a batched version of the `predict` function
batched_predict = vmap(predict, in_axes=(None, 0))
```

+++ {"id": "rLqfeORsERek"}

**Utility and Loss Functions**

You'll now define utility functions for:
- One-hot encoding: Converts class indices to binary vectors.
- Accuracy calculation: Measures the performance of the model on the dataset.
- Loss computation: Calculates the difference between predictions and targets.

To optimize performance:
- [`grad`](https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html#jax.grad) is used to compute gradients of the loss function with respect to network parameters.
- [`jit`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit) compiles the update function, enabling faster execution by leveraging JAX's [XLA](https://openxla.org/xla) compilation.

- [`device_put`](https://jax.readthedocs.io/en/latest/_autosummary/jax.device_put.html) to transfer the dataset to the GPU.

```{code-cell}
:id: sA0a06raEQfS

import time

def one_hot(x, k, dtype=jnp.float32):
  """Create a one-hot encoding of x of size k."""
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

def accuracy(params, images, targets):
  """Calculate the accuracy of predictions."""
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
  return jnp.mean(predicted_class == target_class)

def loss(params, images, targets):
  """Calculate the loss between predictions and targets."""
  preds = batched_predict(params, images)
  return -jnp.mean(preds * targets)

@jit
def update(params, x, y):
  """Update the network parameters using gradient descent."""
  grads = grad(loss)(params, x, y)
  return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]

def reshape_and_one_hot(x, y):
    """Reshape and one-hot encode the inputs."""
    x = jnp.reshape(x, (len(x), num_pixels))
    y = one_hot(y, n_targets)
    return x, y

def train_model(num_epochs, params, training_generator, data_loader_type='streamed'):
    """Train the model for a given number of epochs and device_put for GPU transfer."""
    for epoch in range(num_epochs):
        start_time = time.time()
        for x, y in training_generator() if data_loader_type == 'streamed' else training_generator:
            x, y = reshape_and_one_hot(x, y)
            x, y = device_put(x), device_put(y)
            params = update(params, x, y)

        print(f"Epoch {epoch + 1} in {time.time() - start_time:.2f} sec: "
              f"Train Accuracy: {accuracy(params, train_images, train_labels):.4f}, "
              f"Test Accuracy: {accuracy(params, test_images, test_labels):.4f}")
```

+++ {"id": "Hsionp5IYsQ9"}

## Loading Data with PyTorch DataLoader

This section shows how to load the MNIST dataset using PyTorch's DataLoader, convert the data to NumPy arrays, and apply transformations to flatten and cast images.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: uA7XY0OezHse
outputId: cc3cea2c-e9c3-490d-c0ab-ec5ae504a2b2
---
!pip install torch torchvision
```

```{code-cell}
:id: kO5_WzwY59gE

import numpy as np
from jax.tree_util import tree_map
from torch.utils import data
from torchvision.datasets import MNIST
```

```{code-cell}
:id: 6f6qU8PCc143

def numpy_collate(batch):
  """Collate function to convert a batch of PyTorch data into NumPy arrays."""
  return tree_map(np.asarray, data.default_collate(batch))

class NumpyLoader(data.DataLoader):
    """Custom DataLoader to return NumPy arrays from a PyTorch Dataset."""
    def __init__(self, dataset, batch_size=1,
                  shuffle=False, sampler=None,
                  batch_sampler=None, num_workers=0,
                  pin_memory=False, drop_last=False,
                  timeout=0, worker_init_fn=None):
      super(self.__class__, self).__init__(dataset,
          batch_size=batch_size,
          shuffle=shuffle,
          sampler=sampler,
          batch_sampler=batch_sampler,
          num_workers=num_workers,
          collate_fn=numpy_collate,
          pin_memory=pin_memory,
          drop_last=drop_last,
          timeout=timeout,
          worker_init_fn=worker_init_fn)
class FlattenAndCast(object):
  """Transform class to flatten and cast images to float32."""
  def __call__(self, pic):
    return np.ravel(np.array(pic, dtype=jnp.float32))
```

+++ {"id": "mfSnfJND6I8G"}

**Load Dataset with Transformations**

Standardize the data by flattening the images, casting them to `float32`, and ensuring consistent data types.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: Kxbl6bcx6crv
outputId: 15883515-a465-4bbb-9707-0afb007754ca
---
mnist_dataset = MNIST(data_dir, download=True, transform=FlattenAndCast())
```

+++ {"id": "kbdsqvPZGrsa"}

**Full Training Dataset for Accuracy Checks**

Convert the entire training dataset to JAX arrays.

```{code-cell}
:id: c9ZCJq_rzPck

train_images = np.array(mnist_dataset.data).reshape(len(mnist_dataset.data), -1)
train_labels = one_hot(np.array(mnist_dataset.targets), n_targets)
```

+++ {"id": "WXUh0BwvG8Ko"}

**Get Full Test Dataset**

Load and process the full test dataset.

```{code-cell}
:id: brlLG4SqGphm

mnist_dataset_test = MNIST(data_dir, download=True, train=False)
test_images = jnp.array(mnist_dataset_test.data.numpy().reshape(len(mnist_dataset_test.data), -1), dtype=jnp.float32)
test_labels = one_hot(np.array(mnist_dataset_test.targets), n_targets)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: Oz-UVnCxG5E8
outputId: e99c1c2f-1bd9-4f47-ece0-65c886117fb7
---
print('Train:', train_images.shape, train_labels.shape)
print('Test:', test_images.shape, test_labels.shape)
```

+++ {"id": "mNjn9dMPitKL"}

**Training Data Generator**

Define a generator function using PyTorch's DataLoader for batch training.
Setting `num_workers > 0` enables multi-process data loading, which can accelerate data loading for larger datasets or intensive preprocessing tasks. Experiment with different values to find the optimal setting for your hardware and workload.

Note: When setting `num_workers > 0`, you may see the following `RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.`
This warning can be safely ignored since data loaders do not use JAX within the forked processes.

```{code-cell}
:id: 0LdT8P8aisWF

def pytorch_training_generator(mnist_dataset):
    return NumpyLoader(mnist_dataset, batch_size=batch_size, num_workers=0)
```

+++ {"id": "Xzt2x9S1HC3T"}

**Training Loop (PyTorch DataLoader)**

The training loop uses the PyTorch DataLoader to iterate through batches and update model parameters.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: SqweRz_98sN8
outputId: 0f2bf4c6-f053-4125-bf9a-b3fdac0d4aef
---
train_model(num_epochs, params, pytorch_training_generator(mnist_dataset), data_loader_type='iterable')
```

+++ {"id": "Nm45ZTo6yrf5"}

## Loading Data with TensorFlow Datasets (TFDS)

This section demonstrates how to load the MNIST dataset using TFDS, fetch the full dataset for evaluation, and define a training generator for batch processing. GPU usage is explicitly disabled for TensorFlow.

```{code-cell}
:id: sGaQAk1DHMUx

import tensorflow_datasets as tfds
import tensorflow as tf
```

+++ {"id": "ZSc5K0Eiwm4L"}

**Fetch Full Dataset for Evaluation**

Load the dataset with `tfds.load`, convert it to NumPy arrays, and process it for evaluation.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 104
  referenced_widgets: [05f0a37837a64858822223d67bf16ab2, 2c3d6eb7cd6248de9af64a6d9aef2605,
    1b23d132236845bb98eef458aa1da61c, e2dd0e4d6844457a9e026f28ae216f3d, 0725b8f96f154f9f8e31c2e7c8250d7a,
    a2808570d0874e3d8019d479f33221dd, 76ca055340a74f36bb3fa67b8f936310, ffb8992aa6724051af0e0bbcbea7a6d3,
    d26f223b0d61486ca49b3869b5cb2888, 34b97b4424f744468772974ddccd4e1f, f6cbd14a91194e0080776b86a7711db0]
id: 1hOamw_7C8Pb
outputId: 00de53b4-0249-4c46-82d1-f165ca60105d
---
# tfds.load returns tf.Tensors (or tf.data.Datasets if batch_size != -1)
mnist_data, info = tfds.load(name="mnist", batch_size=-1, data_dir=data_dir, with_info=True)
mnist_data = tfds.as_numpy(mnist_data)
train_data, test_data = mnist_data['train'], mnist_data['test']

# Full train set
train_images, train_labels = train_data['image'], train_data['label']
train_images = jnp.reshape(train_images, (len(train_images), num_pixels))
train_labels = one_hot(train_labels, n_targets)

# Full test set
test_images, test_labels = test_data['image'], test_data['label']
test_images = jnp.reshape(test_images, (len(test_images), num_pixels))
test_labels = one_hot(test_labels, n_targets)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: Td3PiLdmEf7z
outputId: 26ece11c-e89c-4297-8b68-3cc845e92e35
---
print('Train:', train_images.shape, train_labels.shape)
print('Test:', test_images.shape, test_labels.shape)
```

+++ {"id": "dXMvgk6sdq4j"}

**Define the Training Generator**

Create a generator function to yield batches of data for training.

```{code-cell}
:id: vX59u8CqEf4J

def training_generator():
  # as_supervised=True gives us the (image, label) as a tuple instead of a dict
  ds = tfds.load(name='mnist', split='train', as_supervised=True, data_dir=data_dir)
  # You can build up an arbitrary tf.data input pipeline
  ds = ds.batch(batch_size).prefetch(1)
  # tfds.dataset_as_numpy converts the tf.data.Dataset into an iterable of NumPy arrays
  return tfds.as_numpy(ds)
```

+++ {"id": "EAWeUdnuFNBY"}

**Training Loop (TFDS)**

Use the training generator in a custom training loop.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: h2sO13XDGvq1
outputId: 9acd342c-9f02-4e16-e999-a4e4bf56ffb3
---
train_model(num_epochs, params, training_generator)
```

+++ {"id": "-ryVkrAITS9Z"}

## Loading Data with Grain

This section demonstrates how to load MNIST data using Grain, a data-loading library. You'll define a custom dataset class for Grain and set up a Grain DataLoader for efficient training.

+++ {"id": "waYhUMUGmhH-"}

Install Grain

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: L78o7eeyGvn5
outputId: 0339a5b1-c5da-4d2e-f407-39e301d18779
---
!pip install grain
```

+++ {"id": "66bH3ZDJ7Iat"}

Import Required Libraries (import MNIST dataset from torchvision)

```{code-cell}
:id: mS62eVL9Ifmz

import numpy as np
import grain.python as pygrain
from torchvision.datasets import MNIST
```

+++ {"id": "0h6mwVrspPA-"}

**Define Dataset Class**

Create a custom dataset class to load MNIST data for Grain.

```{code-cell}
:id: bnrhac5Hh7y1

class Dataset:
    def __init__(self, data_dir, train=True):
        self.data_dir = data_dir
        self.train = train
        self.load_data()

    def load_data(self):
        self.dataset = MNIST(self.data_dir, download=True, train=self.train)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return np.ravel(np.array(img, dtype=np.float32)), label
```

+++ {"id": "53mf8bWEsyTr"}

**Initialize the Dataset**

```{code-cell}
:id: pN3oF7-ostGE

mnist_dataset = Dataset(data_dir)
```

+++ {"id": "GqD-ycgBuwv9"}

**Get the full train and test dataset**

```{code-cell}
:id: f1VnTuX3u_kL

# Convert training data to JAX arrays and encode labels as one-hot vectors
train_images = jnp.array([mnist_dataset[i][0] for i in range(len(mnist_dataset))], dtype=jnp.float32)
train_labels = one_hot(np.array([mnist_dataset[i][1] for i in range(len(mnist_dataset))]), n_targets)

# Load test dataset and process it
mnist_dataset_test = MNIST(data_dir, download=True, train=False)
test_images = jnp.array([np.ravel(np.array(mnist_dataset_test[i][0], dtype=np.float32)) for i in range(len(mnist_dataset_test))], dtype=jnp.float32)
test_labels = one_hot(np.array([mnist_dataset_test[i][1] for i in range(len(mnist_dataset_test))]), n_targets)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: a2NHlp9klrQL
outputId: 43d66d8c-4656-49f9-93cc-b18e8d309e45
---
print("Train:", train_images.shape, train_labels.shape)
print("Test:", test_images.shape, test_labels.shape)
```

+++ {"id": "1QPbXt7O0JN-"}

**Initialize PyGrain DataLoader**

```{code-cell}
:id: 2jqd1jJt25Bj

sampler = pygrain.SequentialSampler(
    num_records=len(mnist_dataset),
    shard_options=pygrain.NoSharding()) # Single-device, no sharding

def pygrain_training_generator():
    return pygrain.DataLoader(
        data_source=mnist_dataset,
        sampler=sampler,
        operations=[pygrain.Batch(batch_size)],
    )
```

+++ {"id": "mV5z4GLCGKlx"}

**Training Loop (Grain)**

Run the training loop using the Grain DataLoader.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 9-iANQ-9CcW_
outputId: 283429a8-6573-4cd2-b75a-6e86abf2906e
---
train_model(num_epochs, params, pygrain_training_generator)
```

+++ {"id": "o51P6lr86wz-"}

## Loading Data with Hugging Face

This section demonstrates loading MNIST data using the Hugging Face `datasets` library. You'll format the dataset for JAX compatibility, prepare flattened images and one-hot-encoded labels, and define a training generator.

+++ {"id": "69vrihaOi4Oz"}

Install the Hugging Face `datasets` library.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: 19ipxPhI6oSN
outputId: ecf9d716-770d-4779-c4bf-8cff51e0fb57
---
!pip install datasets
```

```{code-cell}
:id: 8v1N59p76zn0

from datasets import load_dataset
```

+++ {"id": "8Gaj11tO7C86"}

Load the MNIST dataset from Hugging Face and format it as `numpy` arrays for quick access or `jax` to get JAX arrays.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
  height: 301
  referenced_widgets: [fa01cba0920a41af82c7fdb8dd04f040, 8374588ccdcb4d0088f6e88453f808d5,
    210aa8da078b4a65afad073cd6d9d00b, 55012f250aec49a7869eaa7949ffafe9, d5917f5b4550454f93f17ae37ebac0e0,
    33a6573c73cc4116b496ef8151b6c622, bc08d6e2b5d9441992009c19fc6d3023, e514ee4117bb41c78cb05626da40f0bb,
    84ec654ce3294a3fb7a36b0875fbe77d, 09bc23276dd24554a21e948140941462, cc984a6292f14d1d94f5a3bb6670e4b2,
    0308b41d985a47fd9af56c11a907c0da, 2b32d8dd61e04c49b531eee64860d729, 3fcad4bdcaf94728bae40c2b6a58c9be,
    780fa609782a45e897deb3644d1634a3, cfabce6e6dfd4efd9192514e7f00119b, 3083ec1b4b064bd6b83bd9fcece24325,
    2101e800875b4c69be2d599da0fc7e9b, 099f92015ecd47259a88b19c2588a669, 737e85b818694b75a9a3ed2cb8bd5190,
    7325be43ac7c4cceb6bb59362a1adfd2, 4ec304a224a34fb9aa894ff8afcaa819, fc38acba420a4a98bc9c35697565e3bd,
    e7a0c8ca2ed04a209f7a865eb6e84d63, 8052edc2f8d546ffb3c03f8aecd0f7e5, b2c932a501d246f5a6af5c0005120d43,
    3b6bd24f99c44ad0af2ca3f1b1ceabcb, 9ab7b33f760e4e2297b7010a955fbd40, 631b8a7abe17404bbe01e3cc489be251,
    b2b0f76b830b43b1a34f1165f499beeb, a89aac1c227b4561b4428b2641a61aba, 90a80a80ff01443c84533f3c6335bee8,
    eb094f1370184aadaa9ef0ba10f36054, 36fcd25da9254a31bf2c5e62369835e3, 941a0107792f4738ab95ecebadf2450d,
    777b1a745e25431db7e3a7ddeeb32615, a1d9ce9f74a74cde9474b7ac0698d338, d85ea433b58a4b83bcf872731edad068,
    1cc16817259545d6b3c69521be7d16d1, c1b91cff736446a3905783bd2b8fe2f5, a4338bd847b04271adf1b8b5c11e87da,
    0380b6673c67458d840ff44fe172da64, b7257ae6cebb4b16baf91d22555d2225, bce6026538754542be31128d506608a7,
    f189c205983745549e4aacfe9ed69e0e, e5aa4252bcde4cd19abe8956843ae8ee, f1fc47db306049f6a18419b49aa44fba,
    3ed4c24ed5f449808e0aebc1d16ce07f, ef187cd9d168464e8e0154e45a717131, caf54b270467495aaa416122f0e05d46,
    df768662e46f4bd382a9c8900bfbe152, 3800500ded8a4231bdf2434fb9be7a81, 6c92dbcae2be4ce682d5512a393520e5,
    29eaaa5eda704c2f9733ee02b4ca3e27, 177a8f6ad2884d33901fae2635c21d26]
id: a22kTvgk6_fJ
outputId: 6f051e99-a928-46bd-ecba-be639f0693ec
---
mnist_dataset = load_dataset("mnist", cache_dir=data_dir).with_format("numpy")
```

+++ {"id": "tgI7dIaX7JzM"}

**Extract images and labels**

Get image shape and flatten for model input.

```{code-cell}
:id: NHrKatD_7HbH

train_images = mnist_dataset["train"]["image"]
train_labels = mnist_dataset["train"]["label"]
test_images = mnist_dataset["test"]["image"]
test_labels = mnist_dataset["test"]["label"]

# Extract image shape
image_shape = train_images.shape[1:]
num_features = image_shape[0] * image_shape[1]

# Flatten the images
train_images = train_images.reshape(-1, num_features)
test_images = test_images.reshape(-1, num_features)

# One-hot encode the labels
train_labels = one_hot(train_labels, n_targets)
test_labels = one_hot(test_labels, n_targets)
```

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: dITh435Z7Nwb
outputId: ab47d9a3-8aa6-4cad-c78a-8663d268e201
---
print('Train:', train_images.shape, train_labels.shape)
print('Test:', test_images.shape, test_labels.shape)
```

+++ {"id": "kk_4zJlz7T1E"}

**Define Training Generator**

Set up a generator to yield batches of images and labels for training.

```{code-cell}
:id: -zLJhogj7RL-

def hf_training_generator():
    """Yield batches for training."""
    for batch in mnist_dataset["train"].iter(batch_size):
        x, y = batch["image"], batch["label"]
        yield x, y
```

+++ {"id": "HIsGfkLI7dvZ"}

**Training Loop (Hugging Face Datasets)**

Run the training loop using the Hugging Face training generator.

```{code-cell}
---
colab:
  base_uri: https://localhost:8080/
id: Ui6aLiZP7aLe
outputId: 675d6cb7-7d4c-4662-f421-a8181924627e
---
train_model(num_epochs, params, hf_training_generator)
```

+++ {"id": "rCJq2rvKlKWX"}

## **Summary**

This notebook has guided you through efficient methods for loading data on a GPU when using JAX. You’ve learned how to leverage popular libraries such as PyTorch DataLoader, TensorFlow Datasets, Grain, and Hugging Face Datasets to streamline the data loading process for your machine learning tasks. Each of these methods offers unique advantages and considerations, allowing you to choose the best approach based on the specific needs of your project.
