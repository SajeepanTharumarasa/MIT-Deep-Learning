# Laboratory 2: Computer Vision

##  Debiasing Facial Detection Systems

In this lab, we explore two prominent aspects of applied deep learning: facial detection and algorithmic bias. 

Deploying fair, unbiased AI systems is critical to their long-term acceptance. Consider the task of facial detection: given an image, is it an image of a face?  This seemingly simple, but extremely important, task is subject to significant amounts of algorithmic bias among select demographics. 

We build a facial detection model that learns the *latent variables* underlying face image datasets and uses this to adaptively re-sample the training data, thus mitigating any biases that may be present in order  to train a *debiased* model.

## Datasets

We used three datasets in this lab. In order to train our facial detection models, we need a dataset of positive examples (i.e., of faces) and a dataset of negative examples (i.e., of things that are not faces). We use these data to train our models to classify images as either faces or not faces. Finally, we need a test dataset of face images. Since we're concerned about the potential *bias* of our learned models against certain demographics, it's important that the test dataset we use has equal representation across the demographics or features of interest. In this lab, we consider skin tone and gender. 

1.   **Positive training data**: [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). A large-scale (over 200K images) of celebrity faces.   
2.   **Negative training data**: [ImageNet](http://www.image-net.org/). Many images across many different categories. We'll take negative examples from a variety of non-human categories. 
[Fitzpatrick Scale](https://en.wikipedia.org/wiki/Fitzpatrick_scale) skin type classification system, with each image labeled as "Lighter'' or "Darker''.

## Variational autoencoder (VAE) for learning latent structure

![The concept of a VAE](https://i.ibb.co/3s4S6Gc/vae.jpg)


### The DB-VAE model
![DB-VAE](https://raw.githubusercontent.com/aamini/introtodeeplearning/2019/lab2/img/DB-VAE.png)



### Defining the DB-VAE loss function

loss function have two components:


1.   **VAE loss ($L_{VAE}$)**: consists of the latent loss and the reconstruction loss.
2.   **Classification loss ($L_y(y,\hat{y})$)**: standard cross-entropy loss for a binary classification problem. 

$$L_{total} = L_y(y,\hat{y}) + \mathcal{I}_f(y)\Big[L_{VAE}\Big]$$

### DB-VAE architecture

To build the DB-VAE, we use the standard CNN classifier from above our defined encoder, and then define a decoder network. We create and initialize the two models, and then construct the end-to-end VAE. We use a latent space with 100 latent variables.

The decoder network take as input the sampled latent variables, run them through a series of deconvolutional layers, and output a reconstruction of the original input image.

Finally, we test our DB-VAE model on the test dataset, looking specifically at its accuracy on each the "Dark Male", "Dark Female", "Light Male", and "Light Female" demographics. We compare the performance of this debiased model against the (potentially biased) standard CNN.
