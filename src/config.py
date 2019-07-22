#Architecture
num_filters=64
image_size=64
hidden_size=64
noise_dimension=64 

#Training
epochs=5
kt_config=0.
diversity_ratio=0.5
lambda_kt=0.0001 
learning_rate=0.0001
batch_size=16

#Saver
project_dir='assets/mnist_model_v2/'
checkpoint_dir='assets/mnist_model_v2/checkpoints'
model_name='assets/mnist_model_v2/checkpoints/mnist_model.model'
snapshot=2440
