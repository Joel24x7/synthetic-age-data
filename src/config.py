#Architecture
num_filters=64
image_size=64
hidden_size=64
noise_dimension=64 #paper

#Training
epochs=100
kt_config=0.
diversity_ratio=0.5
lambda_kt=0.001 #paper
learning_rate=0.0001
step_decay_rate=2
batch_size=16 #paper

#Saver
project_dir='assets/mnist_model/'
checkpoint_dir='assets/mnist_model/checkpoints'
model_name='assets/mnist_model/checkpoints/mnist_model.model'
snapshot=2440
