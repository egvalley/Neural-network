clear all
clc

% Generate the noise 
noise=randn(1,41);

% Generate the training dataset
i=1;
for x=-1:0.05:1
    y=1.2*sin(pi*x)-cos(2.4*pi*x);
    train_set(i,1)=x;
    train_set(i,2)=y+noise(i)/6;
    i=i+1;
end

% Generate the testing dataset
i=1;
for x=-1:0.01:1
    y=1.2*sin(pi*x)-cos(2.4*pi*x);
    test_set(i,1)=x;
    test_set(i,2)=y;
    i=i+1;
end

% Define the training inputs and targets
x_train = train_set(:,1);
y_train = train_set(:,2);

% Define the radial basis function with a Gaussian activation function
rbf = @(x, c, sigma) exp(-(x-c).^2/(2*sigma^2));

% Define the number of radial basis functions to use
num_rbfs = 15;

% Select the centers of the radial basis functions randomly from the training inputs
centers = datasample(x_train, num_rbfs, 'Replace', false);

% Define the standard deviation of the Gaussian activation function
sigma = (max(centers)-min(centers))/sqrt(2*num_rbfs);

% Compute the radial basis function values for each training input and center
phi = zeros(length(x_train), num_rbfs);
for i = 1:num_rbfs
    phi(:,i) = rbf(x_train, centers(i), sigma);
end

% Solve for the network weights using the Moore-Penrose pseudoinverse
w = pinv(phi) * y_train;

% Define the test inputs and compute the corresponding radial basis function values
x_test = test_set(:,1);
phi_test = zeros(length(x_test), num_rbfs);
for i = 1:num_rbfs
    phi_test(:,i) = rbf(x_test, centers(i), sigma);
end

% Compute the network output for the test inputs using the learned weights and radial basis functions
y_test = phi_test * w;

% Plot the training and testing data, as well as the network approximation
figure;
hold on;
plot(x_train, y_train, 'bo');
plot(x_test, y_test, 'r-', 'LineWidth', 2);
legend('Training Data', 'Network Approximation');
xlabel('x');
ylabel('y');
title('Radial Basis Function Network Approximation');