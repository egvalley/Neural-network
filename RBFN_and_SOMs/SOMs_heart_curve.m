clear all
clc


% Set up the input data as a heart curve
theta = linspace(0, 2*pi, 1000);
x = 16*sin(theta).^3;
y = 13*cos(theta)-5*cos(2*theta)-2*cos(3*theta)-cos(4*theta);


% Set up the SOM parameters
num_neurons = 60;
input_dim = 2;
num_epochs = 800;
learning_rate_initial = 0.1;
sigma_initial=30;


% Initialize the weights
weights = rand(num_neurons, input_dim);

% Train the SOM
for epoch = 1:num_epochs

    % Update the SOM parameters
    learning_rate=learning_rate_initial*exp(-epoch/num_epochs);
    sigma_n = sigma_initial*exp(-epoch/(num_epochs/log(sigma_initial)));

    % Randomly select an input vector
    rand_input_idx=randi(length(x));
    input_vector = [x(rand_input_idx) y(rand_input_idx)];
    
    % Compute the distances between the input vector and all neurons
    distances = pdist2(input_vector, weights);
    
    % Find the winning neuron
    [~, winner] = min(distances);
    
    % Update the weights of the winning neuron and its neighbors
    for neuron = 1:num_neurons
        distance_to_winner = abs(neuron - winner); 

        neighborhood_function = exp(-distance_to_winner^2/(2*sigma_n^2));

        weights(neuron, :) = weights(neuron, :) + learning_rate*neighborhood_function*(input_vector - weights(neuron, :));
        
    end
end

% Plot the trained weights
scatter(weights(:, 1), weights(:, 2), 'filled');
hold on;

% Plot lines to connect every topologically adjacent neurons
for i = 1:num_neurons
    if i < num_neurons
        line([weights(i, 1), weights(i+1, 1)], [weights(i, 2), weights(i+1, 2)], 'Color', 'red');
    else
        line([weights(i, 1), weights(1, 1)], [weights(i, 2), weights(1, 2)], 'Color', 'red');
    end
end

% Plot the heart curve
plot(x, y, 'Color', 'blue', 'LineWidth', 2);
axis equal;
title('Self-organizing map for Heart curve');