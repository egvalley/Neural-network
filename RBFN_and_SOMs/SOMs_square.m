clear all
clc

% Set up the input data as a square
trainX = rands(2,500);
x = trainX(1,:);
y = trainX(2,:);

% Set up the SOM parameters
output_dim = [5 10];
input_dim = 2;
num_epochs = 10000;
learning_rate_initial = 0.6;
sigma_initial=sqrt(output_dim(1)^2+output_dim(2)^2)/2;

% Initialize the weights
weights = rand(output_dim(1)*output_dim(2), input_dim);

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
    winner_col=mod(winner,output_dim(2));
    winner_row=(winner-winner_col)/output_dim(2);

    % Update the weights of the winning neuron and its neighbors
    for neuron = 1:output_dim(1)*output_dim(2)
        % Find the place of the neuron in 2D space
        neuron_col= mod(neuron,output_dim(2));
        neuron_row=(neuron-neuron_col)/output_dim(2);
        % Calculate the distance
        row_dist = neuron_row-winner_row;
        col_dist = neuron_col-winner_col;
        distance_to_winner = sqrt(row_dist^2 + col_dist^2);
        % Update the neurons based on its distance
        neighborhood_function = exp(-distance_to_winner^2/(2*sigma_n^2));
        weights(neuron, :) = weights(neuron, :) + learning_rate*neighborhood_function*(input_vector - weights(neuron, :));        
    end
end

% Plot the trained weights as points in a 2D plane
figure;
scatter(weights(:, 1), weights(:, 2), 'filled');
hold on;

% Plot lines to connect every topologically adjacent neurons
for i = 1:output_dim(1)*output_dim(2)

    [row, col] = ind2sub([output_dim(1) output_dim(2)], i);

        if i-1>=(row-1)*output_dim(2)+1
        left = i-1;
        line([weights(i, 1), weights(left, 1),], [weights(i, 2), weights(left, 2)], 'Color', 'blue','LineWidth', 5);
        end
        if i+1<=row*output_dim(2)
        right=i+1;
        line([weights(i, 1), weights(right, 1),], [weights(i, 2), weights(right, 2)], 'Color', 'blue','LineWidth', 5);
        end
        if i-output_dim(2)>=1
        up=i-output_dim(2);
        line([weights(i, 1), weights(up, 1),], [weights(i, 2), weights(up, 2)], 'Color', 'blue','LineWidth', 5);
        end
        if i+output_dim(2)<=output_dim(1)*output_dim(2)
        down=i+output_dim(2);
        line([weights(i, 1), weights(down, 1),], [weights(i, 2), weights(down, 2)], 'Color', 'blue','LineWidth', 5);
        end

end

% Plot the square curve
plot(x, y, 'Color', 'red', 'LineWidth', 0.1);
axis equal;
title('Self-organizing map for square');
