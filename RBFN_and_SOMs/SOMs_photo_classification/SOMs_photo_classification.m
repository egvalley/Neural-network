clear all
clc

load MNIST_database.mat;
% train_data = training data, 784x1000 matrix
% train_classlabel = the labels of the training data, 1x1000 vector
% test_data = test data, 784x250 matrix
% train_classlabel = the labels of the test data, 1x250 vector

% Finding training dataset
trainIdx = find(train_classlabel==1|train_classlabel==3|train_classlabel==5|train_classlabel==6|train_classlabel==7|train_classlabel==8|train_classlabel==9); % find the location of classes 0, 1, 2
Train_ClassLabel = train_classlabel(trainIdx)';
Train_Data = train_data(:,trainIdx);

% Finding testing dataset
testIndx = find(test_classlabel==1 | test_classlabel==3| test_classlabel==5| test_classlabel==6| test_classlabel==7| test_classlabel==8| test_classlabel==9); % find the location of classes 0, 1, 2
Test_ClassLabel = test_classlabel(testIndx)';
Test_Data = test_data(:,testIndx);

% Set up the SOM parameters
nenum=10;
output_dim = [nenum nenum];% the 2D neural map 10*10
num_neurons = output_dim(1)*output_dim(1);
num_epochs = 20;
learning_rate_initial = 1;
sigma_initial=sqrt(output_dim(1)^2+output_dim(2)^2)/2;
num_images=size(Train_Data,2);

% Initialize the weights
weights = rand(size(Train_Data,1), num_neurons);

% Step 4: Train the SOM network using the image data
for epoch = 1:num_epochs
    % Shuffle the data for each epoch
    shuffled_data = Train_Data(:, randperm(num_images));
    
    % Update the SOM parameters
    lr=learning_rate_initial*exp(-epoch/num_epochs);
    sigma_n = sigma_initial*exp(-epoch/(num_epochs/log(sigma_initial)));
    
    % Train the SOM network on the shuffled data
    for i = 1:num_images
        x = shuffled_data(:, i);
        [~, bmu] = min(sum((weights - x).^2)); % Find the best matching unit (BMU)
        bmu_row = mod(bmu-1, output_dim(2)) + 1;
        bmu_col = ceil(bmu/output_dim(2));
        for j = 1:num_neurons
            dist = sqrt((bmu_row - mod(j-1, 10) - 1)^2 + (bmu_col - ceil(j/10))^2); % Calculate the distance between the BMU and the current neuron
            neighbor_function = exp(-dist^2/(2*sigma_n^2));
            % Update the weights of the current neuron
            weights(:, j) = weights(:, j) + lr *neighbor_function* (x - weights(:, j));           
        end
    end
end

% Step 5: Generate a semantic map for each neuron
figure;
for i = 1:num_neurons
    % Find the input patterns that activate the current neuron the most
    [~, index] = min(sum((weights(:,i) - Train_Data).^2)); % Find the best matching image and return the index
    weights_label(i)=Train_ClassLabel(index);
    % Visualize the semantic map
    subplot(10, 10, i);
    imshow(reshape(Train_Data(:,index), 28, 28), []);
    sgtitle(sprintf('Neuron %d', i));
end
hold off;

% Make the prediction based on the Semantic map
for i=1:size(Train_Data,2)
    [~, bmu] = min(sum((weights - Train_Data(:,i)).^2)); % Find the best matching unit (BMU)
    Trian_pred(i)=weights_label(bmu);
end

% Make the prediction based on the Semantic map
for i=1:size(Test_Data,2)
    [~, bmu] = min(sum((weights - Test_Data(:,i)).^2)); % Find the best matching unit (BMU)
    Test_pred(i)=weights_label(bmu);
end

accuracy(Test_pred,Trian_pred,Test_ClassLabel,Train_ClassLabel);

% the evaluation function
function accuracy(Test_pred,Trian_pred,Test_ClassLabel,Train_ClassLabel)
    test_record_array=zeros(1,9);
    for i=1:length(Test_pred)
        if Test_pred(i)==Test_ClassLabel(i)
            test_record_array(Test_pred(i))= test_record_array(Test_pred(i))+1;
        end
    end

    train_record_array=zeros(1,9);
    for i=1:length(Test_pred)
        if Trian_pred(i)==Train_ClassLabel(i)
            train_record_array(Test_pred(i))= train_record_array(Test_pred(i))+1;
        end
    end    

% Calculate the accuracy and Display the accuracy
disp("============test");
accuracy = sum(test_record_array)/length(Test_pred);
Z=['Accuracy ',num2str(accuracy)];
disp(Z);
 for i=1:length(test_record_array)
    X=['there are ',num2str(test_record_array(i)), ' successful predictions of ' ,num2str(i), ' in ',num2str(length(Test_pred)),' input cases'];
    disp(X);
 end


 disp("============train");
 accuracy = sum(train_record_array)/length(Trian_pred);
 Z=['Accuracy ',num2str(accuracy)];
disp(Z);
 for i=1:length(train_record_array)
    X=['there are ',num2str(train_record_array(i)), ' successful predictions of ' ,num2str(i), ' in ',num2str(length(Trian_pred)),' input cases'];
    disp(X);
 end
end
