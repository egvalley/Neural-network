clear all
clc

load('traindata.mat');
load('testdata.mat');
load('testdata2.mat');

% Construct and configure the MLP
epochs = 1000; 
train_num = size(traindata,1);

% Set the train dataset
x=traindata(:,1)';
t=traindata(:,2)';

% specify the structure and learning algorithm for MLP
net = fitnet(20,'trainlm');
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';
net = configure(net,x,t);
view(net);


 % Train the network in sequential mode
 for i = 1 : epochs
     display(['Epoch: ', num2str(i)])
     idx = randperm(train_num); % shuffle the input
     net = adapt(net, x(:,idx), t(:,idx));
 end

% Generate the test input data
input=testdata2(:,1)';
desiredout=testdata2(:,2)';

% Feed the input
pred = net(input);
perf = perform(net, desiredout, pred);

% Plot the fitting results and compare
figure;
plot(input',pred','*','MarkerFaceColor', 'b', 'MarkerEdgeColor', 'b','MarkerSize',10);% Plot the fitting dataset
hold on;
plot(testdata2(:,1),testdata2(:,2),'o','MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r','MarkerSize',5);% Plot the original test dataset