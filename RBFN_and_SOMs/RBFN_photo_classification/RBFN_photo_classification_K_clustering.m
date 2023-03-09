clear all
clc

load MNIST_database.mat;
% train_data = training data, 784x1000 matrix
% train_classlabel = the labels of the training data, 1x1000 vector
% test_data = test data, 784x250 matrix
% train_classlabel = the labels of the test data, 1x250 vector

trainIdx = find(train_classlabel==2 | train_classlabel==4); % find the location of classes 0, 1, 2
Train_ClassLabel = train_classlabel(trainIdx)';
for tmp=1:length(Train_ClassLabel)
    if Train_ClassLabel(tmp)==2
        Train_ClassLabel(tmp)=0
    else
        Train_ClassLabel(tmp)=1
    end
end
Train_Data = train_data(:,trainIdx);

testIdx = find(test_classlabel==2 | test_classlabel==4); % find the location of classes 0, 1, 2
Test_ClassLabel = test_classlabel(testIdx)';
for tmp=1:length(Test_ClassLabel)
    if Test_ClassLabel(tmp)==2
        Test_ClassLabel(tmp)=0
    else
        Test_ClassLabel(tmp)=1
    end
end
Test_Data = test_data(:,testIdx);

% Define the training inputs and targets
x_train = Train_Data;
y_train = Train_ClassLabel;

% Define the radial basis function with a Gaussian activation function
rbf = @(x, sigma) exp(-x.^2/(2*sigma^2));

% Define the number of radial basis functions to use
num_rbfs = 2;

% Select the centers of the radial basis functions with k-means clustering
[idx,centers] = kmeans(Train_Data',num_rbfs)

% Define the standard deviation of the Gaussian activation function
sigma = 100;

% Compute the radial basis function values for each training input and center
% phi = zeros(size(x_train,2), num_rbfs);
for i = 1:num_rbfs
    for j = 1 : size(x_train,2)
        Eucdistance1(j,:)=pdist([x_train(:,j)';centers(i,:)]);
    end    
    phi(:,i) = rbf(Eucdistance1, sigma);
end


% Solve for the network weights using the Moore-Penrose pseudoinverse
lambda=0;
w=inv(phi'*phi+lambda*eye(size(phi'*phi,1),size(phi'*phi,2)))*phi'*y_train;

% Finding the prediction accuracy for the training dataset
Trian_pred=phi*w;

% Define the test inputs and compute the corresponding radial basis function values
x_test = Test_Data;
for i = 1:num_rbfs
    for j = 1 : size(x_test,2)
        Eucdistance2(j,:)=pdist([x_test(:,j)';centers(i,:)]);
    end
    phi_test(:,i) = rbf(Eucdistance2, sigma);
end

% Compute the network output for the test inputs using the learned weights and radial basis functions
Test_pred = phi_test * w;
Test_pred_round=round(Test_pred);


evaluation(Train_ClassLabel,Test_ClassLabel,Trian_pred,Test_pred);
disp("============test");
calacc_classification(Test_ClassLabel,Test_pred_round,size(Test_ClassLabel,1));
disp("============");

function evaluation(TrLabel,TeLabel,TrPred,TePred)

    TrAcc = zeros(1,1000);
    TeAcc = zeros(1,1000);
    thr = zeros(1,1000);
    TrN = length(TrLabel);
    TeN = length(TeLabel);
    for i = 1:1000
        t = (max(TrPred)-min(TrPred)) * (i-1)/1000 + min(TrPred);
        thr(i) = t;
     
        TrAcc(i) = (sum(TrLabel(TrPred<t)==0) + sum(TrLabel(TrPred>=t)==1)) / TrN;
        TeAcc(i) = (sum(TeLabel(TePred<t)==0) + sum(TeLabel(TePred>=t)==1)) / TeN;
    end
    plot(thr,TrAcc,'.- ',thr,TeAcc,'^-');legend('tr','te');

end

function calacc_classification(Test_ClassLabel,y_test,length)

% Record the successful prediction number
k1=0;
k2=0;

% Identify the successful predictions of input
for i = 1:length
    if y_test(i)==0
        if Test_ClassLabel(i) == y_test(i)
            k1=k1+1;
        end
    end
    if y_test(i)==1
        if Test_ClassLabel(i) == y_test(i)
            k2=k2+1;
        end
    end
end

% Calculate the accuracy and Display the accuracy
accuracy = (k1+k2)/length;
Z=['Accuracy ',num2str(accuracy)];
X=['there are ',num2str(k1), ' successful predictions of 0 in ',num2str(length),' input cases'];
Y=['there are ',num2str(k2), ' successful predictions of 1 in ',num2str(length),' input cases'];
disp(Z);
disp(X);
disp(Y);
end