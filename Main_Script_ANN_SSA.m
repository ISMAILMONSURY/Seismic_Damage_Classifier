%##########################################################################
%##########################################################################


% Load and preprocess data#################################################
data = readtable('duzce_cleaned_data.csv'); % Load the dataset
inputs = data{:, 1:end-1}'; % Features (transpose to match MATLAB format)
targets = data{:, end}; % Labels
uniqueClasses = unique(targets);


% Convert categorical targets to numeric indices
numericTargets = double(categorical(targets)); % Convert categorical to numeric indices


% One-hot encode targets for classification################################
numClasses = numel(unique(numericTargets)); % Determine the number of classes
oneHotTargets = full(ind2vec(numericTargets', numClasses)); % One-hot encode the numeric indices


% Split data into training, validation, and test sets######################
trainRatio = 0.7;
valRatio = 0.15;
testRatio = 0.15;
[trainInd, valInd, testInd] = dividerand(size(inputs, 2), trainRatio, valRatio, testRatio);


trainInputs = inputs(:, trainInd);
valInputs = inputs(:, valInd);
testInputs = inputs(:, testInd);


trainTargets = oneHotTargets(:, trainInd);
valTargets = oneHotTargets(:, valInd);
testTargets = oneHotTargets(:, testInd);


% Initialize and configure the neural network##############################
numNeurons = 10; % Number of neurons in the hidden layer
net = patternnet(numNeurons);

% Explicitly configure the network for the input and target sizes
net = configure(net, trainInputs, trainTargets);


% Debugging: Validate network configuration###############################
disp(['Configured network input size: ', num2str(net.inputs{1}.size)]);
disp(['Configured network output size: ', num2str(size(trainTargets, 1))]);


% Configure the network for classification#################################
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;
net.performFcn = 'crossentropy'; % Cross-entropy loss for classification


% ======= Evaluate Before Optimization =======#############################
disp('Evaluating Before Optimization');
netBefore = train(net, trainInputs, trainTargets); % Train using default training
predictionsBefore = netBefore(testInputs);
[~, predictedClassesBefore] = max(predictionsBefore, [], 1);
[~, actualClasses] = max(testTargets, [], 1);


% Confusion matrix and classification report before optimization###########
confMatBefore = confusionmat(actualClasses, predictedClassesBefore);
precisionBefore = diag(confMatBefore) ./ sum(confMatBefore, 2); % Precision
recallBefore = diag(confMatBefore) ./ sum(confMatBefore, 1)';   % Recall
f1ScoreBefore = 2 * (precisionBefore .* recallBefore) ./ (precisionBefore + recallBefore);
accuracyBefore = sum(diag(confMatBefore)) / sum(confMatBefore(:)); % Accuracy


% Table for metrics before optimization####################################
classMetricsBefore = table(uniqueClasses, precisionBefore, recallBefore, f1ScoreBefore, ...
    'VariableNames', {'Class', 'Precision', 'Recall', 'F1Score'});
disp('Confusion Matrix Before Optimization:');
disp(confMatBefore);
disp('Classification Report Before Optimization:');
disp(classMetricsBefore);
disp(['Accuracy Before Optimization: ', num2str(accuracyBefore * 100), '%']);


% ======= Evaluate After Optimization =======##############################
disp('Evaluating After Optimization');

% Calculate total weights and biases
numInputs = size(trainInputs, 1);
numWeightsAndBiases = (numInputs * numNeurons) + ... % Weights from input to hidden
                      numNeurons + ...              % Biases in hidden layer
                      (numNeurons * numClasses) + ... % Weights from hidden to output
                      numClasses;                   % Biases in output layer

% Define objective function for SSA optimization
h = @(bestX) Crosentropy(bestX, net, trainInputs, trainTargets); % Objective function

% Perform SSA optimization
[bestX, ~] = Sparrow(numWeightsAndBiases, h);

% Validate size of bestX
if numel(bestX) ~= numWeightsAndBiases
    error('Length of bestX (%d) does not match the expected number of weights and biases (%d).', ...
        numel(bestX), numWeightsAndBiases);
end

% Apply optimized weights and biases to the network
netAfter = setwb(net, bestX');

% Evaluate test set
predictionsAfter = netAfter(testInputs);
[~, predictedClassesAfter] = max(predictionsAfter, [], 1);


% Confusion matrix and classification report after optimization############
confMatAfter = confusionmat(actualClasses, predictedClassesAfter);
precisionAfter = diag(confMatAfter) ./ sum(confMatAfter, 2); % Precision
recallAfter = diag(confMatAfter) ./ sum(confMatAfter, 1)';   % Recall
f1ScoreAfter = 2 * (precisionAfter .* recallAfter) ./ (precisionAfter + recallAfter);
accuracyAfter = sum(diag(confMatAfter)) / sum(confMatAfter(:)); % Accuracy


% Table for metrics after optimization#####################################
classMetricsAfter = table(uniqueClasses, precisionAfter, recallAfter, f1ScoreAfter, ...
    'VariableNames', {'Class', 'Precision', 'Recall', 'F1Score'});
disp('Confusion Matrix After Optimization:');
disp(confMatAfter);
disp('Classification Report After Optimization:');
disp(classMetricsAfter);
disp(['Accuracy After Optimization: ', num2str(accuracyAfter * 100), '%']);


% ======= Comparison =======###############################################
disp('Performance Comparison Before and After Optimization:');
comparisonTable = table(uniqueClasses, precisionBefore, precisionAfter, recallBefore, recallAfter, ...
    f1ScoreBefore, f1ScoreAfter, repmat(accuracyBefore, numel(uniqueClasses), 1), ...
    repmat(accuracyAfter, numel(uniqueClasses), 1), 'VariableNames', ...
    {'Class', 'Precision_Before', 'Precision_After', 'Recall_Before', ...
    'Recall_After', 'F1Score_Before', 'F1Score_After', 'Accuracy_Before', 'Accuracy_After'});
disp(comparisonTable);