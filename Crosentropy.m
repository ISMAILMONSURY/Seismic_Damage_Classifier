%##########################################################################
%##########################################################################

function crossEntropyLoss = Crosentropy(wb, net, input, target)
    % Error for classification with cross-entropy loss

    % wb: weights and biases from SSA
    % net: neural network object
    % input: input data
    % target: one-hot encoded target labels

    % Update the network weights and biases
    net = setwb(net, wb');

    
    % Check input size compatibility
    if size(input, 1) ~= net.inputs{1}.size
        error('Input size (%d) does not match the network input size (%d).', ...
              size(input, 1), net.inputs{1}.size);
    end

    
    % Get network predictions
    predictions = net(input);

    % Add numerical stability to avoid log(0)
    epsilon = 1e-12;
    predictions = predictions + epsilon;

    % Compute cross-entropy loss
    crossEntropyLoss = -mean(sum(target .* log(predictions), 1));
end