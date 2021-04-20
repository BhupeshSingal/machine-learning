function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
delta=X*theta; % X = 47 x 3 and theta = 3 X 1 then delta = 47 X 1
delta=delta-y;  % y= 47 X 1 and delta = 47 X 1 then delta = 47 X 1
delta=X'*delta; % X'= 3 X 47 and delta = 47 X 1 then delta = 3 X 1
delta=alpha* ((1/m) * delta);  % alpha is a number , then delta = 3 X 1
theta=theta - delta; % theta = 3 X 1 then updated theta = 3 X 1
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    %J_history(iter)
end

end
