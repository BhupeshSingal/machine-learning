function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
        total_sum1=0;
    total_sum2=0;
    sum=0;
    for i=1:m
        sum=0;
        sum=theta(1)*X(i,1)+theta(2)*X(i,2);
        sum=sum-y(i,1);
        total_sum1=total_sum1+(sum*X(i,1));
        total_sum2=total_sum2+(sum*X(i,2));
    end
    theta(1)=theta(1)- ((alpha/m)*total_sum1);
    theta(2)=theta(2)- ((alpha/m)*total_sum2);
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
%     if iter > 1
%     if J_history(iter-1) > J_history(iter) 
%         fprintf('J(theta) is decreasing');
%     else
%         fprintf('J(theta) is increasing');
%     end
%     end
% end

end
