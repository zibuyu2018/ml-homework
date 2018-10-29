function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y);
% number of training examples
% You need to return the following variables correctly 
J = 0;
i=size(theta,1);
grad = zeros(size(theta));
%theta1=zeros(size(t[ones(m,1) X]heta));
%m2=size(X,2);
%theta1(2:m2-1,:)=theta(2:m2-1,:);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
[J1,grad1]=costFunction(theta,X,y);
J=J1+lambda*(theta(2:i,:)'*theta(2:i,:))/(2*m);
grad(1,:)=grad1(1,:);
grad(2:i,:)=grad1(2:i,:)+(lambda*theta(2:i,:))/m;

% ==============================*===============================

end
