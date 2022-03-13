%/////////////////////////////////////////////////
% Project - ECE572 Neural Network (Group Project)
% Auther: Group
% Date  : 03/13/2022
%/////////////////////////////////////////////////

%////////////////////////////////////////////////////////////////////
%        Neural Networks Group Project (Sub Net XX)
%////////////////////////////////////////////////////////////////////
clear all;              % clear workspace
clc;                    % clear console

% Number of layers and neuron count 
Li = 1;                 % Number of input layers
Lh = 3;                 % Number of hidden layers
Lo = 1;                 % Number of output layers
L = Li + Lh + Lo;       % Number of layers (with input and output layer)
Uh = 8 + 1;             % Number of neurons per hidden layer (with bias)
Ui = 626;               % Number of inputs (with bias (x = 1))
Uo = 1;                 % Number of outputs

% Weights matrices
Wi = rand((Uh - 1), Ui);                % Input to first hidden layer mapping weight matrix
Wh = rand(Uh, (Uh - 1), (Lh - 1));      % Hidden layer to hidden layer mapping weight matrix
Wo = rand(Uh, Uo);                      % Hidden layer to output layer mapping weight matrix

% Decision boundary (Z) matrices
Zh = zeros((Uh - 1), Lh);               % Decision boundaries matrix for the hidden layers(bias terms do not have boundary)
Zo = zeros(Uo, Lo);                     % Decision boundary for the output layers (scaler value in this case)

% Activation matrices
Ah = [zeros((Uh - 1), Lh); ones(1, Lh)];% Activation matrix for the hidden layers (bias term has an activation of 1)
Ao = zeros(Uo, Lo);                     % Activation for the output layers (scaler value in this case)

% Back propergation error matrices
Dh = zeros((Uh - 1), Lh);               % Back Propergation error matrices for hidden layers(bias terms do not have back propergation errors)
Do = zeros(Uo, Lo);                     % Back Propergation error for the output layers (scaler value in this case)

% Partial derivative matrices
dWi = zeros((Uh - 1), Ui);               % Input to first hidden layer partial derivative matrix
dWh = zeros(Uh, (Uh - 1), (Lh - 1));     % Hidden layer to hidden layer partial derivative matrix
dWo = zeros(Uh, Uo);                     % Hidden layer to output layer partial derivative matrix
