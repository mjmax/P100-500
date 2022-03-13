%/////////////////////////////////////////////////
% Project - 1 (simulation)
% Auther: Janaka (856518198)
% Date  : 11/18/2021
% Question: Q1 Part (c)
%/////////////////////////////////////////////////

%////////////////////////////////////////////////////////////////////
%        Neural Networks
%////////////////////////////////////////////////////////////////////
clear all;              % clear workspace
clc;                    % clear console
load dataset1.mat;      % load dataset1 to workspace

L = 4;                  % maximum Layers
U = 3;                  % maximum units per layer
N = length(Y);          % number of examples to train
%w = randi([1 100], U, (U + 1), (L - 1)); % randomly initialized weight matrix
w = rand(U, (U + 1), (L - 1)); % randomly initialized weight matrix
w(:,4,1) = 0;           % eleminate the unwanted w values
w(2:3,:,3) = 0;
ac = @(z)(1/(1+exp(-z)));% Activation sigmoid function
dac = @(z)ac(z).*(1-ac(z));% Derivative of activation function

Z = zeros(U,L);         % weighted values (Z value)
a = [ones(1 , L) ; zeros(U,L)];       % activation values matrix (first colunm should be 1s)
delta = zeros(U,L);     % delata metrix (error matrix)
dJw = zeros(U,(U+1),L-1);   %partial derivative matrix
Jw = 0;                 % cost value
Jw_all = 0;             % cost vale over all iteration;
gradJw = 1;             % gradient of the cost function (randomy initialize)
Y_predict = zeros(N,1); % Predicted output through newral network;
epsilon = 0.01;         % gradient discent convergence
alpha = 4e-2;           % convergence rate
iter = 0;               % number of iteration of gradient discend algorithm

while(norm(gradJw) > epsilon)
    %--------------start of newral network----------------------%
   for n = 1 : N - 50
        a(: , 1) = [flip(X(n,:)) zeros((U+1) - length(X(n,:)))']';   % initiate input layer
        y = [1 Y(n) zeros(1,(U+1 - length(Y(n))))]';                 % initialize output vector (only one value of the output vector is usable. this is for consistency of the notation)
        
        %------------Forward Propergation-----------------%
        for l = 1 : L - 1                                           
            for u = 1 : U
                Z(u, l+1) = w(u,:,l)*a(:,l);        % compute Z value relative to the node
                a(u+1,l+1) = ac(Z(u,l+1));          % activate the node
            end
        end
        a(3:L,L) = 0;                               % eleminate unwanted uotput (only one output needed)

        %----------Backword Propergation------------------%
        % step 1: calculate error matrix
        for l = L : -1 : 2
            for u = 1 : U
                switch l                            % calculate error matrix 
                    case L
                        delta(u,l) = a(u+1,l) - y(u+1);
                    otherwise
                        deltatemp = 0;
                        for m = 1 : U
                            deltatemp = deltatemp + w(m,u+1,l)*delta(m,l+1);
                        end
                        delta(u,l) = deltatemp*dac(Z(u,l));
                end
            end
        end

        %step 2: calculate partial derivative matrix (3D)
        for l = 1 : L - 1
            for u = 1 : U
                for m = 1 : U + 1
                    dJw(u,m,l) = dJw(u,m,l) + delta(u,l+1)*a(m,l);
                end
            end
        end

        if(Y(n)==0)
            pc1 = 0;
            pc2 = (1 - Y(n))*log(1 - a(2,4));
        else
            pc1 = Y(n)*log(a(2,4));
            pc2 = 0;
        end
        Jw = Jw + pc1 + pc2;
     end
     Jw = -Jw;
    %-----------------end of newral network--------------------%
    

    %----------------start of gradietn discent------------------%
    clear gradJw;
    gradJw = 1;
    for l = 1 : L-1
        for u = 1 : U
            gradJw = [gradJw dJw(u,:,l)];
        end  
    end
    gradJw = gradJw(2 : end);
    w = w - alpha.*dJw;

    dJw = zeros(U,(U+1),L-1);   %partial derivative matrix
    %delta = zeros(U,L);     % delata metrix (error matrix)
    iter = iter + 1;
    fprintf('iteration number=%3d norm grad = %2.6f ...cost = %2.6f\n',iter,norm(gradJw),Jw);
    Jw_all = [Jw_all Jw];
    Jw = 0;
end


%-------------------------predicted output----------------%
for n = 1 : N
    a(: , 1) = [flip(X(n,:)) zeros((U+1) - length(X(n,:)))']';   % initiate input layer
    
    %------------Forward Propergation-----------------%
    for l = 1 : L - 1                                           
        for u = 1 : U
            Z(u, l+1) = w(u,:,l)*a(:,l);       % compute Z value relative to the node
            a(u+1,l+1) = ac(Z(u,l+1));         % activate the node
        end
    end
    a(3:L,L) = 0;
    Y_predict(n) = a(2,L);
end
Y_comp = [Y Y_predict];         % compare the predict output Vs Actual output

figure
Jw_all = Jw_all(2 : end);
plot(Jw_all);    % plot cost variation