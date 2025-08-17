function [costas_seq] = costas_sequence(n)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% Costas Sequence Generation
    % Length of the sequence (you can adjust this as needed)
    costas_seq = zeros(1, n);
    
    % Generate the sequence
    for k = 1:n
        costas_seq(k) = mod(k*(k+1)/2, n) + 1;
    end
    
    % Display the sequence
    % disp('Costas Sequence:');
    % disp(costas_seq);
end