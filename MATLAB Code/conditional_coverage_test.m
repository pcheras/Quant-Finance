function test_stat = cond_cov_test(hit_sequence) % Returns the test statistic of the Conditional Coverage test
%  COND_COV_TEST 
t = length(hit_sequence);
j = zeros(t,4); % Matrix to store the results
for i = 2:t
    j(i,1) = hit_sequence(i-1) == 0 & hit_sequence(i) == 0; % This entry is 1 if the condition is met, otherwise is 0 (acts as an indicator function) 
    j(i,2) = hit_sequence(i-1) == 0 & hit_sequence(i) == 1;
    j(i,3) = hit_sequence(i-1) == 1 & hit_sequence(i) == 0;
    j(i,4) = hit_sequence(i-1) == 1 & hit_sequence(i) == 1;  
end
V_00 = sum(j(:,1));
V_01 = sum(j(:,2));
V_10 = sum(j(:,3));
V_11 = sum(j(:,4));
p_00 = V_00 / (V_00 + V_01);
p_01 = V_01 / (V_00 + V_01);
p_10 = V_10 / (V_10 + V_11);
p_11 = V_11 / (V_10 + V_11);
p_hat = (V_01 + V_11) / (V_00 + V_01 + V_10 + V_11);
LL_U = log(1 - p_hat)*(V_00 + V_10) + log(p_hat) * (V_01 + V_11); % Unrestricted Log-likelihood value
LL_R = log(p_00)*(V_00) + log(p_01)*(V_01) + log(p_10)*(V_10) + log(p_11)*(V_11); % Restricted Log-likelihood value
test_stat = -2 * (LL_U - LL_R);
end