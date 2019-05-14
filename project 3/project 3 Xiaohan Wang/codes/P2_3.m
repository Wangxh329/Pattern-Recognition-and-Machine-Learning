% data
load('P2_2_gov.mat', 'cor');
gov_cor = cor;
load('P2_2_sen.mat', 'cor');
sen_cor = cor;
clear cor;

max_cor = max(sen_cor);
min_cor = min(sen_cor);
sen_cor = (sen_cor - min_cor) / (max_cor - min_cor);

max_cor = max(gov_cor);
min_cor = min(gov_cor);
gov_cor = (gov_cor - min_cor) / (max_cor - min_cor);

save('P2_3.mat', 'gov_cor', 'sen_cor')
