---
title: 'TUTORIAL: Fitting Multinomial Logistic Regression Models to Behavioural Data Using Dummy Coding in MATLAB'
date: 2026-09-22
permalink: /_posts/2026-09-22-Multinomial_Logistic_Regression
tags:
  - Tutorials
  - Behavioural modelling
  - Statistics
---

(Version française à venir)

This post introduces multinomial logistic regression through a hands-on MATLAB Live Script. The tutorial shows how to model behavioural choices as a function of choices and rewards received on previous trials, using explicit dummy coding to construct the regression design matrix.

[📓 Download the MATLAB Live Script](/files/MultiLogReg_eng.mlx)

# What is Multinomial Logistic Regression?

In statistical analysis, **regression** models are a way of estimating the relationship between an outcome of interest, the so\-called **dependent variable** (or response), and one or more predictors, also known as **independent variables**. This requires observations $$(x_i ,y_i)$$, $$i=1,...,n$$, of a response $$y_i$$ paired with predictor values $$x_i$$, to which a model is then fitted in order to predict $$Y$$ for different values of $$X$$ (because these are random variables, I use capitalised letters to distinguish them from the observations which are samples of these random variables). The most famous example is probably a **simple linear regression** in which the dependent and independent variables are continuous quantities, such as when trying to predict weight from height. The model is based on the assumption that, given the predictor, the response equals a linear function of it plus a random error:

 $$ y_i =\beta_0 +\beta_1 x_i +\varepsilon_i $$ 

where $$\varepsilon_i$$ is a random error term with zero mean. With this assumption,  **the linear regression model describes the expected value of $$Y$$ conditioned on each observation $$x_{i}$$** as a simple trend line with an intercept $$\beta_0$$ and a slope $$\beta_1$$:

 $$ E(Y|x_i) = \beta_0 +\beta_1 x_i $$ 

The central question, which applies to any regression model, is to find the parameter values that fit the data best according to some criterion: for linear regression the sum of squared errors between observations $$y_i$$ and expected values $$E(Y \mid x_i)$$ is minimised. A first extension is to admit multiple predictors $$x_1, ... ,x_p$$, collected in the vector $$\mathbf{x}=(x_1, ...,x_p)^T$$, in which case the model becomes:

 $$ E(Y|\mathbf{x}) =\beta_0 +\beta_1 x_1 +\beta_2 x_2 + ... +\beta_p x_p =\beta_0 +\sum_{j=1}^p \beta_j x_j $$ 

Sometimes the variable we want to predict is not a continuous quantity like height but a category, i.e. a **nominal variable**. In the simplest case there are only two categories, for instance `smoker` vs. `non-smoker` or `yes` vs. `no`. A linear model cannot be applied directly to such outcomes, since they are not numbers, and if we coded them as 0 and 1 a linear function of $$\mathbf{x}$$ would not keep the predicted probability between 0 and 1. The trick at the heart of **logistic regression** is to model the probability $$\pi(\mathbf{x})=P(Y=\textrm{yes} \mid \mathbf{x})$$ through its **log-odds** (or logit), the logarithm of the **odds** $$\pi /(1-\pi)$$, which can take any real value, and to make **this** quantity a linear function of the predictors:

 $$ \log \frac{\pi (\mathbf{x})}{1-\pi (\mathbf{x})}=\beta_0 +\sum_{j=1}^p \beta_j x_j $$ 

Technically, without going into further detail, the criterion used to optimise the $\beta$ coefficients is no longer the sum of squared errors, which must be minimised in linear regression, but the likelihood of observations which is maximised. Because there are just two possible outcomes, increasing the value of one of the coefficients increases the log-odds of the reference outcomes and so the probability of that outcome.

Finally, when there are more than two categories, a **multinomial logistic regression** is used . Let the outcome $$Y$$ take values in $$\lbrace 1,\ldots,K\rbrace$$ and write $$\pi_k(\mathbf{x})=P(Y=k \mid \mathbf{x})$$, so that $$\pi_1 + \cdots +\pi_K =1$$. One category, here $$K$$, is chosen as the reference, and the log-odds of each of the other categories relative to it are modelled, which gives a system of $$K-1$$ equations. In the case I will be working on here, we look at the probabilities of choosing levers in a 3-armed bandit task. The outcome on a given trial is nominal, e.g. the agent chose lever 1, and since we have $$K=3$$ possible choices we want to find the parameters of a system of 2 equations which, taking lever 3 as reference, is:

 $$ \left\lbrace \begin{array}{c} \log \frac{\pi_1 (\mathbf{x})}{\pi_3 (\mathbf{x})}=\beta_{1,0} +\sum_{j=1}^p \beta_{1,j} x_j =\eta_1 \newline \log \frac{\pi_2 (\mathbf{x})}{\pi_3 (\mathbf{x})}=\beta_{2,0} +\sum_{j=1}^p \beta_{2,j} x_j =\eta_2  \end{array}\right. $$ 

Here $$\beta_{k,j}$$ is the coefficient of predictor $$j$$ in the equation for category $$k$$ ($$j=0$$ being the intercept), and $$\eta_k$$ is the linear predictor of equation $$k$$. Note that the predictors $$x_j$$ are the same in both equations: **predictors are shared between all equations, it is how the coefficients of each equation transform them that produces different responses**. Once the parameters are fitted, you can work your way back to the estimated probabilities based on two facts. First, the definition of the log-odds gives $$\pi_k =\pi_3 e^{\eta_k }$$ for $$k=1,2$$. Second, the probabilities sum to one, so that:

 $$ \pi_3 \left(1+e^{\eta_1 } +e^{\eta_2 } \right)=\pi_1 +\pi_2 +\pi_3 =1 $$ 

Solving for $$\pi_3$$ and substituting back, the estimated probability of each action is:

 $$ \left\lbrace \begin{array}{c} \pi_1 =\frac{e^{\eta_1 } }{1+e^{\eta_1 } +e^{\eta_2 } }\newline \pi_2 =\frac{e^{\eta_2 } }{1+e^{\eta_1 } +e^{\eta_2 } }\newline \pi_3 =\frac{1}{1+e^{\eta_1 } +e^{\eta_2 } } \end{array}\right. $$ 

It's worth noticing here how complex these probabilities are, as they all depend on both $$\eta$$ predictors; if $$\eta_1$$ increases, although the log-odds of choosing lever 1 relative to lever 3 also increase, $$\pi_1$$ itself may in fact decrease depending on how $$\eta_2$$ is behaving.

# Dummy Coding

I've explained how to deal with nominal dependent variables, now comes the turn of nominal independent variables. Here we can't use the previous trick of replacing variables with log-odds since the observations $$\mathbf{x}$$ which are indeed samples of random variable $$X$$ are nonetheless fixed once observed. **Dummy coding** solves this problem by replacing the nominal predictors with $$K-1$$ variables, where $$K$$ is the total number of categories. Each **dummy variable** equals 1 if the data point belongs to a given category and 0 otherwise. The omitted category is the reference (or default), identified by all dummy variables being 0. Other coding schemes exist, for instance effect coding with values 1, 0 and -1, but they change the interpretation of the coefficients and I will not use them. As an example, if we want to predict height based on sex, we introduce a dummy variable $$d$$ equal to 1 if the subject is female and 0 if male and run a simple linear regression as before:

 $$ y_i =\beta_0 +\beta_1 d_i +\varepsilon_{i\;} $$ 

The meaning of $$\beta_0$$ and $$\beta_1$$ follows directly from the conditional expectation of $$Y$$. For a male, $$E(Y \mid d=0)=\beta_0$$ is the expected height of a male (estimated by the sample mean height of the males); for a female, $$E(Y \mid d=1) =\beta_0 +\beta_1$$ which means that $$\beta_1$$ is the difference in expected height between females and males. With three categories we would use two dummy variables, and each coefficient would be the difference in expected outcome between that category and the reference category.

# A Small Simulated Dataset

We are now ready to tackle the main objective of this tutorial, which is to fit a multinomial logistic regression to data collected in a **multi\-armed bandit task**. These tasks, which are a classic **reinforcement learning** problem commonly used in neuroscience, are made of discrete trials in which subjects have to choose one action among several in the hope of obtaining a reward. Past rewards and choices can be used as predictors in a logistic regression model to determine the impact of these past events on current choices, as in the study of **[Lau and Glimcher (2005)](https://doi.org/10.1901/jeab.2005.110-04)** which used logistic regression to study the choices of monkeys in a two-alternative task and found that the impact of past rewards decayed over time. This study inspired me to test a more complex multinomial logistic regression when analysing a three-armed bandit task (**[Cinotti et al. (2024)](https://doi.org/10.1111/ejn.16449)**), which proved to be a more difficult challenge than initially expected so that I eventually resorted to simple logistic regressions fitted separately to each action. I have since had the opportunity to revisit and master this technique which prompted me to write this tutorial in the hope it might help others faced with similar difficulties.

To start with, we shall need data, which will be a simple synthetic collection of 900 trials with recorded choices and rewards. This virtual experiment will consist of blocks of 100 trials during which one of the levers will be rewarded with a 60% probability if selected, while the two others will be rewarded with a probability of 20% each. To generate plausibly realistic choices, we will use a Q\-learning model with a learning rate of 0.1, which also serves as the forgetting rate of the values of non\-selected levers, paired with a softmax action selection with an inverse temperature of 5. If you do not understand what this all means, all you need to know is that this is a way of generating interesting data.

```matlab
%% Simulate some behavioural data

rng(1);  % make the example reproducible

trials_per_block = 100;
schedule = repmat([1, 2, 3], 1, 3); % identities of the best lever in each block of trials
n_blocks = length(schedule);
n_trials = n_blocks * trials_per_block;

data = struct('choices', nan(n_trials, 1), 'rewards', nan(n_trials,1)); % one entry per trial

% initialise the Q-learner
Q = zeros(1, 3);
learningRate = 0.1;
inverseTemperature = 5;

for trial = 1 : n_trials
    
    % given trial number, get the current best lever
    block = ceil(trial / trials_per_block);
    best = schedule(block);
    
    % generate a random choice based on current Q-values
    probs = exp(inverseTemperature * Q) ./ sum(exp(inverseTemperature * Q));
    choice = randsample(1:3, 1, true, probs);
    
    % generate a stochastic reward based on the choice
    reward = rand < (choice == best) * 0.6 + (choice ~= best) * 0.2;

    % update Q-values for next step
    Q = (1 - learningRate) * Q; % forgetting step that applies to all levers
    Q(choice) = Q(choice) + learningRate * reward; % learning that applies only to the selected lever    
   
    % store trial observations
    data.choices(trial) = choice;
    data.rewards(trial) = reward;
end

```

To get an idea of what this data looks like, here is a quick plot of the rates of selection of each lever using 20-trial moving averages. You should see distinct periods in which the three levers dominate in turn for blocks of roughly 100 trials, illustrating how the Q-learning algorithm successfully keeps track of the reward schedule. Choices are nonetheless noisy, and if you have kept the same random seed, you'll notice that in the fourth block (trials 300-400), the selection rate of lever 1, which was in fact the best lever for that block, was quite low.

```matlab
%% running averages of lever selection for quick visualisation

figure()
windowSize = 20;
runningChoiceRate = movmean(data.choices == (1:3), windowSize, 1);

plot(runningChoiceRate, 'LineWidth', 1.5);
xlabel('Trial');
ylabel('Choice proportion');
legend('Lever 1', 'Lever 2', 'Lever 3', 'Location', 'best');
```

![Selection rates](/images/multinomial_regression/figure_0.png)

# Constructing the Design Matrix

We now ask whether previous choices and rewards can predict the choice on the current trial. Let $$c_t \in \lbrace 1,2,3\rbrace$$ be the lever chosen and $$r_t \in \lbrace 0,1\rbrace$$ the reward obtained on trial $$t$$. At this point, the design of the regression is really up to you. In my case, I wanted to use the past $$L=10$$ trials (the **horizon**) to predict $$c_t$$, with five predictors for each lag $$\ell =1,\ldots,L$$: indicators for having chosen lever 1, having chosen lever 2, having been rewarded on lever 1, having been rewarded on lever 2, and having been rewarded on lever 3, so that there are $$p=5L=50$$ predictors in total, and the model to be fitted is the system of equations introduced above, with $$\mathbf{x_t}={\left(x_{t,1} ,\ldots,x_{t,p} \right)}^T$$ as predictors:

 $$ \log \frac{P\left.\left(c_t =k\right|{\mathbf{x}}_t \right)}{P\left.\left(c_t =3\right|{\mathbf{x}}_t \right)}=\beta_{k,0} +\sum_{j=1}^p \beta_{k,j} x_{t,j} ,~~k=1,2 $$ 

A few things to note about this design. There is no predictor for having chosen lever 3, which means lever 3 is the reference lever, as above. The last three predictors are choice-reward combinations, `rewarded on lever k` meaning that lever $$k$$ was chosen and rewarded: predictor 3 is the effect of being rewarded on lever 1 in addition to the effect of selecting lever 1, which is predictor 1. Alternative designs might want to separate these effects differently, and this choice matters when trying to interpret the final results. The **design matrix $$X$$** is the collection of the predictors for every trial; it has one row per observation (trial) and one column per predictor. The trial-by-trial choices $$c_t$$ are similarly collected in the **response vector $$y$$**.

```matlab
horizon = 10;
n_levers = 3;
ref_lever = 3; % reference lever (the dummy coding below assumes lever 3)
non_ref_levers = setdiff(1:n_levers, ref_lever);

n_choice_cols = numel(non_ref_levers);
n_reward_cols = n_levers;

cols_per_lag = n_choice_cols + n_reward_cols;
n_pred = cols_per_lag * horizon;
```

We can now construct $$X$$ and $$y$$.

```matlab
% We lose the first 'horizon' trials because there is not enough previous history to construct the predictors.
n_rows = n_trials - horizon;

X = zeros(n_rows, n_pred);
y = zeros(n_rows, 1);

row = 1;

for t = horizon + 1 : n_trials % we go through trials filling in the rows of X and y
    
    % Current choice is the response variable
    y(row) = data.choices(t);
    
    % -------------------------------------------------------------
    % Construct predictors from previous trials
    % -------------------------------------------------------------
    
    for lag = 1 : horizon % we repeat these same steps for each lag trial in turn
        
        % First column belonging to this lag
        lag_offset = (lag-1) * cols_per_lag;
        
        past_choice = data.choices(t-lag);
        past_reward = data.rewards(t-lag);
        
        % ---------------------------------------------------------
        % Previous choice
        % ---------------------------------------------------------
        
        % Lever 3 is the reference category, so it does not get
        % its own dummy variable.
        if past_choice == 1
            X(row, lag_offset + 1) = 1;
        elseif past_choice == 2
            X(row, lag_offset + 2) = 1;
        end
        
        % ---------------------------------------------------------
        % Previous rewarded choice
        % ---------------------------------------------------------
        
        % If the previous choice was rewarded, activate the
        % indicator corresponding to the chosen lever.
        if past_reward == 1            
            reward_col = lag_offset + ...
                n_choice_cols + past_choice;            
            X(row, reward_col) = 1;            
        end        
    end    
    row = row + 1;
end

```

To get a better sense of what $$X$$ looks like, it's worth giving a look at the first few rows of the original data and $$X$$ side-by-side:

```matlab
% Original data: trial number, choice and reward in one table
n_show = 15;
dataTable = table((1:n_show)', data.choices(1:n_show), data.rewards(1:n_show), ...
    'VariableNames', {'Trial', 'Choice', 'Reward'});
disp(dataTable)
```

```matlabTextOutput
    Trial    Choice    Reward
    _____    ______    ______

      1        2         0   
      2        1         1   
      3        1         1   
      4        1         1   
      5        1         1   
      6        1         0   
      7        1         0   
      8        1         0   
      9        1         1   
     10        1         1   
     11        2         0   
     12        1         0   
     13        3         0   
     14        1         1   
     15        1         0   
```

```matlab

% First rows of X, labelled by the trial they predict
n_rows_shown = 5;
lags_shown   = 2;   % lags 1 and 2 

% Readable column names, e.g. Chose1_L1, Chose2_L1, Rew1_L1, ..., Rew3_L2
baseNames = [compose("Chose%d", non_ref_levers), compose("Rew%d", 1:n_levers)];
predNames = strings(1, 0);
for lag = 1 : lags_shown
    predNames = [predNames, baseNames + "_L" + lag];
end

XTable = array2table(X(1:n_rows_shown, 1:numel(predNames)), ...
    'VariableNames', cellstr(predNames));

% Add the trial each row predicts
XTable = addvars(XTable, (horizon + 1 : horizon + n_rows_shown)', ...
    'Before', 1, 'NewVariableNames', 'Trial');
disp(XTable)
```

```matlabTextOutput
    Trial    Chose1_L1    Chose2_L1    Rew1_L1    Rew2_L1    Rew3_L1    Chose1_L2    Chose2_L2    Rew1_L2    Rew2_L2    Rew3_L2
    _____    _________    _________    _______    _______    _______    _________    _________    _______    _______    _______

     11          1            0           1          0          0           1            0           1          0          0   
     12          0            1           0          0          0           1            0           1          0          0   
     13          1            0           0          0          0           0            1           0          0          0   
     14          0            0           0          0          0           1            0           0          0          0   
     15          1            0           1          0          0           0            0           0          0          0   
```

The first row of $$X$$ corresponds to trial 11, and the first five columns describe what happened in trial 10 in which, provided the random seed is unchanged, lever 1 was selected and rewarded so that $$X\left(1,1\right)=1$$ and $$X\left(1,3\right)=1$$. The second row of $$X$$ corresponds to trial 12, so that the events of trial 10 are shifted to columns 6-10, while columns 1-5 describe trial 11 where the agent chose lever 2 and was not rewarded.

# Fitting the Regression Model

We have now built our design matrix $$X$$, which contains the agent's behavioural history, and $$y$$, which contains the choice made on the current trial. For `horizon = 10`, there are 5 predictors per lag × 10 lags = 50 predictors, so $$X$$ has 50 columns. The first five columns describe the immediately preceding trial (`lag 1`), the next five describe the trial before that (`lag 2`), and so on. To fit the multinomial regression model by maximum likelihood we use the built-in `mnrfit` function. Recent MATLAB releases also provide a more sophisticated `fitmnr` function, but I chose `mnrfit` because constructing the dummy\-coded design matrix $$X$$ makes the predictor encoding more transparent.

```matlab
% Fit the multinomial logistic regression
[B, ~, ~] = mnrfit(X, y, 'Model', 'nominal');

```

The output $$B$$ contains the fitted coefficients $$\beta_{k,j}$$ of the model. It has two columns, the first for the log-odds of choosing lever 1 rather than lever 3 ( $$k=1$$ ), and the second for the log-odds of lever 2 over lever 3 ( $$k=2$$ ). It contains 51 rows: the first row is for the intercepts $$\beta_{1,0}$$ and $$\beta_{2,0}$$, and the remaining 50 for the predictor coefficients, so that $$B\left(j+1,k\right)=\beta_{k,j}$$. To check the model is working, you can calculate and plot the predicted probabilities of each action using the `mnrval` function, the fitted $$B$$, and the original data $$X$$ (instead of recycling $$X$$, the recommended method is usually to hold out some of the data from the fitting and to use that as a test dataset).

```matlab
% Evaluate fitted choice probabilities on the observed predictors

P = mnrval(B, X);

figure()
plot(P, 'LineWidth', 1.5);
xlabel('Trial');
ylabel('Predicted choice probability');
legend('Lever 1', 'Lever 2', 'Lever 3', 'Location', 'best');
grid on;
```

![Predicted choice probabilities](/images/multinomial_regression/figure_1.png)

Compared to the running averages we plotted before, the fitted probabilities reproduce broad patterns of the original data. We see dominance of the different levers alternating between blocks, and, unless you have changed the random seed, you should see that the fourth block, in which the agent had trouble picking the correct lever 1, is also ambiguous from the fitted model's point of view, thus matching an idiosyncratic feature of the original data.

# A Misguided Attempt to Interpret the Fitted Model

Following the example of **[Lau and Glimcher (2005)](https://doi.org/10.1901/jeab.2005.110-04)**, you might want to simply plot some of the fitted coefficients to see how past predictors affect choices (Figure 6 of that original publication). An obvious strategy could be to look at the separate impacts of being rewarded on a lever and of selecting a lever without necessarily being rewarded on the log-odds of that same lever, as measurements of **the effects of reinforcement** and **choice persistence** on behaviour respectively. In the case of lever 1, the coefficients for the predictor `lever 1 chosen` are found at rows 2, 7, 12, etc. of $$B$$ and the coefficients for the predictor `lever 1 rewarded` at rows 4, 9, 14, etc. For lever 2, we are interested in the coefficients of `lever 2 chosen` (rows 3, 8, 13, ...) and `lever 2 rewarded` (rows 5, 10, 15, ...).

```matlab
figure()

subplot(2,2,1)
hold on
plot(1 : horizon, B(2 : cols_per_lag : end,1))
title('Perseveration effect on lever1')
xlabel('Past trials')
ylabel('log odds')
set(gca,'XDir', 'reverse')
axis square

subplot(2,2,2)
hold on
plot(1 : horizon, B(4 : cols_per_lag : end,1))
title('Reinforcement effect on lever1')
xlabel('Past trials')
ylabel('log odds')
set(gca,'XDir', 'reverse')
axis square

subplot(2,2,3)
hold on
plot(1 : horizon, B(3 : cols_per_lag : end,2))
title('Perseveration effect on lever2')
xlabel('Past trials')
ylabel('log odds')
set(gca,'XDir', 'reverse')
axis square

subplot(2,2,4)
hold on
plot(1 : horizon, B(5 : cols_per_lag : end,2))
title('Reinforcement effect on lever2')
xlabel('Past trials')
ylabel('log odds')
set(gca,'XDir', 'reverse')
axis square
```

![Fitted coefficients](/images/multinomial_regression/figure_2.png)

While the other coefficients have little obvious trend, the coefficients associated with rewarded choices on lever 1 are positive and tend to decrease as we look further into the past, suggesting a positive and decreasing impact of past rewards on selecting lever 1. However, contrary to the original study of **Lau and Glimcher (2005)**, which relied on a simple logistic regression with just two possible outcomes, interpretation of this observation is more delicate. Each coefficient $$\beta_{k,j}$$ tells us how predictor $$j$$ affects the log-odds of lever $$k$$ versus lever 3, rather than how the choice probability $$\pi_k$$, the quantity we are really interested in, behaves. Additionally, because the predictors are shared between both log-odds equations, it's possible for a predictor to have an effect on an action probability by primarily affected the log-odds of another action. For instance, `being rewarded on lever 1` might increase the probability of selecting lever 1 by primarily decreasing the log-odds of lever 2 rather than directly increasing those of lever 1. This intricacy between the log-odds equations might explain why the other coefficients seem noisy at first glance. To know how the predictors affect choice probabilities, we must take into account the other alternatives, a topic for another time.

---

Cet article présente la régression logistique multinomiale à l'aide d'un Live Script MATLAB. Le tutoriel montre comment modéliser des choix comportementaux en fonction des choix et des récompenses obtenues lors des essais précédents, en utilisant un codage indicateur explicite pour construire la matrice de conception de la régression.

[📓 Télécharger le Live Script MATLAB](/files/MultiLogReg_fr.mlx)

# Qu'est-ce qu'une régression logistique multinomiale ?

En analyse statistique, les modèles de **régression** sont un moyen d'estimer la relation entre une variable d'intérêt, appelée **variable dépendante** (ou réponse), et une ou plusieurs **variables indépendantes** (ou prédicteurs). Cela nécessite des observations $$(x_i ,y_i)$$, $$i=1,...,n$$, d'une réponse $$y_i$$ associée à des valeurs de prédicteurs $$x_i$$, auxquelles un modèle est ensuite ajusté afin de prédire $$Y$$ pour différentes valeurs de $$X$$ (comme il s'agit de variables aléatoires, j'utilise ici des majuscules pour les distinguer des observations, qui sont des échantillons de ces variables aléatoires). L'exemple le plus célèbre est probablement une **régression linéaire simple** entre des variables dépendante et indépendante qui sont des quantités continues, par exemple lorsqu'on cherche à prédire le poids à partir de la taille. Le modèle repose sur l'hypothèse que, étant donné le prédicteur, la réponse est égale à une fonction linéaire de celui-ci plus une erreur aléatoire :

 $$ y_i =\beta_0 +\beta_1 x_i +\varepsilon_i $$ 
 
où $$\varepsilon_i$$ est un terme d'erreur aléatoire d'espérance nulle. Avec cette hypothèse, **le modèle de régression linéaire décrit l'espérance de $$Y$$ conditionnellement à chaque observation $$x_{i}$$** sous la forme d'une simple droite de tendance, d'ordonnée à l'origine $$\beta_0$$ et de pente $$\beta_1$$ :

 $$ E(Y|x_i) = \beta_0 +\beta_1 x_i $$ 

La question centrale, qui est la même pour tous les modèles de régression, est de trouver les valeurs des paramètres qui reflètent le mieux les données selon un certain critère : pour la régression linéaire, on minimise la somme des carrés des écarts entre les observations $$y_i$$ et les valeurs attendues $$E(Y \mid x_i)$$. Une première extension consiste à admettre plusieurs prédicteurs $$x_1, ... ,x_p$$, rassemblés dans le vecteur $$\mathbf{x}=(x_1, ...,x_p)^T$$, auquel cas le modèle devient :

 $$ E(Y|\mathbf{x}) =\beta_0 +\beta_1 x_1 +\beta_2 x_2 + ... +\beta_p x_p =\beta_0 +\sum_{j=1}^p \beta_j x_j $$ 

Parfois, la variable que l'on veut prédire n'est pas une quantité continue comme la taille mais une catégorie, c'est-à-dire une **variable nominale**. Dans le cas le plus simple, il n'y a que deux catégories, par exemple `fumeur` vs `non-fumeur` ou `oui` vs `non`. On ne peut pas appliquer directement un modèle linéaire à de tels résultats, car ce ne sont pas des nombres, et si on les codait par 0 et 1, une fonction linéaire de $$\mathbf{x}$$ ne garantirait pas que la probabilité prédite reste comprise entre 0 et 1. L'astuce au cœur de **la régression logistique** consiste à modéliser la probabilité $$\pi(\mathbf{x})=P(Y=\textrm{oui} \mid \mathbf{x})$$ à travers son ***log-odds*** (ou logit), le logarithme de la **cote** (*odds* en anglais, le rapport de deux probabilités) $$\pi /(1-\pi)$$, qui peut prendre n'importe quelle valeur réelle, et à faire de cette quantité une fonction linéaire des prédicteurs :

$$ \log \frac{\pi (\mathbf{x})}{1-\pi (\mathbf{x})}=\beta_0 +\sum_{j=1}^p \beta_j x_j $$ 

Sur le plan technique, sans entrer dans les détails, le critère utilisé pour optimiser les coefficients $\beta$ n'est plus la somme des carrés des écarts, que l'on minimise en régression linéaire, mais la vraisemblance des observations, que l'on maximise. Remqarquez comment, du fait qu'il n'y a que deux issues possibles, augmenter la valeur d'un des coefficients augmente la cote de l'issue de référence, et donc la probabilité de cette issue.

Enfin, lorsqu'il y a plus de deux catégories, on utilise une **régression logistique multinomiale**. Soient une variable dépendante $$Y$$ prenant ses valeurs dans $$\lbrace 1,\ldots,K\rbrace$$ et la probabilité conditionnelle $$\pi_k(\mathbf{x})=P(Y=k \mid \mathbf{x})$$, de sorte que $$\pi_1 + \cdots +\pi_K =1$$. Une catégorie, ici $$K$$, est choisie comme référence, et on modélise les log-odds de chacune des autres catégories relativement à celle-ci, ce qui donne un système de $$K-1$$ équations. Dans le cas que je vais traiter ici, on s'intéresse aux probabilités de choisir des leviers dans une tâche de bandit à 3 bras. L'issue d'un essai donné est nominale, par exemple `levier 1 choisi`, et comme il y a $$K=3$$ choix possibles, on cherche les paramètres d'un système de 2 équations qui, en prenant le levier 3 comme référence, s'écrit :

 $$ \left\lbrace \begin{array}{c} \log \frac{\pi_1 (\mathbf{x})}{\pi_3 (\mathbf{x})}=\beta_{1,0} +\sum_{j=1}^p \beta_{1,j} x_j =\eta_1 \newline \log \frac{\pi_2 (\mathbf{x})}{\pi_3 (\mathbf{x})}=\beta_{2,0} +\sum_{j=1}^p \beta_{2,j} x_j =\eta_2  \end{array}\right. $$

Ici, $$\beta_{k,j}$$ est le coefficient du prédicteur $$j$$ dans l'équation de la catégorie $$k$$ ($$j=0$$ correspondant à l'ordonnée à l'origine), et $$\eta_k$$ est la somme pondérée des prédicteurs de l'équation $$k$$. Notez que les prédicteurs $$x_j$$ sont les mêmes dans les deux équations : **les prédicteurs sont partagés entre toutes les équations, et c'est la façon dont ils sont transformés par les coefficients de chaque équation qui donne des log-odds différents pour chaque réponse**. Une fois les paramètres optimisés, on peut retrouver les probabilités estimées à partir de deux faits. Premièrement, la définition du log-odds donne $$\pi_k =\pi_3 e^{\eta_k }$$ pour $$k=1,2$$. Deuxièmement, les probabilités somment à un, de sorte que :

 $$ \pi_3 \left(1+e^{\eta_1 } +e^{\eta_2 } \right)=\pi_1 +\pi_2 +\pi_3 =1 $$ 

En résolvant pour $$\pi_3$$ et en substituant, la probabilité estimée de chaque action est :

$$ \left\lbrace \begin{array}{c} \pi_1 =\frac{e^{\eta_1 } }{1+e^{\eta_1 } +e^{\eta_2 } }\newline \pi_2 =\frac{e^{\eta_2 } }{1+e^{\eta_1 } +e^{\eta_2 } }\newline \pi_3 =\frac{1}{1+e^{\eta_1 } +e^{\eta_2 } } \end{array}\right. $$ 

Il est intéressant de remarquer la complexité de ces probabilités, puisqu'elles dépendent toutes des deux prédicteurs linéaires $$\eta$$ : $$\eta_1$$ peut augmenter ce qui accroît effectivement le log-odds de choisir le levier 1 par rapport au levier 3 sans nécessairement augmenter $$\pi_1$$ lui-même si $$\eta_2$$ est également en train de varier.

# Le codage indicateur

J'ai expliqué comment traiter les variables dépendantes nominales ; vient maintenant le tour des variables indépendantes nominales. Ici, on ne peut pas utiliser l'astuce précédente consistant à remplacer les variables par des log-odds, car les observations $$\mathbf{x}$$, qui sont certes des échantillons de la variable aléatoire $$X$$, sont néanmoins fixées une fois observées. Le **codage indicateur** (*dummy coding*) résout ce problème en remplaçant les prédicteurs nominaux par $$K-1$$ **variables indicatrices**, où $$K$$ est le nombre total de catégories. Chaque variable indicatrice vaut 1 si l'observation appartient à une catégorie donnée et 0 sinon. La catégorie omise est la référence (ou catégorie par défaut), identifiée par le fait que toutes les variables indicatrices valent 0. D'autres schémas de codage existent, par exemple le codage par effets avec les valeurs 1, 0 et -1, mais ils modifient l'interprétation des coefficients et je ne les utiliserai pas. À titre d'exemple, si l'on veut prédire la taille en fonction du sexe, on introduit une variable indicatrice $$d$$ égale à 1 si le sujet est une femme et à 0 s'il s'agit d'un homme et effectuer une régression linéaire simple comme auparavant avec cette nouvelle variable:

 $$ y_i =\beta_0 +\beta_1 d_i +\varepsilon_{i\;} $$ 

La signification de $$\beta_0$$ et de $$\beta_1$$ découle directement de l'espérance conditionnelle de $$Y$$. Pour un homme, $$E(Y \mid d=0)=\beta_0$$ est la taille attendue d'un homme (estimée par la taille moyenne des hommes de l'échantillon) ; pour une femme, $$E(Y \mid d=1) =\beta_0 +\beta_1$$, ce qui signifie que $$\beta_1$$ est la différence de taille attendue entre les femmes et les hommes. Avec trois catégories, on utiliserait deux variables indicatrices, et chaque coefficient serait la différence de résultat attendu entre cette catégorie et la catégorie de référence.

# Simulation de quelques données
Nous sommes maintenant prêts à aborder l'objectif principal de ce tutoriel, qui est d'ajuster une régression logistique multinomiale à des données collectées dans une **tâche de bandit à plusieurs bras**. Ces tâches, qui sont un problème classique **d'apprentissage par renforcement** couramment utilisé en neurosciences, sont constituées d'essais discrets au cours desquels les sujets doivent choisir une action parmi plusieurs dans l'espoir d'obtenir une récompense. Les récompenses et les choix passés peuvent servir de prédicteurs dans un modèle de régression logistique afin de déterminer l'impact de ces événements passés sur les choix actuels, comme dans l'étude de **[Lau and Glimcher (2005)](https://doi.org/10.1901/jeab.2005.110-04)**, qui a utilisé la régression logistique pour étudier les choix de singes dans une tâche à deux alternatives et a montré que l'impact des récompenses passées décroissait avec le temps. Cette étude m'a incité à tester une régression logistique multinomiale plus complexe lors de l'analyse d'une tâche de bandit à trois bras (**[Cinotti et al. (2024)](https://doi.org/10.1111/ejn.16449)**), ce qui s'est révélé plus difficile que prévu, si bien que j'ai finalement eu recours dans ce papier à de simples régressions logistiques optimisées séparément pour chaque action. J'ai depuis eu l'occasion de revenir sur cette technique et de finalement la maîtriser, ce qui m'a poussé à écrire ce tutoriel dans l'espoir qu'il puisse aider d'autres personnes confrontées à des difficultés similaires.

Pour commencer, il nous faut des données : ce sera un simple jeu de données synthétiques de 900 essais, avec les choix et les récompenses enregistrés. Cette expérience virtuelle sera composée de blocs de 100 essais au cours desquels l'un des leviers sera récompensé avec une probabilité de 60 % s'il est sélectionné, tandis que les deux autres le seront avec une probabilité de 20 % chacun. Pour générer des choix plausibles, nous utiliserons un modèle de Q-learning avec un taux d'apprentissage de 0,1, qui sert aussi de taux d'oubli pour les valeurs des leviers non sélectionnés, associé à une sélection d'action softmax avec une température inverse de 5. Si vous ne comprenez pas ce que cela signifie, tout ce que vous avez besoin de retenir, c'est que c'est une manière de générer des données intéressantes.

```matlab
%% Simulate some behavioural data

rng(1);  % make the example reproducible

trials_per_block = 100;
schedule = repmat([1, 2, 3], 1, 3); % identities of the best lever in each block of trials
n_blocks = length(schedule);
n_trials = n_blocks * trials_per_block;

data = struct('choices', nan(n_trials, 1), 'rewards', nan(n_trials,1)); % one entry per trial

% initialise the Q-learner
Q = zeros(1, 3);
learningRate = 0.1;
inverseTemperature = 5;

for trial = 1 : n_trials
    
    % given trial number, get the current best lever
    block = ceil(trial / trials_per_block);
    best = schedule(block);
    
    % generate a random choice based on current Q-values
    probs = exp(inverseTemperature * Q) ./ sum(exp(inverseTemperature * Q));
    choice = randsample(1:3, 1, true, probs);
    
    % generate a stochastic reward based on the choice
    reward = rand < (choice == best) * 0.6 + (choice ~= best) * 0.2;

    % update Q-values for next step
    Q = (1 - learningRate) * Q; % forgetting step that applies to all levers
    Q(choice) = Q(choice) + learningRate * reward; % learning that applies only to the selected lever    
   
    % store trial observations
    data.choices(trial) = choice;
    data.rewards(trial) = reward;
end
```

Pour vous faire une idée de ce à quoi ressemblent ces données, voici un graphique rapide des taux de sélection de chaque levier, calculés par moyennes mobiles sur 20 essais. Vous devriez observer des périodes distinctes pendant lesquelles les trois leviers dominent tour à tour pendant des blocs d'environ 100 essais, ce qui illustre la façon dont l'algorithme de Q-learning suit correctement le programme de récompenses. Les choix restent néanmoins bruités, et si vous avez conservé la même graine aléatoire, vous remarquerez que dans le quatrième bloc (essais 300-400), le taux de sélection du levier 1, qui était pourtant le meilleur levier de ce bloc, était assez faible.

```matlab
%% running averages of lever selection for quick visualisation

figure()
windowSize = 20;
runningChoiceRate = movmean(data.choices == (1:3), windowSize, 1);

plot(runningChoiceRate, 'LineWidth', 1.5);
xlabel('Trial');
ylabel('Choice proportion');
legend('Lever 1', 'Lever 2', 'Lever 3', 'Location', 'best');
```

![Selection rates](/images/multinomial_regression/figure_0.png)

# Construction de la matrice de conception

Nous nous demandons maintenant si les choix et les récompenses précédents permettent de prédire le choix de l'essai courant. Soient $$c_t \in \lbrace 1,2,3\rbrace$$ le levier choisi et $$r_t \in \lbrace 0,1\rbrace$$ la récompense obtenue à l'essai $$t$$. À ce stade, la conception de la régression, c'est-à-dire le choix des prédicteurs et la façon de les encoder, dépend vraiment de vous. Dans mon cas, je voulais utiliser les $$L=10$$ derniers essais (**l'horizon**) pour prédire $$c_t$$, avec cinq prédicteurs pour chaque essai de l'horizon passé $$\ell =1,\ldots,L$$ : des variables indicatrices pour `avoir choisi le levier 1`, `avoir choisi le levier 2`, `avoir été récompensé sur le levier 1`, `avoir été récompensé sur le levier 2` et `avoir été récompensé sur le levier 3`, ce qui fait $$p=5L=50$$ prédicteurs au total, et le modèle à ajuster est le système d'équations introduit plus haut, avec $$\mathbf{x_t}={\left(x_{t,1} ,\ldots,x_{t,p} \right)}^T$$ comme prédicteurs:

 $$ \log \frac{P\left.\left(c_t =k\right|{\mathbf{x}}_t \right)}{P\left.\left(c_t =3\right|{\mathbf{x}}_t \right)}=\beta_{k,0} +\sum_{j=1}^p \beta_{k,j} x_{t,j} ,~~k=1,2 $$ 

Quelques remarques sur cet encodage. Premièrement, il n'y a pas de prédicteur pour le fait d'avoir choisi le levier 3, ce qui signifie que le levier 3 est le levier de référence, comme ci-dessus. Deuxièmement, les trois derniers prédicteurs sont des combinaisons choix-récompense : récompensé sur le levier k signifie que le levier $$k$$ a été choisi et récompensé ; le prédicteur 3 par exemple représente l'effet d'avoir été récompensé sur le levier 1 en plus de l'effet d'avoir sélectionné le levier 1, qui est déjà encodé par le prédicteur 1. D'autres encodages pourraient séparer ces effets différemment, et ce choix a son importance lorsqu'on cherche à interpréter les résultats finaux. La **matrice de conception** $$X$$, aussi apellée **matrice de régression**, est la collection des prédicteurs pour tous les essais ; elle a une ligne par observation (essai) et une colonne par prédicteur. Les choix $$c_t$$ à chaque essai sont de même rassemblés dans le vecteur de réponse $$y$$.

```matlab
horizon = 10;
n_levers = 3;
ref_lever = 3; % reference lever (the dummy coding below assumes lever 3)
non_ref_levers = setdiff(1:n_levers, ref_lever);

n_choice_cols = numel(non_ref_levers);
n_reward_cols = n_levers;

cols_per_lag = n_choice_cols + n_reward_cols;
n_pred = cols_per_lag * horizon;
```

Nous pouvons maintenant construire $$X$$ et $$y$$.

```matlab
% We lose the first 'horizon' trials because there is not enough previous history to construct the predictors.
n_rows = n_trials - horizon;

X = zeros(n_rows, n_pred);
y = zeros(n_rows, 1);

row = 1;

for t = horizon + 1 : n_trials % we go through trials filling in the rows of X and y
    
    % Current choice is the response variable
    y(row) = data.choices(t);
    
    % -------------------------------------------------------------
    % Construct predictors from previous trials
    % -------------------------------------------------------------
    
    for lag = 1 : horizon % we repeat these same steps for each lag trial in turn
        
        % First column belonging to this lag
        lag_offset = (lag-1) * cols_per_lag;
        
        past_choice = data.choices(t-lag);
        past_reward = data.rewards(t-lag);
        
        % ---------------------------------------------------------
        % Previous choice
        % ---------------------------------------------------------
        
        % Lever 3 is the reference category, so it does not get
        % its own dummy variable.
        if past_choice == 1
            X(row, lag_offset + 1) = 1;
        elseif past_choice == 2
            X(row, lag_offset + 2) = 1;
        end
        
        % ---------------------------------------------------------
        % Previous rewarded choice
        % ---------------------------------------------------------
        
        % If the previous choice was rewarded, activate the
        % indicator corresponding to the chosen lever.
        if past_reward == 1            
            reward_col = lag_offset + ...
                n_choice_cols + past_choice;            
            X(row, reward_col) = 1;            
        end        
    end    
    row = row + 1;
end

```

Pour mieux comprendre à quoi ressemble $$X$$, il est utile de regarder côte à côte les premières lignes des données originales et de $$X$$ :

```matlab
% Original data: trial number, choice and reward in one table
n_show = 15;
dataTable = table((1:n_show)', data.choices(1:n_show), data.rewards(1:n_show), ...
    'VariableNames', {'Trial', 'Choice', 'Reward'});
disp(dataTable)
```

```matlabTextOutput
    Trial    Choice    Reward
    _____    ______    ______

      1        2         0   
      2        1         1   
      3        1         1   
      4        1         1   
      5        1         1   
      6        1         0   
      7        1         0   
      8        1         0   
      9        1         1   
     10        1         1   
     11        2         0   
     12        1         0   
     13        3         0   
     14        1         1   
     15        1         0   
```

```matlab

% First rows of X, labelled by the trial they predict
n_rows_shown = 5;
lags_shown   = 2;   % lags 1 and 2 

% Readable column names, e.g. Chose1_L1, Chose2_L1, Rew1_L1, ..., Rew3_L2
baseNames = [compose("Chose%d", non_ref_levers), compose("Rew%d", 1:n_levers)];
predNames = strings(1, 0);
for lag = 1 : lags_shown
    predNames = [predNames, baseNames + "_L" + lag];
end

XTable = array2table(X(1:n_rows_shown, 1:numel(predNames)), ...
    'VariableNames', cellstr(predNames));

% Add the trial each row predicts
XTable = addvars(XTable, (horizon + 1 : horizon + n_rows_shown)', ...
    'Before', 1, 'NewVariableNames', 'Trial');
disp(XTable)
```

```matlabTextOutput
    Trial    Chose1_L1    Chose2_L1    Rew1_L1    Rew2_L1    Rew3_L1    Chose1_L2    Chose2_L2    Rew1_L2    Rew2_L2    Rew3_L2
    _____    _________    _________    _______    _______    _______    _________    _________    _______    _______    _______

     11          1            0           1          0          0           1            0           1          0          0   
     12          0            1           0          0          0           1            0           1          0          0   
     13          1            0           0          0          0           0            1           0          0          0   
     14          0            0           0          0          0           1            0           0          0          0   
     15          1            0           1          0          0           0            0           0          0          0   
```


La première ligne de $$X$$ correspond à l'essai 11, et les cinq premières colonnes décrivent ce qui s'est passé à l'essai 10 où, à condition que la graine aléatoire n'ait pas été modifiée, le levier 1 a été sélectionné et récompensé, de sorte que $$X\left(1,1\right)=1$$ et $$X\left(1,3\right)=1$$. La deuxième ligne de $$X$$ correspond à l'essai 12, de sorte que les événements de l'essai 10 sont décalés vers les colonnes 6 à 10, tandis que les colonnes 1 à 5 décrivent l'essai 11, où l'agent a choisi le levier 2 sans être récompensé.

# Optimisation du modèle de régression

Nous avons maintenant construit la matrice $$X$$, qui contient l'historique comportemental de l'agent, et le vecteur $$y$$, qui contient le choix effectué à l'essai courant. Pour horizon = 10, il y a 5 prédicteurs par décalage × 10 décalages = 50 prédicteurs, donc $$X$$ a 50 colonnes. Les cinq premières colonnes décrivent l'essai immédiatement précédent (décalage 1), les cinq suivantes décrivent l'essai d'avant (décalage 2), et ainsi de suite. Pour ajuster le modèle de régression multinomiale par maximum de vraisemblance, nous utilisons la fonction intégrée `mnrfit`. Les versions récentes de MATLAB proposent aussi une fonction plus sophistiquée, `fitmnr`, mais j'ai choisi `mnrfit` parce que la construction de la matrice de design $$X$$ à codage indicateur rend l'encodage des prédicteurs plus transparent.

```matlab
% Fit the multinomial logistic regression
[B, ~, ~] = mnrfit(X, y, 'Model', 'nominal');

```

La sortie $$B$$ contient les coefficients ajustés $$\beta_{k,j}$$ du modèle. Elle a deux colonnes, la première pour le log-odds de choisir le levier 1 plutôt que le levier 3 ( $$k=1$$ ), et la seconde pour le log-odds du levier 2 par rapport au levier 3 ( $$k=2$$ ). Elle comporte 51 lignes : la première correspond aux ordonnées à l'origine $$\beta_{1,0}$$ et $$\beta_{2,0}$$, et les 50 suivantes aux coefficients des prédicteurs, de sorte que $$B\left(j+1,k\right)=\beta_{k,j}$$. Pour vérifier que le modèle fonctionne, vous pouvez calculer et tracer les probabilités prédites de chaque action à l'aide de la fonction `mnrval`, des coefficients ajustés $$B$$ et des données originales $$X$$ (au lieu de réutiliser $$X$$ comme je le fais ici, la méthode recommandée consisterait plutôt à mettre de côté une partie des données pour servir de test).

```matlab
% Evaluate fitted choice probabilities on the observed predictors

P = mnrval(B, X);

figure()
plot(P, 'LineWidth', 1.5);
xlabel('Trial');
ylabel('Predicted choice probability');
legend('Lever 1', 'Lever 2', 'Lever 3', 'Location', 'best');
grid on;
```

![Predicted choice probabilities](/images/multinomial_regression/figure_1.png)

Par rapport aux moyennes mobiles que nous avons tracées plus haut, les probabilités ajustées reproduisent les grandes tendances des données originales. On voit la sélection des différents leviers alterner en intensité d'un bloc à l'autre et, à moins que vous n'ayez changé la graine aléatoire, vous devriez constater que le quatrième bloc, dans lequel l'agent avait du mal à choisir le bon levier 1, est également ambigu du point de vue du modèle optimisé, ce qui reproduit donc une particularité des données originales.

# Une tentative malavisée d'interpréter les coefficients du modèle

À l'instar de **[Lau and Glimcher (2005)](https://doi.org/10.1901/jeab.2005.110-04)**, vous pourriez vous contenter de tracer certains des coefficients ajustés pour voir comment les prédicteurs passés affectent les choix (figure 6 de cette publication). Une stratégie évidente pourrait consister à examiner séparément les impacts d'avoir été récompensé sur un levier et d'avoir sélectionné un levier sans nécessairement avoir été récompensé, sur le log-odds de ce même levier, comme mesures respectives de **l'effet du renforcement** et de **la persévération du choix** sur le comportement. Dans le cas du levier 1, les coefficients du prédicteur levier 1 choisi se trouvent aux lignes 2, 7, 12, etc. de $$B$$ et les coefficients du prédicteur levier 1 récompensé aux lignes 4, 9, 14, etc. Pour le levier 2, nous nous intéressons aux coefficients de levier 2 choisi (lignes 3, 8, 13, ...) et de levier 2 récompensé (lignes 5, 10, 15, ...).

```matlab
figure()

subplot(2,2,1)
hold on
plot(1 : horizon, B(2 : cols_per_lag : end,1))
title('Perseveration effect on lever1')
xlabel('Past trials')
ylabel('log odds')
set(gca,'XDir', 'reverse')
axis square

subplot(2,2,2)
hold on
plot(1 : horizon, B(4 : cols_per_lag : end,1))
title('Reinforcement effect on lever1')
xlabel('Past trials')
ylabel('log odds')
set(gca,'XDir', 'reverse')
axis square

subplot(2,2,3)
hold on
plot(1 : horizon, B(3 : cols_per_lag : end,2))
title('Perseveration effect on lever2')
xlabel('Past trials')
ylabel('log odds')
set(gca,'XDir', 'reverse')
axis square

subplot(2,2,4)
hold on
plot(1 : horizon, B(5 : cols_per_lag : end,2))
title('Reinforcement effect on lever2')
xlabel('Past trials')
ylabel('log odds')
set(gca,'XDir', 'reverse')
axis square
```

![Fitted coefficients](/images/multinomial_regression/figure_2.png)

Alors que les autres coefficients ne montrent pas de tendance évidente, les coefficients associés aux choix récompensés sur le levier 1 sont positifs et tendent à diminuer à mesure que l'on remonte dans le passé, ce qui suggère un impact positif et décroissant des récompenses passées sur la sélection du levier 1. Cependant, contrairement à l'étude originale de **[Lau and Glimcher (2005)](https://doi.org/10.1901/jeab.2005.110-04)**, qui reposait sur une simple régression logistique à deux issues possibles, l'interprétation de cette observation est plus délicate qu'il n'y paraît. En effet, comme je l'ai déjà souligné, chaque coefficient $$\beta_{k,j}$$ nous indique comment le prédicteur $$j$$ affecte le log-odds du levier $$k$$ par rapport au levier 3, plutôt que la façon dont se comporte la probabilité de choix $$\pi_k$$, qui est la quantité qui nous intéresse réellement. En outre, il ne faut pas oublier que les prédicteurs sont partagés entre les deux log-odds, en sorte qu'obtenir une récompense sur le levier 1 peut également augmenter la probabilité de re-sélectionner ce levier en diminuant le log-odds du levier 2 plutôt qu'en jouant sur le log-odds du levier 1. Cette inter-dépendance des log-odds pourrait expliquer pourquoi les autres coefficients semblent bruités à première vue. Pour savoir comment les prédicteurs affectent les probabilités de choix, il faut donc tenir compte des autres alternatives, un sujet pour une autre fois.

---
# References

- Lau, B. and Glimcher, P.W. (2005), Dynamic Response-By-Response Models of Matching Behavior in Rhesus Monkeys. Journal of the Experimental Analysis of Behavior, 84: 555-579. [https://doi.org/10.1901/jeab.2005.110-04](https://doi.org/10.1901/jeab.2005.110-04)
- Cinotti, F., Coutureau, E., Khamassi, M., Marchand, A. R., & Girard, B. (2024). Regulation of reinforcement learning parameters captures long-term changes in rat behaviour. European Journal of Neuroscience, 60(4), 4469–4490. [https://doi.org/10.1111/ejn.16449](https://doi.org/10.1111/ejn.16449)
