---
title: "Regulation of reinforcement learning parameters captures long-term changes in rat behaviour"
collection: publications
category: manuscripts
permalink: /publication/2024-06-24-metalearning
excerpt: 'This final paper from my PhD looks at the training data from the experiment used in the dopamine paper to find evidence of meta-learning, active regulation of behavioural parameters.'
date: 2024-06-24
venue: 'European Journal of Neuroscience'
paperurl: 'https://doi.org/10.1111/ejn.16449'
codeurl: 'https://github.com/frct/Metalearning'
citation: 'Cinotti, F., Coutureau, E., Khamassi, M., Marchand, A. R., & Girard, B. (2024). Regulation of reinforcement learning parameters captures long-term changes in rat behaviour. European Journal of Neuroscience, 60(4), 4469–4490. https://doi.org/10.1111/ejn.16449'
---
In uncertain environments in which resources fluctuate continuously, animals must permanently decide whether to stabilise learning and exploit what they currently believe to be their best option, or instead explore potential alternatives and learn fast from new observations. While such a trade-off has been extensively studied in pretrained animals facing non-stationary decision-making tasks, it is yet unknown how they progressively tune it while learning the task structure during pretraining. Here, we compared the ability of different computational models to account for long-term changes in the behaviour of 24 rats while they learned to choose a rewarded lever in a three-armed bandit task across 24 days of pretraining. We found that the day-by-day evolution of rat performance and win-shift tendency revealed a progressive stabilisation of the way they regulated reinforcement learning parameters. We successfully captured these behavioural adaptations using a meta-learning model in which either the learning rate or the inverse temperature was controlled by the average reward rate.
