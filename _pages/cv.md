---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Education
======
* PhD in Computational Neuroscience, Institute of Intelligent Systems and Robotics, Université Paris Sorbonne, 2015-2019
* MSc in Cognitive Neuroscience, Université Paris Descartes, 2015
* MSc in Interdisciplinary Approaches to Life Sciences, Université Paris Diderot, 2014
* Engineering degree in Life Sciences, AgroParisTech, 2011-2014

Work experience
======
* Feb 2022 - Sept 2025: Postdoctoral Research Assistant, University of Reading
    * Interdisciplinary collaboration on the "Mistakes in Living Systems" project.
    * Modelling of thrombus growth: conceptual design, Python implementation, simulation and fitting to experimental data
    * Building an app in R Shiny for the automated analysis and visualisation of pre-processed platelet aggregometry data

* Jan 2021 - Jan 2022: Postdoctoral Research Assistant, University of Oxford, Walton lab
  * Modelling of a foraging experiment: data processing, model design and coding in Python, fitting to data through log-likelihood maximisation, and simulation
  * Collaboration as modelling expert on a study of behavioural flexibility

* June 2019 - Dec 2021: Postdoctoral Research Fellow, University of Nottingham, Humphries lab
  * Development of a Bayesian method for estimating neuronal connection rates
  * Collaboration as a modeller on an international project on goal-directed behaviour
 
* Sep 2015 - June 2019: Research engineer / PhD student, Institute of Intelligent Systems and Robotics, Université Pierre et Marie Curie, Paris
  * Supervisors: CNRS researchers Benoît Girard and Mehdi Khamassi
  * Data analysis of a three-armed bandit task with pharmacological manipulation of dopamine
  * Modelling of three-armed bandit task: model design (MATLAB), optimisation, and simulation
  * Reimplementation into Python and adjustment of a sign-tracking/goal-tracking model previously written in C++ and new simulation-based predictions
  
Skills
======
* Computational modelling
  * Topics: Animal behaviour, Reinforcement learning, Computational Fluid Dynamics, Cellular biology
  * Python, MATLAB, R, GitHub
  * High Performance Computing: Linux clusters, slurm, bash, MPI 
* Data analysis and statistics
  * Data cleaning, pre-processing, and plotting
  * Parametric/non-parametric tests, ANOVAs, ANCOVAs, MANOVAs, PCA
  * Bayesian inference
* Scientific Communication: manuscript writing (LaTeX), poster presentation, and public speaking at conferences and workshops

Publications
======
  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Talks
======
  <ul>{% for post in site.talks reversed %}
    {% include archive-single-talk-cv.html  %}
  {% endfor %}</ul>
