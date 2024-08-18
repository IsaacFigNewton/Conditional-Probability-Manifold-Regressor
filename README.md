<h1>Probability Mass Function Regressor v1</h1>
<h2>Purpose</h2>
<p>Represents the class probability distribution of a dataset using KNN on classes' conditional probabilities for locations in the feature space.</p>
<p>Meant to generalize nonlinear relationships better than normal DT's or Gaussian NB alone.</p>

<br>
<br>

<h2>Definitions</h2>
<p><strong>Let:</strong></p>
<ul>
    <li>The training datapoint count be represented with <strong>n</strong></li>
    <li>The feature count be represented with <strong>m</strong></li>
    <li>The class count be represented with <strong>l</strong></li>
    <li>The training dataset be denoted <strong>A = {a<sub>0</sub>, a<sub>1</sub>, a<sub>2</sub>, … a<sub>i</sub>, …, a<sub>n</sub>}</strong></li>
    <li>The dimensions of the feature space in which all <strong>a</strong> in <strong>A</strong> reside be denoted as <strong>D = {f<sub>0</sub>, f<sub>1</sub>, f<sub>2</sub>, … f<sub>j</sub>, …, f<sub>m</sub>}</strong></li>
    <li>The classes be denoted with <strong>C = {c<sub>0</sub>, c<sub>1</sub>, c<sub>2</sub>, … c<sub>k</sub>, …, c<sub>l</sub>}</strong></li>
    <li>A testing data point be denoted <strong>x</strong></li>
</ul>

<br>
<br>

<h2>Steps:</h2>

<h3>Fitting:</h3>
<ul>
    <li>Calculate local PMF values for each <strong>a<sub>i</sub></strong> in <strong>A</strong> (so you don’t have to do it again later) by finding <strong>P(class | value of a<sub>i</sub> feature<sub>1</sub> ∧ value of a<sub>i</sub> feature<sub>2</sub> ∧ …)</strong> using Bayes’ Theorem.</li>
    <li>AKA: Get the probability of <strong>a<sub>i</sub></strong>’s value in <strong>f<sub>j</sub></strong>, denoted <strong>p<sub>ij</sub></strong></li>
    <li>Find <strong>i,j = 0…i, j = n, m…p<sub>ij</sub></strong></li>
    <li>Multiply by <strong>P(class)</strong></li>
    <li>Repeat this process for all classes</li>
    <li>Basically apply Bayes’ Theorem to all the training data points for each class</li>
</ul>

<h3>Use KNN with k nearest neighbors in each dimension:</h3>
<p>This ensures a standard "resolution" of the algorithm in every dimension but gives a maximum of <strong>k * m</strong> neighbors to consider for estimating the value of <strong>x</strong> and a minimum of <strong>k</strong> neighbors for consideration</p>

<h3>Get weighted average of all neighbors' likelihoods</h3>
<ul>
    <li>Calculate the weighted mean of the probabilities using inverse-distance kernel weighting</li>
</ul>

<br>
<br>

<h2>For the future</h2>
<ul>
    <li>Recursively perform this process on the lth neighbors-of-neighbors of the original, unknown seed datapoint</li>
    <li>Add additional weighting based on variability in class attributable to each feature</li>
</ul>
