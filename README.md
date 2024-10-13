<h1>Conditional Probability Manifold Regressor v2</h1>
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
    <li>Calculate local conditional probability values for each <strong>a<sub>i</sub></strong> in <strong>A</strong> (so you don’t have to do it again later) by finding <strong>P(class | value of a<sub>i</sub> feature<sub>1</sub> ∧ value of a<sub>i</sub> feature<sub>2</sub> ∧ …)</strong> using Bayes’ Theorem.</li>
    <li>AKA: Get the probability of <strong>a<sub>i</sub></strong>’s value in <strong>f<sub>j</sub></strong>, denoted <strong>p<sub>ij</sub></strong></li>
    <li>Find <img src="images/product-of-conditional-probabilities.png" alt="product of conditional probabilities" width=100></li>
    <li>Multiply by <strong>P(class)</strong></li>
    <li>Repeat this process for all classes</li>
    <li>Basically apply Bayes’ Theorem to all the training data points for each class</li>
</ul>

<h3>Estimation:</h3>
<ul>
    <li>Apply KNN on the test point using the conditional probabilities found in training.</li>
</ul>

<br>
<br>

<h2>Features to Implement</h2>
    <ul>
        <li>
            <strong>Implement hyperparameters:</strong>
            <ul>
                <li><strong>branching_depth:</strong>
                    <ul>
                        <li>Define how many times to recursively expand bins on neighboring points.</li>
                    </ul>
                </li>
                <li><strong>n_neighbors:</strong>
                    <ul>
                        <li>Set the number of neighboring points to include for bin expansion at each step.</li>
                        <li>Ensure a standard "resolution" of the algorithm in each dimension.</li>
                        <li>If there are fewer than n_neighbors available to a point (which can only be the case if n_neighbors is greater than the sample), throw an error.</li>
                    </ul>
                </li>
                <li><strong>training_fidelity:</strong>
                    <ul>
                        <li>Define the maximum number of neighbors or bin expansions for training data points.</li>
                        <li>Ensure it's only applied if branching_depth > 0 and n_neighbors > 0.</li>
                    </ul>
                </li>
                <li><strong>test_fidelity:</strong>
                    <ul>
                        <li>Define the maximum number of neighbors or bin expansions for testing data points.</li>
                    </ul>
                </li>
            </ul>
        </li>
        <li>
            <strong>Binning of training data:</strong>
            <ul>
                <li> Binning Process:
                    <ul>
                        <li>Define the binning process for creating discretized bins of increasing sizes around a point.</li>
                        <li>Get the mean class probabilities for each bin.</li>
                        <li>
                            <strong>Increment bin width in all features/dimensions simultaneously:</strong>
                            <ul>
                                <li>Identify the overall nearest neighbor in each dimension using Euclidean distance.</li>
                                <li>Increment bin width in all dimensions based on the distance to this nearest neighbor.</li>
                                <li>Repeat this for multiple bin expansions.</li>
                            </ul>
                        </li>
                    </ul>
                </li>
                <li> Variance Collection:
                    <ul>
                        <li>Calculate the variance matrix for each bin based on points within the bin.</li>
                        <li>Calculate t or z values based on the sample size of points in the bin.</li>
                        <li>Apply inverse weighting based on these variance matrices and t/z values.</li>
                        <li>Weigh incrementally larger discretized bins inversely proportional to their individual variance matrices and t or z values</li>
                    </ul>
                </li>
                <li>
                    <strong>Bins as inputs to a Kalman filter:</strong>
                    <ul>
                        <li>Implement Kalman filtering to process input from different sized bins.</li>
                        <li>Ensure compatibility between varying bin sizes and the Kalman filter input.</li>
                        <li>Note that as the bin width increases, the precision decreases but the accuracy increases.</li>
                    </ul>
                </li>
            </ul>
        </li>
        <li>
            <strong>Binning of test data:</strong>
            <ul>
                <li>Replace KNN with the binning method above, but for test data points as well.</li>
                <li>Implement binning around the test point itself, using the modified conditional probabilities found for the training points.</li>
            </ul>
        </li>
    </ul>
