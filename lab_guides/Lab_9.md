

[]{#ch09}Chapter 9. Fighting Bias {#chapter-9.-fighting-bias .title}
---------------------------------

</div>

</div>
:::

We like to think that machines are more rational than us: heartless
silicon applying cold logic. Thus, when computer science introduced
automated decision making into the economy, many hoped that computers
would reduce prejudice and discrimination. Yet, as we mentioned earlier
when looking at mortgage applications and ethnicity, computers are made
and trained by humans, and the data that those machines use stems from
an unjust world. Simply put, if we are not careful, our programs will
amplify human biases.

In the financial industry, anti-discrimination is not only a matter of
morality. Take, for instance, the [**Equal Credit Opportunity
Act**]{.strong} ([**ECOA**]{.strong}), which came into force in 1974 in
the United States. This law explicitly forbids creditors from
discriminating applicants based on race, sex, marital status, and
several other attributes. It also requires creditors to inform
applicants about the reasons for denial.

The algorithms discussed in this book are discrimination machines. Given
an objective, these machines will find the features that it's best to
discriminate on. Yet, as we've discussed discrimination is not always
okay.

While it\'s okay to target ads for books from a certain country to
people who are also from that country, it\'s usually not okay, and
thanks to the ECOA, often illegal, to deny a loan to people from a
certain country. Within the financial domain, there are much stricter
rules for discrimination than those seen in book sales. This is because
decisions[]{#id716 .indexterm} in the financial domain have a much more
severe impact on people\'s lives than those of book sales.

Equally, discrimination in this context is [**feature
specific**]{.strong}. For example, while it\'s okay to discriminate
against loan applicants based on their history of repaying loans, it\'s
not okay to do so based on their country of origin, unless there are
sanctions against that country or similar overarching laws in place.

Throughout this chapter, we\'ll discuss the following:

::: {.itemizedlist}
-   Where bias in machines comes from

-   The legal implications of biased [**machine learning**]{.strong}
    ([**ML**]{.strong}) models

-   How observed unfairness can be reduced

-   How models can be inspected for bias and unfairness

-   How causal modeling can reduce bias

-   How unfairness is a complex systems failure that needs to be
    addressed in non-technical ways
:::

The algorithms discussed in this book are feature extraction algorithms.
Even if regulated features are omitted, an algorithm might infer them
from proxy features and then discriminate based on them anyway. As an
example of this, ZIP codes can be used to predict race reasonably well
in many cities in the United States. Therefore, omitting regulated
features is not enough when it comes to combating bias.



[]{#ch09lvl1sec105}Sources of unfairness in machine learning {#sources-of-unfairness-in-machine-learning .title style="clear: both"}
------------------------------------------------------------

</div>

</div>

------------------------------------------------------------------------
:::

As we have discussed many[]{#id717 .indexterm} times throughout this
book, models are a function of the data that they are trained on.
Generally speaking, more data will lead to smaller errors. So, by
definition, there is less[]{#id718 .indexterm} data on minority groups,
simply because there are fewer people in those groups.

This [**disparate sample size**]{.strong} can lead to worse model
performance for the minority group. As a result, this increased error is
often known as a [**systematic error**]{.strong}. The model might have
to overfit the majority group data so that the relationships it found do
not[]{#id719 .indexterm} apply to the minority group data. Since there
is little minority group data, this is not punished as much.

Imagine you are training a credit scoring model, and the clear majority
of your data comes from people living in lower Manhattan, and a small
minority of it comes from people living in rural areas. Manhattan
housing is much more expensive, so the model might learn that you need a
very high income to buy an apartment. However, rural housing is much
cheaper in comparison. Even so, because the model is largely trained on
data from Manhattan, it might deny loan applications to rural applicants
because they tend to have lower incomes than their Manhattan peers.

Aside from sample size issues, our data can be biased by itself. For
example, \"raw data\" does not exist. Data does not appear naturally,
instead it\'s measured by humans using human-made measurement protocols,
which in themselves can be biased in many different ways.

Biases could include having [**sampling biases**]{.strong}, such as in
the[]{#id720 .indexterm} Manhattan housing example, or having
[**measurement biases**]{.strong}, which is when your sample might not
measure what it is intended to measure, or may even discriminate against
one group.

Another bias that\'s possible is [**pre-existing social
biases**]{.strong}. These are []{#id721 .indexterm}visible in word
vectors, for instance, in Word2Vec, where the mapping from father to
doctor in latent space maps from mother to nurse. Likewise, the vector
from man to computer programmer maps from woman to homemaker. This is
because sexism is encoded within the written language of our sexist
society. Until today, typically speaking doctors have usually been men
and nurses have usually been women. Likewise, tech companies\' diversity
statistics reveal that far more men are computer programmers than women,
and these biases get encoded into models.



[]{#ch09lvl1sec106}Legal perspectives {#legal-perspectives .title style="clear: both"}
-------------------------------------

</div>

</div>

------------------------------------------------------------------------
:::

There are two doctrines in anti-discrimination law: [*disparate
treatment*]{.emphasis}, and [*disparate impact*]{.emphasis}. Let\'s take
a minute to look at each of these:

::: {.itemizedlist}
-   [**Disparate treatment**]{.strong}: This is one kind of unlawful
    discrimination. Intentionally discriminating against ZIP codes with
    the hope of discriminating against race is not legal. Disparate
    treatment problems have less to do with the algorithm and more to do
    with the organization running it.

-   [**Disparate impact**]{.strong}: This can be a problem if an
    algorithm is deployed that has a different impact []{#id722
    .indexterm}on different groups, even without the organization
    knowing about it. Let\'s walk through a lending scenario in which
    disparate impact could be a problem. Firstly, the plaintiff must
    establish that there is a disparate impact. Assessing if there\'s a
    disparate impact is[]{#id723 .indexterm} usually done with the
    [**four-fifths rule**]{.strong}, which says that if the selection
    rate of a group is less than 80% of the group, then it is regarded
    as evidence of adverse impact. If a lender has 150 loan applicants
    from group A, of which 100, or 67%, are accepted, and 50 applicants
    from group B, of which 25 are accepted, the difference in selection
    is 0.5/0.67 = 0.746, which qualifies as evidence for discrimination
    against group B. The defendant can counter this by showing that the
    decision procedure is justified as necessary.

    After this is done, the plaintiff has the opportunity to show that
    the goal of the procedure could also be achieved with a different
    procedure that shows a smaller disparity.
:::

::: {.note style="margin-left: 0.5in; margin-right: 0.5in;"}
### []{#note46}Note {#note .title}

[**Note**]{.strong}: For a more in-depth overview of these topics, see
Moritz Hardt\'s 2017 NeurIPS presentation on the topic at
<http://mrtz.org/nips17/#/11>.
:::

The disparate treatment doctrine tries to achieve procedural fairness
and equal opportunity. The disparate impact doctrine aims for
distributive justice and minimized inequality in outcomes.

There is an intrinsic tension between the two doctrines, as illustrated
by the Ricci V. DeStefano case from 2009. In this case, 19 white
firefighters and 1 Hispanic firefighter sued their employer, the New
Haven Fire Department. The firefighters had all passed their test for
promotion, yet their black colleagues did not score the mark required
for the promotion. Fearing a disparate impact lawsuit, the city
invalidated the test results and did not promote the firefighters.
Because the evidence for disparate impact was not strong enough, the
Supreme Court of the United States eventually ruled that the
firefighters should have been promoted.

Given the complex legal and technical situation around fairness in
machine learning, we\'re going to dive into how we can define and
quantify fairness, before using this insight to create fairer models.


[]{#ch09lvl1sec107}Observational fairness {#observational-fairness .title style="clear: both"}
-----------------------------------------

</div>

</div>

------------------------------------------------------------------------
:::

Equality is often seen as[]{#id724 .indexterm} a purely qualitative
issue, and as such, it\'s often dismissed by quantitative-minded
modelers. As this section will show, equality can be seen from a
quantitative perspective, too. Consider a classifier, [*c,*]{.emphasis}
with input [*X*]{.emphasis}, some sensitive input, [*A*]{.emphasis}, a
target, [*Y*]{.emphasis} and output [*C*]{.emphasis}. Usually, we would
denote the classifier output as
[![](4_files/B10354_09_001.jpg)]{.inlinemediaobject}, but
for readability, we follow CS 294 and name it [*C*]{.emphasis}.

Let\'s say that our classifier is being used to decide who gets a loan.
When would we consider this classifier to be fair and free of bias? To
answer this question, picture two demographics, group A and B, both loan
applicants. Given a credit score, our classifier must find a cutoff
point. Let\'s look at the distribution of applicants in this graph:

::: {.note style="margin-left: 0.5in; margin-right: 0.5in;"}
### []{#note47}Note {#note .title}

[**Note**]{.strong}: The data for this example is synthetic; you can
find the Excel file used for these calculations in the GitHub repository
of this book,
<https://github.com/PacktPublishing/Machine-Learning-for-Finance/blob/master/9.1_parity.xlsx>.
:::

::: {.mediaobject}
![](4_files/B10354_09_01.jpg)

::: {.caption}
Max profits
:::
:::

For this exercise, we assume that a successful applicant yields a profit
of \$300, while a defaulting successful applicant costs \$700. The
cutoff point here has been chosen to maximize profits:

So, what can we see? We can see the following:

::: {.itemizedlist}
-   In orange are applicants who would not have[]{#id725 .indexterm}
    repaid the loan and did not get accepted: [**true
    negatives**]{.strong} ([**TNs**]{.strong}).

-   In blue are applicants who would have repaid[]{#id726 .indexterm}
    the loan but did not get accepted: [**false negatives**]{.strong}
    ([**FNs**]{.strong}).

-   In yellow are applicants who did get the[]{#id727 .indexterm} loan
    but did not pay it back: [**false positives**]{.strong}
    ([**FPs**]{.strong}).

-   In gray are applicants who did receive the loan[]{#id728 .indexterm}
    and paid it back: [**true positives**]{.strong}
    ([**TPs**]{.strong}).
:::

As you can see, there []{#id729 .indexterm}are several issues with this
choice of cutoff point. [**Group B**]{.strong} applicants need to have a
better score to get a loan than [**Group A**]{.strong} applicants,
indicating disparate treatment. At the same time, only around 51% of
[**Group A**]{.strong} applicants get a loan but only 37% of [**Group
B**]{.strong} applicants do, indicating disparate impact.

A [*group unaware threshold*]{.emphasis}, which we can see below, would
give both groups the same minimum score:

::: {.mediaobject}
![](4_files/B10354_09_02.jpg)

::: {.caption}
Equal cutoff
:::
:::

In the preceding graph, while both groups have the same cutoff rate,
[**Group A**]{.strong} has been given fewer loans. At the same time,
predictions for [**Group A**]{.strong} have a lower accuracy than the
predictions given for [**Group B**]{.strong}. It seems that although
both groups face the same score threshold, [**Group A**]{.strong} is at
a disadvantage.

Demographic parity aims to achieve fairness by ensuring that both groups
have the same chance of receiving the loan. This method aims to achieve
the same selection rate for both groups, which is what impact disparity
is measured by. Mathematically, this process can be expressed as
follows:

::: {.mediaobject}
![](4_files/B10354_09_002.jpg)
:::

If we apply this[]{#id730 .indexterm} rule to the same context as we
used previously, we\'ll arrive at the following cutoff points:

::: {.mediaobject}
![](4_files/B10354_09_03.jpg)

::: {.caption}
Equal pick rate
:::
:::

While this method cannot be blamed for statistical discrimination and
disparate impact, it can be blamed for disparate treatment. In the equal
pick rate graphic we can see how [**Group A**]{.strong} is given a lower
threshold score; meanwhile, there are more successful [**Group
A**]{.strong} applicants who default on their loans. In fact, [**Group
A**]{.strong} is not profitable and gets subsidized by [**Group
B**]{.strong}. Accepting a worse economic outcome to favor a certain
group is also known as taste-based discrimination. It could be said that
the higher thresholds for [**Group B**]{.strong} are unfair, as they
have a lower FP rate.

TP parity, which is also called equal opportunity, means that both
demographics have the same TP rate. For people who can pay back the
loan, the same chance of getting a loan should exist. Mathematically,
this can be expressed as follows:

::: {.mediaobject}
![](4_files/B10354_09_003.jpg)
:::

Applied to our data, this policy looks similar to demographic parity,
except that the group cutoff point is even lower:

::: {.mediaobject}
![](4_files/B10354_09_04.jpg)

::: {.caption}
Equal opportunity
:::
:::

Equal opportunity []{#id731 .indexterm}can address many of the problems
of demographic parity, as most people believe that everyone should be
given the same opportunities. Still, our classifier is less accurate for
[**Group A**]{.strong}, and there is a form of disparate treatment in
place.

Accuracy parity tells us that the accuracy of predictions should be the
same for both groups. Mathematically, this can be expressed as follows:

::: {.mediaobject}
![](4_files/B10354_09_004.jpg)
:::

The probability that the classifier is correct should be the same for
the two possible values of the sensitive variable [*A*]{.emphasis}. When
we apply this criteria to our data, we arrive at the following output:

::: {.mediaobject}
![](4_files/B10354_09_05.jpg)

::: {.caption}
Equal accuracy
:::
:::

From the preceding diagram, the downside becomes apparent. In order to
satisfy the accuracy constraint, members of [**Group B**]{.strong} are
given much easier access to loans.

Therefore to solve this, trade-offs are[]{#id732 .indexterm} necessary
because no classifier can have precision parity, TP parity, and FP
parity unless the classifier is perfect. [*C = Y,*]{.emphasis} or both
demographics have the same base rates:

::: {.mediaobject}
![](4_files/B10354_09_005.jpg)
:::

There are many more ways to express fairness. The key takeaway, however,
is that none of them perfectly satisfies all of the fairness criteria.
For any two populations with unequal base rates, and unequal chances of
repaying their loan, establishing statistical parity requires the
introduction of a treatment disparity.

This fact has led to a number of debates, with the best practice to
express and eliminate discrimination having not been agreed on yet. With
that being said, even if the perfect mathematical expression of fairness
was found, it would not immediately lead to perfectly fair systems.

Any machine learning algorithm is part of a bigger system. Inputs
[*X*]{.emphasis} are often not as clearly defined as a different
algorithm in the same system that might use different inputs.
Demographic groups [*A*]{.emphasis} are often not clearly defined or
inferred. Even the output, [*C,*]{.emphasis} of the classifier can often
not be clearly distinguished, as many algorithms together might perform
the classification task while each algorithm is predicting a different
output, such as a credit score or a profitability estimate.

Good technology is not a substitute for good policy. Blindly following
an algorithm without the opportunity for individual consideration or
appeal will always lead to unfairness. With that being said, while
mathematical fairness criteria cannot solve all the fairness issues that
we face, it is surely worth trying to make machine learning algorithms
fairer, which is what the next section is about.



[]{#ch09lvl1sec108}Training to be fair {#training-to-be-fair .title style="clear: both"}
--------------------------------------

</div>

</div>

------------------------------------------------------------------------
:::

There are multiple ways to[]{#id733 .indexterm} train models to be
fairer. A simple approach could be using the different fairness measures
that we have listed in the previous section as an additional loss.
However, in practice, this approach has turned out to have several
issues, such as having poor performance on the actual classification
task.

An alternative approach is to use an adversarial network. Back in 2016,
Louppe, Kagan, and Cranmer published the paper [*Learning to Pivot with
Adversarial Networks*]{.emphasis}, available at
<https://arxiv.org/abs/1611.01046>. This paper showed how to use an
adversarial network to train a classifier to ignore a nuisance
parameter, such as a sensitive feature.

In this example, we will train a classifier to predict whether an adult
makes over \$50,000 in annual income. The challenge here is to make our
classifier unbiased from the influences of race and gender, with it only
focusing on features that we can discriminate on, including their
occupation and the gains they make from their capital.

To this end, we must train a classifier and an adversarial network. The
[]{#id734 .indexterm}adversarial network aims to classify the sensitive
attributes, [*a*]{.emphasis}, gender and race, from the predictions of
the classifier:

::: {.mediaobject}
![](5_files/B10354_09_06.jpg)

::: {.caption}
Making an unbiased classifier to detect the income of an adult
:::
:::

The classifier aims to classify by income but also aims to fool the
adversarial network. The classifier\'s minimization objective formula is
as follows:

::: {.mediaobject}
![](5_files/B10354_09_006.jpg)
:::

Within that formula,
[![](5_files/B10354_09_007.jpg)]{.inlinemediaobject} is a binary
cross-entropy loss of the classification, while
[![](5_files/B10354_09_008.jpg)]{.inlinemediaobject} is the adversarial
loss. [![](5_files/B10354_09_009.jpg)]{.inlinemediaobject} represents a
hyperparameter that we can use to amplify or reduce the impact of the
adversarial loss.

::: {.note style="margin-left: 0.5in; margin-right: 0.5in;"}
### []{#note48}Note {#note .title}

[**Note**]{.strong}: This implementation of the adversarial fairness
method follows an implementation by Stijn Tonk and Henk Griffioen. You
can find the code to this chapter on Kaggle at
<https://www.kaggle.com/jannesklaas/learning-how-to-be-fair>.

Stijn\'s and Henk\'s original blogpost can be found here:
<https://blog.godatadriven.com/fairness-in-ml>.
:::

To train this model fairly, we not only need data [*X*]{.emphasis} and
targets [*y*]{.emphasis}, but also data about the sensitive attributes,
[*A*]{.emphasis}. In the example we\'re going to work on, we\'ll be
taking data from the 1994 US census provided by the UCI repository:
<https://archive.ics.uci.edu/ml/datasets/Adult>.

To make[]{#id735 .indexterm} loading the data easier, it has been
transformed into a CSV file with column headers. As a side note, please
refer to the online version to see the data as viewing the data would be
difficult in the format of the book.

First, we load the data. The dataset contains data about people from a
number of different races, but for the simplicity of this task, we will
only be focusing on white and black people for the `race`{.literal}
attribute. To do this, we need to run the following code:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
path = '../input/adult.csv'
input_data = pd.read_csv(path, na_values="?")
input_data = input_data[input_data['race'].isin(['White', 'Black'])]
```
:::

Next, we select the sensitive attributes, in this case we\'re focusing
on race and gender, into our sensitive dataset, `A`{.literal}. We
one-hot encode the data so that \"Male\" equals one for the
`gender`{.literal} attribute and `White`{.literal} equals one for the
`race`{.literal} attribute. We can achieve this by running the following
code:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
sensitive_attribs = ['race', 'gender']
A = input_data[sensitive_attribs]
A = pd.get_dummies(A,drop_first=True)
A.columns = sensitive_attribs
```
:::

Our target is the `income`{.literal} attribute. Therefore, we need to
encode `>50K`{.literal} as 1 and everything else as zero, which is
achieved by writing this code:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
y = (input_data['income'] == '>50K').astype(int)
```
:::

To get our training data, we firstly remove the sensitive and target
attributes. Then we fill all of the missing values and one-hot encode
all of the data, as you can see in the following code:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
X = input_data.drop(labels=['income', 'race', 'gender'],axis=1)

X = X.fillna('Unknown')

X = pd.get_dummies(X, drop_first=True)
```
:::

Finally, we split the data into train and test sets. As seen in the
following code, we then stratify the data to ensure that the same number
of high earners are in both the test and training data:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
X_train, X_test, y_train, y_test, A_train, A_test = \
train_test_split(X, y, A, test_size=0.5, stratify=y, random_state=7)
```
:::

To ensure the data works nicely with the neural network, we\'re now
going to scale the data using scikit-learn\'s
`StandardScaler`{.literal}:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
scaler = StandardScaler().fit(X_train)

X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
                       
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
```
:::

We need a metric of how[]{#id736 .indexterm} fair our model is. We are
using the disparate impact selection rule. The `p_rule`{.literal} method
calculates the share of people classified to have over \$50,000 income
from both groups and then returns the ratio of selections in the
disadvantaged demographic over the ratio of selections in the advantaged
group.

The goal is for the `p_rule`{.literal} method to return at least 80% in
order to meet the four-fifths rule for both race and gender. The
following code shows how this function is only used for monitoring, and
not as a loss function:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
def p_rule(y_pred, a_values, threshold=0.5):
    y_a_1 = y_pred[a_values == 1] > threshold if threshold else y_pred[a_values == 1]                                           #1
    y_a_0 = y_pred[a_values == 0] > threshold if threshold else y_pred[a_values == 0] 
    odds = y_a_1.mean() / y_a_0.mean()                          #2
    return np.min([odds, 1/odds]) * 100
```
:::

Let\'s explore this code in some more detail. As you can see from the
preceding code block, it\'s created with two key features, which are as
follows:

::: {.orderedlist}
1.  Firstly, we select who is given a selected threshold. Here, we
    classify everyone whom the model assigns a chance of over 50% of
    making \$50,000 or more as a high earner.

2.  Secondly, we calculate the selection ratio of both demographics. We
    divide the ratio of the one group by the ratio of the other group.
    By returning the minimum of either the odds or one divided by the
    odds, we ensure the return of a value below one.
:::

To make the model setup a bit easier, we need to define the number of
input features and the number of sensitive features. This is something
that is simply done by running these two lines:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
n_features=X_train.shape[1]
n_sensitive=A_train.shape[1]
```
:::

Now we set up our []{#id737 .indexterm}classifier. Note how this
classifier is a standard classification neural network. It features
three hidden layers, some dropout, and a final output layer with a
sigmoid activation, which occurs since this is a binary classification
task. This classifier is written in the Keras functional API.

To make sure you understand how the API works, go through the following
code example and ensure you understand why the steps are taken:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
clf_inputs = Input(shape=(n_features,))
x = Dense(32, activation='relu')(clf_inputs)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(1, activation='sigmoid', name='y')(x)
clf_net = Model(inputs=[clf_inputs], outputs=[outputs])
```
:::

The adversarial network is a classifier with two heads: one to predict
the applicant\'s race from the model output, and one to predict the
applicant\'s gender:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
adv_inputs = Input(shape=(1,))
x = Dense(32, activation='relu')(adv_inputs)
x = Dense(32, activation='relu')(x)
x = Dense(32, activation='relu')(x)
out_race = Dense(1, activation='sigmoid')(x)
out_gender = Dense(1, activation='sigmoid')(x)
adv_net = Model(inputs=[adv_inputs], outputs=[out_race,out_gender])
```
:::

As with generative adversarial networks, we have to make the networks
trainable and untrainable multiple times. To make this easier, the
following function will create a function that makes a network and all
its layers either trainable or untrainable:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
def make_trainable_fn(net):              #1
    def make_trainable(flag):            #2
        net.trainable = flag             #3
        for layer in net.layers:
            layer.trainable = flag
    return make_trainable                #4
```
:::

From the preceding code, there are four key features that we should take
a moment to explore:

::: {.orderedlist}
1.  The function accepts a Keras neural network, for which the train
    switch function will be created.

2.  Inside the function, a second function is created. This second
    function accepts a Boolean flag
    (`True`{.literal}/`False`{.literal}).

3.  When[]{#id738 .indexterm} called, the second function sets the
    network\'s trainability to the flag. If `False`{.literal} is passed,
    the network is not trainable. Since the layers of the network can
    also be used in other networks, we ensure that each individual layer
    is not trainable, too.

4.  Finally, we return the function.
:::

Using a function to create another function might seem convoluted at
first, but this allows us to create \"switches\" for the neural network
easily. The following code snippet shows us how to create switch
functions for the classifier and the adversarial network:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
trainable_clf_net = make_trainable_fn(clf_net)
trainable_adv_net = make_trainable_fn(adv_net)
```
:::

To make the classifier trainable, we can use the function with the
`True`{.literal} flag:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
trainable_clf_net(True)
```
:::

Now we can compile our classifier. As you will see later on in this
chapter, it is useful to keep the classifier network as a separate
variable from the compiled classifier with which we make predictions:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
clf = clf_net
clf.compile(loss='binary_crossentropy', optimizer='adam')
```
:::

Remember that to train our classifier, we need to run its predictions
through the adversary as well as obtaining the adversary loss and
applying the negative adversary loss to the classifier. This is best
done by packing the classifier and adversary into one network.

To do this, we must first create a new model that maps from the
classifier inputs to the classifier and adversary outputs. We define the
adversary output to be a nested function of the adversarial network and
the classifier network. This way, the predictions of the classifier get
immediately passed on to the adversary:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
adv_out = adv_net(clf_net(clf_inputs))
```
:::

We then define the classifier output to be the output of the classifier
network, just as we would for classification:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
clf_out = clf_net(clf_inputs)
```
:::

Then, we define the combined model to map from the classifier input,
that is, the data about an applicant, to the classifier output and
adversary output:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
clf_w_adv = Model(inputs=[clf_inputs], outputs=[clf_out]+adv_out)
```
:::

When training the []{#id739 .indexterm}combined model, we only want to
update the weights of the classifier, as we will train the adversary
separately. We can use our switch functions to make the classifier
network trainable and the adversarial network untrainable:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
trainable_clf_net(True)
trainable_adv_net(False)
```
:::

Remember the hyperparameter,
[![](5_files/B10354_09_010.jpg)]{.inlinemediaobject}, from the preceding
minimization objective. We need to set this parameter manually for both
sensitive attributes. As it turns out, the networks train best if lambda
for race is set much higher than lambda for gender.

With the lambda values in hand, we can create the weighted loss:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
loss_weights = [1.]+[-lambda_param for lambda_param in lambdas]
```
:::

The preceding expression leads to loss weights of \[1.,-130,-30\]. This
means the classification error has a weight of 1, the race prediction
error of the adversary has a weight of -130, and the gender prediction
error of the adversary has a weight of -30. Since the losses of the
adversarial\'s prediction have negative weights, gradient descent will
optimize the parameters of the classifier to [*increase*]{.emphasis}
these losses.

Finally, we can compile the combined network:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
clf_w_adv.compile(loss='binary_crossentropy'), loss_weights=loss_weights,optimizer='adam')
```
:::

With the classifier and combined classifier-adversarial model in place,
the only thing missing is a compiled adversarial model. To get this,
we\'ll first define the adversarial model to map from the classifier
inputs to the outputs of the nested adversarial-classifier model:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
adv = Model(inputs=[clf_inputs], outputs=adv_net(clf_net(clf_inputs)))
```
:::

Then, when training the adversarial model, we want to optimize the
weights of the adversarial network and not of the classifier network, so
we use our switch functions to make the adversarial trainable and the
classifier not trainable:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
trainable_clf_net(False)
trainable_adv_net(True)
```
:::

Finally, we compile the adversarial model just like we would with a
regular Keras model:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
adv.compile(loss='binary_crossentropy', optimizer='adam')
```
:::

With all the pieces in hand, we can now pretrain the classifier. This
means we train the classifier without any special fairness
considerations:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
trainable_clf_net(True)
clf.fit(X_train.values, y_train.values, epochs=10)
```
:::

After we have trained the model, we can make predictions on the
validation set to evaluate both the model\'s fairness and accuracy:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
y_pred = clf.predict(X_test)
```
:::

Now we\'ll []{#id740 .indexterm}calculate the model\'s accuracy and
`p_rule`{.literal} for both gender and race. In all calculations, we\'re
going to use a cutoff point of 0.5:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
acc = accuracy_score(y_test,(y_pred>0.5))* 100
print('Clf acc: {:.2f}'.format(acc))

for sens in A_test.columns:
    pr = p_rule(y_pred,A_test[sens])
    print('{}: {:.2f}%'.format(sens,pr))
```
:::

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
out:
Clf acc: 85.44
race: 41.71%
gender: 29.41%
```
:::

As you can see, the classifier achieves a respectable accuracy, 85.44%,
in predicting incomes. However, it is deeply unfair. It gives women only
a 29.4% chance to make over \$50,000 than it does men.

Equally, it discriminates strongly on race. If we used this classifier
to judge loan applications, for instance, we would be vulnerable to
discrimination lawsuits.

::: {.note style="margin-left: 0.5in; margin-right: 0.5in;"}
### []{#note49}Note {#note-1 .title}

[**Note**]{.strong}: Neither gender or race was included in the features
of the classifier. Yet, the classifier discriminates strongly on them.
If the features can be inferred, dropping sensitive columns is not
enough.
:::

To get out of this mess, we will pretrain the adversarial network before
training both networks to make fair predictions. Once again, we use our
switch functions to make the classifier untrainable and the adversarial
trainable:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
trainable_clf_net(False)
trainable_adv_net(True)
```
:::

As the distributions for race and gender in the data might be skewed,
we\'re going to use weighted classes to adjust for this:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
class_weight_adv = compute_class_weights(A_train)
```
:::

We then train the adversary to predict race and gender from the training
data through the predictions of the classifier:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
adv.fit(X_train.values, np.hsplit(A_train.values, A_train.shape[1]), class_weight=class_weight_adv, epochs=10)
```
:::

NumPy\'s `hsplit`{.literal} function splits the 2D `A_train`{.literal}
matrix into two vectors that are then used to train the two model heads.

With the classifier[]{#id741 .indexterm} and adversary pretrained, we
will now train the classifier to fool the adversary in order to get
better at spotting the classifier\'s discrimination. Before we start, we
need to do some setup. We want to train for 250 epochs, with a batch
size of 128, with two sensitive attributes:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
n_iter=250
batch_size=128
n_sensitive = A_train.shape[1]
```
:::

The combined network of the classifier and adversarial also needs some
class weights. The weights for the income predictions, less/more than
\$50,000, are both one. For the adversarial heads of the combined model,
we use the preceding computed adversarial class weights:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
class_weight_clf_w_adv = [{0:1., 1:1.}]+class_weight_adv
```
:::

To keep track of metrics, we set up one DataFrame for validation
metrics, accuracy, and area under the curve, as well as for the fairness
metrics. The fairness metrics are the `p_rule`{.literal} values for race
and gender:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
val_metrics = pd.DataFrame()
fairness_metrics = pd.DataFrame()
```
:::

Inside the main training loop, three steps are performed: training the
adversarial network, training the classifier to be fair, and printing
out validation metrics. For better explanations, all three are printed
separately here.

Within the code, you will find them in the same loop, where
`idx`{.literal} is the current iteration:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
for idx in range(n_iter):
```
:::

The first step is to train the adversarial network. To this end, we\'re
going to make the classifier untrainable, the adversarial network
trainable, and then train the adversarial network just as we did before.
To do this, we need to run the following code block:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
trainable_clf_net(False)
trainable_adv_net(True)
adv.fit(X_train.values, np.hsplit(A_train.values, A_train.shape[1]), batch_size=batch_size, class_weight=class_weight_adv, epochs=1, verbose=0)
```
:::

Training the classifier to be a good classifier but also to fool the
adversary and be fair involves three steps. Firstly, we make the
adversary untrainable and the classifier trainable:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
trainable_clf_net(True)
trainable_adv_net(False)
```
:::

Then we sample a batch from `X`{.literal}, `y`{.literal}, and
`A`{.literal}:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
indices = np.random.permutation(len(X_train))[:batch_size]
X_batch = X_train.values[indices]
y_batch = y_train.values[indices]
A_batch = A_train.values[indices]
```
:::

Finally, we []{#id742 .indexterm}train the combined adversary and
classifier. Since the adversarial network is set to not be trainable,
only the classifier network will be trained. However, the loss from the
adversarial network\'s predictions of race and gender gets
backpropagated through the entire network, so that the classifier learns
to fool the adversarial network:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
clf_w_adv.train_on_batch(X_batch, [y_batch]+\np.hsplit(A_batch, n_sensitive),class_weight=class_weight_clf_w_adv)
```
:::

Finally, we want to keep track of progress by first making predictions
on the test:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
y_pred = pd.Series(clf.predict(X_test).ravel(), index=y_test.index)
```
:::

We then calculate the area under the curve (`ROC AUC`{.literal}) and the
accuracy of the predictions, and save them in the
`val_metrics`{.literal} DataFrame:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
roc_auc = roc_auc_score(y_test, y_pred)
acc = accuracy_score(y_test, (y_pred>0.5))*100

val_metrics.loc[idx, 'ROC AUC'] = roc_auc
val_metrics.loc[idx, 'Accuracy'] = acc
```
:::

Next up, we calculate `p_rule`{.literal} for both race and gender and
save those values in the fairness metrics:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
for sensitive_attr :n A_test.columns:
    fairness_metrics.loc[idx, sensitive_attr] =\
    p_rule(y_pred,A_test[sensitive_attr])
```
:::

If we plot both the fairness and validation metrics, we\'ll arrive at
the following plot:

::: {.mediaobject}
![](5_files/B10354_09_07.jpg)

::: {.caption}
Pivot train progress
:::
:::

As you can []{#id743 .indexterm}see, the fairness scores of the
classifier steadily increase with training. After about 150 epochs, the
classifier satisfies the four-fifths rule. At the same time, the
p-values are well over 90%. This increase in fairness comes at only a
small decrease in accuracy and area under the curve. The classifier
trained in this manner is clearly a fairer classifier with similar
performance, and is thus preferred over a classifier trained without
fairness criteria.

The pivot approach to fair machine learning has a number of advantages.
Yet, it cannot rule out unfairness entirely. What if, for example, there
was a group that the classifier discriminates against that we did not
think of yet? What if it discriminates on treatment, instead of impact?
To make sure our models are not biased, we need more technical and
social tools, namely [*interpretability*]{.emphasis},
[*causality*]{.emphasis}, and [*diverse development teams*]{.emphasis}.

In the next section, we\'ll discuss how to train machine learning models
that learn causal relationships, instead of just statistical
associations.



[]{#ch09lvl1sec109}Causal learning {#causal-learning .title style="clear: both"}
----------------------------------

</div>

</div>

------------------------------------------------------------------------
:::

This book is by and large a book about []{#id744 .indexterm}statistical
learning. Given data [*X*]{.emphasis} and targets [*Y*]{.emphasis}, we
aim to estimate [![](6_files/B10354_09_011.jpg)]{.inlinemediaobject},
the distribution of target values given certain data points. Statistical
learning allows us to create a number of great models with useful
applications, but it doesn\'t allow us to claim that [*X*]{.emphasis}
being [*x*]{.emphasis} caused [*Y*]{.emphasis} to be [*y*]{.emphasis}.

This statement is critical if we intend to manipulate [*X*]{.emphasis}.
For instance, if we want to know whether giving insurance to someone
leads to them behaving recklessly, we are not going to be satisfied with
the statistical relationship that people with insurance behave more
reckless than those without. For instance, there could be
a self-selection bias present about the number of reckless people
getting insurance, while those who are not marked as reckless don\'t.

Judea Pearl, a famous computer scientist, invented a notation for causal
models called do-calculus; we are interested in
[![](6_files/B10354_09_012.jpg)]{.inlinemediaobject}, which is the
probability of someone behaving recklessly after we manipulated
[*P*]{.emphasis} to be [*p*]{.emphasis}. In a causal notation,
[*X*]{.emphasis} usually stands for observed features, while
[*P*]{.emphasis} stands for the policy features that we can manipulate.
This notation can be a bit confusing, as [*p*]{.emphasis} now expresses
both a probability and a policy. Yet, it is important to distinguish
between observed and influenced features. So, if you see
[![](6_files/B10354_09_013.jpg)]{.inlinemediaobject}, [*p*]{.emphasis}
is a feature that is influenced, and if you see
[![](6_files/B10354_09_014.jpg)]{.inlinemediaobject}, [*p*]{.emphasis}
is a probability function.

So, the formula [![](6_files/B10354_09_015.jpg)]{.inlinemediaobject}
expresses the statistical relationship that insurance holders are more
reckless on average. This is what supervised models learn.
[![](6_files/B10354_09_016.jpg)]{.inlinemediaobject} expresses the
causal relationship that people who get insurance become more reckless
because they are insured.

Causal models are a great tool for fair learning. If we only build our
models in a causal way, then we\'ll avoid most of the statistical
discrimination that occurs in statistical models. Do females
statistically earn less than males? Yes. Do females earn less because
they are females and females are somehow undeserving of high salaries?
No. Instead, the earnings difference is caused by other factors, such as
different jobs being offered to males and females, discrimination in the
workplace, cultural stereotypes, and so on.

That does not mean we have to throw statistical models out of the
window. They are great for the many cases where causality is not as much
of an important factor and where we do not intend to set the values of
[*X*]{.emphasis}. For instance, if we are creating a natural language
model, then we are not interested in whether the[]{#id745 .indexterm}
occurrence of a word caused the sentence to be about a certain topic.
Knowing that the topic and the word are related is enough to make
predictions about the content of the text.

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch09lvl2sec123}Obtaining causal models {#obtaining-causal-models .title}

</div>

</div>
:::

The golden route to[]{#id746 .indexterm} obtaining information about
[![](6_files/B10354_09_017.jpg)]{.inlinemediaobject} is to actually go
and manipulate the policy, [*P,*]{.emphasis} in a randomized control
trial. Many websites, for instance, measure the impact of different ads
by showing different ads to different customers, a process known as A/B
testing. Equally, a trader might choose different routes to market to
figure out which one is the best. Yet, it\'s not always possible or even
ethical to do an A/B test. For instance, in our focus on finance, a bank
cannot deny a loan with the explanation, \"Sorry, but you are the
control group.\"

Yet, often causal inference can be made without the need for an A/B
test. Using do-calculus, we can infer the effect of our policy on our
outcome. Take the example of us wondering whether giving people
insurance makes them reckless; the applicant\'s moral hazard, if you
will. Given features [*X*]{.emphasis} and a policy, [*P*]{.emphasis}, we
want to predict the outcome distribution,
[![](6_files/B10354_09_018.jpg)]{.inlinemediaobject}.

In this case, given observed information about the applicant, such as
their age or history of risky behavior, we want to predict the
probability of the applicant behaving recklessly,
[![](6_files/B10354_09_019.jpg)]{.inlinemediaobject}, given that we
manipulate the policy, [*P,*]{.emphasis} of granting insurance. The
observed features often end up influencing both the policy and the
response. An applicant with a high-risk appetite might, for example, not
be given insurance, but might also be more likely to behave recklessly.

Additionally, we have to deal with unobserved, confounding variables,
[*e,*]{.emphasis} which often influence both policy and response. A
prominent media article titled [*Freestyle skiing is safe, and you
should not get insurance*]{.emphasis}, for example, would reduce the
number of people taking insurance as well as the number of reckless
skiers.
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch09lvl2sec124}Instrument variables {#instrument-variables .title}

</div>

</div>
:::

To distinguish the influence on []{#id747 .indexterm}policy and
response, we need access to an [**instrument, Z**]{.strong}. An
instrument is a variable that influences the policy, but nothing else.
The reinsurance cost, for example, could prompt the insurance company to
give out fewer insurance policies. This relationship can be seen in the
flowchart below, where the relationship has been mapped:

::: {.mediaobject}
![](6_files/B10354_09_08.jpg)

::: {.caption}
Causal flowchart
:::
:::

The field of []{#id748 .indexterm}econometrics already has a built a
method to work with these kinds of situations called [**instrumental
variables two-stage least squares**]{.strong} ([**IV2SLS,**]{.strong} or
just [**2SLS**]{.strong}). In a nutshell, 2SLS first fits a linear
regression model between the instrument, [*z,*]{.emphasis} and the
policy, [*p*]{.emphasis}, which in econometrics called the endogenous or
treatment variable.

From this linear[]{#id749 .indexterm} regression, it then estimates an
\"adjusted treatment variable,\" which is the treatment variable as it
can be explained by the instrument. The idea is that this adjustment
removes the influence of all other factors on the treatment. A second
linear regression model then creates a linear model mapping from the
features, [*x,*]{.emphasis} and the adjusted treatment variable,
[![](6_files/B10354_09_020.jpg)]{.inlinemediaobject}, to the outcome,
[*y*]{.emphasis}.

In the following diagram, you can see an overview of how 2SLS works:

::: {.mediaobject}
![](6_files/B10354_09_09.jpg)

::: {.caption}
IV2SLS
:::
:::

2SLS is probably []{#id750 .indexterm}what the insurance company in our
case would use since it is an established method. We won\'t go into
details here, beyond giving you a brief overview of how to use 2SLS in
Python. The `linear model`{.literal} package in Python features an easy
way to run 2SLS.

::: {.note style="margin-left: 0.5in; margin-right: 0.5in;"}
### []{#note50}Note {#note .title}

[**Note**]{.strong}: You can find the package on GitHub at
<https://github.com/bashtage/linearmodels>.
:::

You can install the package by running:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
pip install linearmodels
```
:::

If you have data `X`{.literal}, `y`{.literal}, `P`{.literal}, and
`Z`{.literal}, you can run a 2SLS regression as follows:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
from linearmodels.iv import IV2SLS
iv = IV2SLS(dependent=y,exog=X,endog=P],instruments=Z).fit(cov_type='unadjusted')
```
:::
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch09lvl2sec125}Non-linear causal models {#non-linear-causal-models .title}

</div>

</div>
:::

What if the relationships[]{#id751 .indexterm} between features, the
treatment, and the outcome[]{#id752 .indexterm} are complex and
non-linear? In this case, we need to perform a process similar to 2SLS,
but with a non-linear model, such as a neural network, instead of linear
regression.

Ignoring the confounding variables for a minute, function
[*g*]{.emphasis} determines the recklessness of behavior
[*y*]{.emphasis} given insurance policy
[![](6_files/B10354_09_022.jpg)]{.inlinemediaobject} and a set of
applicant\'s features, [*x*]{.emphasis}:

::: {.mediaobject}
![](6_files/B10354_09_023.jpg)
:::

Function [*f*]{.emphasis} determines policy
[![](6_files/B10354_09_024.jpg)]{.inlinemediaobject} given the
applicant\'s features, [*x,*]{.emphasis} as well as the instrument,
[*z*]{.emphasis}:

::: {.mediaobject}
![](6_files/B10354_09_025.jpg)
:::

Given these two functions, the following identity holds, if the
confounding variable has a mean of zero overall features:

::: {.mediaobject}
![](6_files/B10354_09_026.jpg)
:::

This means that if we[]{#id753 .indexterm} can reliably estimate the
function, [*g,*]{.emphasis} and distribution, [*F*]{.emphasis}, we can
make causal statements about the effects of policy
[![](6_files/B10354_09_027.jpg)]{.inlinemediaobject}. If we have
[]{#id754 .indexterm}data about the actual outcome, [*y*]{.emphasis},
features [*x*]{.emphasis}, policy
[![](6_files/B10354_09_028.jpg)]{.inlinemediaobject}, and instrument
[*z*]{.emphasis}, we can optimize the following:

::: {.mediaobject}
![](6_files/B10354_09_029.jpg)
:::

The preceding function is the squared error between the predicted
outcome using the prediction function, [*g,*]{.emphasis} and the actual
outcome, [*y*]{.emphasis}.

Notice the similarity to 2SLS. In 2SLS, we estimated [*F*]{.emphasis}
and [*g*]{.emphasis} with two separate linear regressions. For[]{#id755
.indexterm} the more complex functions, we can also estimate them with
two separate neural networks. Back in 2017, Jason Hartfort and others
presented just such an approach with their paper, [*Deep IV: A Flexible
Approach for Counterfactual Prediction*]{.emphasis}, - available at:
<http://proceedings.mlr.press/v70/hartford17a/hartford17a.pdf> - the
overview of which you can see in the following diagram:

::: {.mediaobject}
![](6_files/B10354_09_10.jpg)

::: {.caption}
Deep IV
:::
:::

The idea of Deep IV is[]{#id756 .indexterm} to first train a neural
network to express a distribution, [*F(z,x),*]{.emphasis} which
describes the distribution of policies given certain features,
[*x*]{.emphasis}, and instrument values, [*z*]{.emphasis}. A second
neural network is predicting the response, [*y*]{.emphasis}, from the
estimated policy distribution and features. Deep IV\'s advantage is that
it can learn complex, non-linear relationships from complex data,
such[]{#id757 .indexterm} as text.

The authors of the [*Deep IV*]{.emphasis} paper have also published a
custom Keras model which is used for handling sampling and learning from
a distribution part, which you can find on GitHub:
<https://github.com/jhartford/DeepIV>.

While their code is too long to be discussed in depth here, it is
interesting to think about what the source of our causal claim is, both
in Deep IV and IV2SLS. In our insurance case, we assumed that either
having or not having an insurance would influence behavior, not the
other way around. We never showed or tested the truth behind this
direction of causality.

In our case, assuming that insurance influences behavior is justified
because the insurance contract is signed before the behavior is
observed. However, the direction of causality is not always as
straightforward. There is no way to establish the direction of causality
other than logical[]{#id758 .indexterm} reasoning or experiments. In the
absence of experiments, we have to assume and logically reason, for
example, through the sequence of events. Another important assumption
that we make is that the instrument is actually an independent
instrument. If it is not independent, our estimation of the policy will
break down.

With these two limitations in mind, causal inference becomes a great
tool and an active area of research from which we can hope to see great
results in the future. In the best case, your discrimination-sensitive
models would only contain causal variables. In practice, this is usually
not possible. However, keeping the difference between statistical
correlation in mind, as expressed by standard statistical models
and causation, can help you avoid statistical biases and wrong
associations.

A final, more technical, method[]{#id759 .indexterm} to reduce
unfairness is to peek inside the model to ensure it is fair. We already
looked at interpretability in the last chapter, mostly to debug data and
spot overfitting, but now, we will give it another look, this time to
justify the model\'s predictions.



[]{#ch09lvl1sec110}Interpreting models to ensure fairness {#interpreting-models-to-ensure-fairness .title style="clear: both"}
---------------------------------------------------------

</div>

</div>

------------------------------------------------------------------------
:::

In [Chapter
8](https://subscription.packtpub.com/book/data/9781789136364/8){.link},
[*Privacy, Debugging, and Launching Your Products,*]{.emphasis} we
discussed model interpretability as a debugging method. We []{#id760
.indexterm}used LIME to spot the features that the model is overfitting
to.

In this section, we will use a slightly more[]{#id761 .indexterm}
sophisticated method called [**SHAP**]{.strong} ([**SHapley Additive
exPlanation**]{.strong}). SHAP combines several different explanation
approaches into one neat method. This method lets us generate
explanations for individual predictions as well as for entire datasets
in order to understand the model better.

You can find SHAP on GitHub at <https://github.com/slundberg/shap> and
install it locally with `pip install shap`{.literal}. Kaggle kernels
have SHAP preinstalled.

::: {.note style="margin-left: 0.5in; margin-right: 0.5in;"}
### []{#tip05}Note {#note .title}

The example code given here is from the SHAP example notebooks. You can
find a slightly extended version of the notebook on Kaggle:

<https://www.kaggle.com/jannesklaas/explaining-income-classification-with-keras>
:::

SHAP combines seven model interpretation methods, those being LIME,
Shapley sampling values, DeepLIFT, [**Quantitative Input
Influence**]{.strong} ([**QII**]{.strong}), layer-wise relevance
propagation, Shapley regression values, and a tree interpreter that has
two modules: a model-agnostic `KernelExplainer`{.literal} and a
`TreeExplainer`{.literal} module specifically for tree-based methods
such as `XGBoost`{.literal}.

The mathematics of how and when the interpreters are used is not
terribly relevant for using SHAP. In a[]{#id762 .indexterm} nutshell,
given a function, [*f*]{.emphasis}, expressed through a neural network,
for instance, and a data point, [*x*]{.emphasis}, SHAP compares
[![](7_files/B10354_09_030.jpg)]{.inlinemediaobject} to
[![](7_files/B10354_09_031.jpg)]{.inlinemediaobject} where
[![](7_files/B10354_09_032.jpg)]{.inlinemediaobject} is the \"expected
normal output\" generated for a larger sample. SHAP will then create
smaller models, similar to LIME, to see which features explain the
difference between [![](7_files/B10354_09_033.jpg)]{.inlinemediaobject}
and [![](7_files/B10354_09_034.jpg)]{.inlinemediaobject}.

In our loan example, this corresponds to having an applicant,
[*x,*]{.emphasis} and a distribution of many applicants,
[*z*]{.emphasis}, and trying to explain why the chance of getting a loan
for applicant [*x*]{.emphasis} is different from the expected chance for
the other applicants, [*z*]{.emphasis}.

SHAP does not only compare
[![](7_files/B10354_09_035.jpg)]{.inlinemediaobject} and
[![](7_files/B10354_09_036.jpg)]{.inlinemediaobject}, but also compares
[![](7_files/B10354_09_037.jpg)]{.inlinemediaobject} to
[![](7_files/B10354_09_038.jpg)]{.inlinemediaobject}.

This means it compares the importance of certain features that are held
constant, which allows it to better estimate the interactions between
features.

Explaining a single prediction can very important, especially in the
world of finance. Your customers might ask you, \"Why did you deny me a
loan?\" You\'ll remember from earlier on that the ECOA act stipulates
that you must give the customer a valid reason, and if you have no good
explanation, you might find yourself in a tough situation. In this
example, we are once again working with the income prediction dataset,
with the objective of explaining why our model made a single decision.
This process works in three steps.

Firstly, we need to define the explainer and provide it with a
prediction method and values, [*z*]{.emphasis}, to estimate a \"normal
outcome.\" Here we are using a wrapper, `f`{.literal}, for Keras\'
prediction function, which makes working with SHAP much easier. We
provide 100 rows of the dataset as values for `z`{.literal}:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
explainer = shap.KernelExplainer(f, X.iloc[:100,:])
```
:::

Next, we need to calculate the SHAP values indicating the importance of
different features for a single example. We let SHAP create 500
permutations of each sample from [*z*]{.emphasis} so that SHAP has a
total of 50,000 examples to compare the one example to:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
shap_values = explainer.shap_values(X.iloc[350,:], nsamples=500)
```
:::

Finally, we can plot the influence of the features with SHAP\'s own
plotting tool. This time, we provide[]{#id763 .indexterm} a row from
`X_display`{.literal}, not `X`{.literal}. `X_display`{.literal}, which
contains the unscaled values and is only used for annotation of the plot
to make it easier to read:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
shap.force_plot(explainer.expected_value, shap_values)
```
:::

We can see the output of the code in the following graph:

::: {.mediaobject}
![](7_files/B10354_09_11.jpg)

::: {.caption}
The influence of features with the SHAP plotting tool
:::
:::

If you look at the preceding plot, the predictions of the model seem, by
and large, reasonable. The model gives the applicant a high chance of
having a high income because they have a master\'s degree, and because
they\'re an executive manager who works 65 hours a week. The applicant
could have an even higher expected income score were it not for a
capital loss. Likewise, the model seems to take the fact that the
applicant is married as a big factor of a high income. In fact, in our
example, it seems that marriage is more important than either the long
hours or the job title.

Our model also has some problems that become clear once we calculate and
plot the SHAP values of another applicant:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
shap_values = explainer.shap_values(X.iloc[167,:], nsamples=500)
shap.force_plot(explainer.expected_value, shap_values)
```
:::

The following outputted graph is then shown. This also shows some of the
problems that we\'ve encountered:

::: {.mediaobject}
![](7_files/B10354_09_12.jpg)

::: {.caption}
The SHAP values showing some of the problems we can encounter
:::
:::

In this example, the applicant also has a good education, and works 48
hours a week in the technology industry, but the model gives her a much
lower chance of having a high income because of the fact that she\'s a
female, an Asian-Pacific islander who has never been married and has no
other family relationship. A loan rejection on these grounds is a
lawsuit waiting to happen as per the ECOA act.

The two individual cases that we just looked at might have been
unfortunate glitches by the model. It might have overfitted to some
strange combination that gave an undue importance to marriage. To
investigate whether our[]{#id764 .indexterm} model is biased, we should
investigate a number of different predictions. Fortunately for us, the
SHAP library has a number of tools that can do just that.

We can use the SHAP value calculations for multiple rows:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
shap_values = explainer.shap_values(X.iloc[100:330,:], nsamples=500)
```
:::

Then, we can plot a forced plot for all of these values as well:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
shap.force_plot(explainer.expected_value, shap_values)
```
:::

Again, this code produces a SHAP dataset graph, which we can see in the
following graphic:

::: {.mediaobject}
![](7_files/B10354_09_13.jpg)

::: {.caption}
SHAP dataset
:::
:::

The preceding plot shows 230 rows of the dataset, grouped by similarity
with the forces of each feature that matter to them. In your live
version, if you move the mouse over the graph, you\'ll be able to read
the features and their values.

By exploring this graph, you can get an idea of what kind of people the
model classifies as either high or low earners. On the very left, for
example, you\'ll see most people with low education who work as
cleaners. The big red block between 40 and 60 are mostly highly educated
people who work a high number of hours.

To further examine the impact of marital status, you can change what
SHAP displays on the [*y*]{.emphasis}-axis. Let\'s look at the impact of
marriage:

::: {.mediaobject}
![](7_files/B10354_09_14.jpg)

::: {.caption}
SHAP marriage outcome
:::
:::

As you can see[]{#id765 .indexterm} in this chart, marriage status
either strongly positively or negatively impacts people from different
groups. If you move your mouse over the chart, you can see that the
positive influences all stem from civic marriages.

Using a summary plot, we can see which features matter the most to our
model:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
shap.summary_plot(shap_values, X.iloc[100:330,:])
```
:::

This code then outputs the final summary plot graph, which we can see
below:

::: {.mediaobject}
![](7_files/B10354_09_15.jpg)

::: {.caption}
SHAP summary plot
:::
:::

As you can see, education is the most important influence on our model.
It also has the widest spread of influence. Low education levels really
drag predictions down, while strong education levels really boost
predictions up. Marital status is the second most important predictor.
Interestingly, though, capital losses are important to[]{#id766
.indexterm} the model, but capital gains are not.

To dig deeper into the effects of marriage, we have one more tool at our
disposal, a dependence plot, which can show the SHAP values of an
individual feature together with a feature for which SHAP suspects high
interaction. With the following code snippet, we can inspect the effect
of marriage on our model\'s predictions:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
shap.dependence_plot("marital-status", shap_values, X.iloc[100:330,:], display_features=X_display.iloc[100:330,:])
```
:::

As a result of running this code, we can now see a visualized
representation of the effect of marriage in the following graph:

::: {.mediaobject}
![](7_files/B10354_09_16.jpg)

::: {.caption}
SHAP marriage dependence
:::
:::

As you can see, [**Married-civ-spouse**]{.strong}, the census code for a
civilian marriage with no partner in the armed forces,[]{#id767
.indexterm} stands out with a positive influence on model outcomes.
Meanwhile, every other type of arrangement has slightly negative scores,
especially never married.

Statistically, rich people tend to stay married for longer, and younger
people are more likely to have never been married. Our model correctly
correlated that marriage goes hand in hand with high income, but not
because marriage causes high income. The model is correct in making the
correlation, but it would be false to make decisions based on the model.
By selecting, we effectively[]{#id768 .indexterm} manipulate the
features on which we select. We are no longer interested in just
[![](7_files/B10354_09_039.jpg)]{.inlinemediaobject}, but in
[![](7_files/B10354_09_040.jpg)]{.inlinemediaobject}.



[]{#ch09lvl1sec111}Unfairness as complex system failure {#unfairness-as-complex-system-failure .title style="clear: both"}
-------------------------------------------------------

</div>

</div>

------------------------------------------------------------------------
:::

In this chapter, you []{#id769 .indexterm}have been equipped with an
arsenal of technical tools to make machine learning models fairer.
However, a model does not operate in a vacuum. Models are embedded in
complex socio-technical systems. There are humans developing and
monitoring the model, sourcing the data and creating the rules for what
to do with the model output. There are also other []{#id770
.indexterm}machines in place, producing data or using outputs from the
model. Different players might try to game the system in different ways.

Unfairness is equally complex. We\'ve already discussed the two general
definitions of unfairness, [*disparate impact*]{.emphasis} and
[*disparate treatment*]{.emphasis}. Disparate treatment can occur
against any combination of features (age, gender, race, nationality,
income, and so on), often in complex and non-linear ways. This section
examines Richard Cook\'s 1998 paper, [*How complex systems
fail*]{.emphasis} - available at
<https://web.mit.edu/2.75/resources/random/How%20Complex%20Systems%20Fail.pdf>
- which looks at how complex machine learning-driven systems fail to be
fair. Cook lists 18 points, some of which will be discussed in the
following sections.

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch09lvl3sec30}Complex systems are intrinsically hazardous systems {#complex-systems-are-intrinsically-hazardous-systems .title}

</div>

</div>
:::

Systems are []{#id771 .indexterm}usually complex because they are
hazardous, and many safeguards have been created because of that fact.
The financial system is a hazardous system; if it goes off the rails, it
can break the economy or ruin people\'s lives. Thus, many regulations
have been created and many players in the market work to make the system
safer.

Since the financial system is so hazardous, it is important to make sure
it is safe against unfairness, too. Luckily, there are a number of
safeguards in place to keep the system fair. Naturally, these safeguards
can break, and they do so constantly in a number of small ways.
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch09lvl3sec31}Catastrophes are caused by multiple failures {#catastrophes-are-caused-by-multiple-failures .title}

</div>

</div>
:::

In a complex system, no single point of failure can cause catastrophes
since there are many safeguards in place. Failure[]{#id772 .indexterm}
usually results from multiple points of failure. In the financial
crises, banks created risky products, but regulators didn\'t stop them.

For widespread discrimination to happen, not only does the model have to
make unfair predictions, but employees must blindly follow the model and
criticism must be suppressed. On the flip side, just fixing your model
will not magically keep all unfairness away. The procedures and culture
inside and outside the firm can also cause discrimination, even with a
fair model.
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch09lvl3sec32}Complex systems run in degraded mode {#complex-systems-run-in-degraded-mode .title}

</div>

</div>
:::

In most accident reports, there is a section that lists
\"proto-accidents,\" which are instances in the past where the same
accident nearly happened but did not happen. The model might have made
erratic predictions before, but a[]{#id773 .indexterm} human operator
stepped in, for example.

It is important to know that in a complex system, failures that nearly
lead to catastrophe always occur. The complexity of the system makes it
prone to error, but the heavy safeguards against catastrophe keep them
from happening. However, once these[]{#id774 .indexterm} safeguards
fail, catastrophe is right around the corner. Even if your system seems
to run smoothly, check for proto-accidents and strange behavior before
it is too late.
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch09lvl3sec33}Human operators both cause and prevent accidents {#human-operators-both-cause-and-prevent-accidents .title}

</div>

</div>
:::

Once things have gone wrong, blame is often put at the human operators
who \"must have known\" that their behavior would \"inevitably\" lead to
an accident. On the other hand, it is usually humans who step in at the
last minute to prevent accidents from happening. Counterintuitively, it
is rarely one human and one action that causes the accident, but the
behavior of many humans over many actions. For models to be fair, the
entire team has to work to keep it fair.
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch09lvl3sec34}Accident-free operation requires experience with failure {#accident-free-operation-requires-experience-with-failure .title}

</div>

</div>
:::

In fairness, the single biggest problem is often that the designers of a
system do not experience the[]{#id775 .indexterm} system discriminating
against them. It is thus important to get the insights of a diverse
group of people into the development process. Since your system
constantly fails, you should capture the learning from these small
failures before bigger accidents happen.



[]{#ch09lvl1sec112}A checklist for developing fair models {#a-checklist-for-developing-fair-models .title style="clear: both"}
---------------------------------------------------------

</div>

</div>

------------------------------------------------------------------------
:::

With the preceding[]{#id776 .indexterm} information, we can create a
short checklist that can be used when creating fair models. Each issue
comes with several sub-issues.

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch09lvl3sec35}What is the goal of the model developers? {#what-is-the-goal-of-the-model-developers .title}

</div>

</div>
:::

::: {.itemizedlist}
-   Is fairness an explicit goal?

-   Is the model evaluation metric chosen to reflect the fairness of the
    model?

-   How do model developers get promoted and rewarded?

-   How does the model influence business results?

-   Would the model discriminate against the developer\'s demographic?

-   How diverse is the development team?

-   Who is responsible when things go wrong?
:::
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch09lvl3sec36}Is the data biased? {#is-the-data-biased .title}

</div>

</div>
:::

::: {.itemizedlist}
-   How was the data collected?

-   Are there statistical misrepresentations in the sample?

-   Are sample sizes for minorities adequate?

-   Are sensitive variables included?

-   Can sensitive []{#id777 .indexterm}variables be inferred from the
    data?

-   Are there interactions between features that might only affect
    subgroups?
:::
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch09lvl3sec37}Are errors biased? {#are-errors-biased .title}

</div>

</div>
:::

::: {.itemizedlist}
-   What are the error rates for different subgroups?

-   What is the error rate of a simple, rule-based alternative?

-   How do the errors in the model lead to different outcomes?
:::
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch09lvl3sec38}How is feedback incorporated? {#how-is-feedback-incorporated .title}

</div>

</div>
:::

::: {.itemizedlist}
-   Is there an appeals/reporting process?

-   Can mistakes be attributed back to a model?

-   Do model developers get insight into what happens with their
    model\'s predictions?

-   Can the model be audited?

-   Is the model open source?

-   Do people know which features are used to make predictions about
    them?
:::
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch09lvl3sec39}Can the model be interpreted? {#can-the-model-be-interpreted .title}

</div>

</div>
:::

::: {.itemizedlist}
-   Is a model interpretation, for example, individual results, in
    place?

-   Can the interpretation be understood by those it matters to?

-   Can findings from the interpretation lead to changes in the model?
:::
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch09lvl3sec40}What happens to models after deployment? {#what-happens-to-models-after-deployment .title}

</div>

</div>
:::

::: {.itemizedlist}
-   Is there a central repository to keep track of all the models
    deployed?

-   Are input assumptions checked continuously?

-   Are accuracy and fairness metrics monitored continuously?



[]{#ch09lvl1sec113}Exercises {#exercises .title style="clear: both"}
----------------------------

</div>

</div>

------------------------------------------------------------------------
:::

In this chapter, you have learned a lot about both the technical and
non-technical considerations of fairness in machine learning. These
exercises will help you think much more deeply about the topic:

::: {.itemizedlist}
-   Think about the organization you work for. How is fairness
    incorporated in your organization? What works well and what could be
    improved?

-   Revisit any of the models developed in this book. Are they fair? How
    would you test them for fairness?

-   Fairness is only one of the many complex issues large models can
    have. Can you think of an issue in your area of work that could be
    tackled with the tools discussed in this chapter?


[]{#ch09lvl1sec114}Summary {#summary .title style="clear: both"}
--------------------------

</div>

</div>

------------------------------------------------------------------------
:::

In this chapter, you have learned about fairness in machine learning in
different aspects. First, we discussed legal definitions of fairness and
quantitative ways to measure these definitions. We then discussed
technical methods to train models to meet fairness criteria. We also
discussed causal models. We learned about SHAP as a powerful tool to
interpret models and find unfairness in a model. Finally, we learned how
fairness is a complex systems issue and how lessons from complex systems
management can be applied to make models fair.

There is no guarantee that following all the steps outlined here will
make your model fair, but these tools vastly increase your chances of
creating a fair model. Remember that models in finance operate in
high-stakes environments and need to meet many regulatory demands. If
you fail to do so, damage could be severe.

In the next, and final, chapter of this book, we will be looking at
probabilistic programming and Bayesian inference.
