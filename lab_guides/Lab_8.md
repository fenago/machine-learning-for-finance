

[]{#ch08}Chapter 8. Privacy, Debugging, and Launching Your Products {#chapter-8.-privacy-debugging-and-launching-your-products .title}
-------------------------------------------------------------------

</div>

</div>
:::

Over the course of the last seven chapters we\'ve developed a large
toolbox of machine learning algorithms that we could use for machine
learning problems in finance. To help round-off this toolbox, we\'re now
going to look at what you can do if your algorithms don\'t work.

Machine learning models fail in the worst way: silently. In traditional
software, a mistake usually leads to the program crashing, and while
they\'re annoying for the user, they are helpful for the programmer. At
least it\'s clear that the code failed, and often the developer will
find an accompanying crash report that describes what went wrong. Yet as
you go beyond this book and start developing your own models, you\'ll
sometimes encounter machine learning code crashes too, which, for
example, could be caused if the data that you fed into the algorithm had
the wrong format or shape.

These issues can usually be debugged by carefully tracking which shape
the data had at what point. More often, however, models that fail just
output poor predictions. They\'ll give no signal that they have failed,
to the point that you might not even be aware that they\'ve even failed
at all, but at other times, the model might not train well, it
won\'t converge, or it won\'t achieve a low loss rate.

In this chapter, we\'ll be focusing on how you debug these silent
failures so that they don\'t impact the machine learning algorithms that
you\'ve created. This will include looking at the following subject
areas:

::: {.itemizedlist}
-   Finding flaws in your data that lead to flaws in your learned model

-   Using creative tricks to make your model learn more from less data

-   Unit testing data in production or training to ensure standards are
    met

-   Being mindful of privacy and regulation, such as GDPR

-   Preparing data for training and avoiding common pitfalls

-   Inspecting the model and peering into the \"black box\"

-   Finding optimal hyperparameters

-   Scheduling learning rates in order to reduce overfitting

-   Monitoring training progress with TensorBoard

-   Deploying machine learning products and iterating on them

-   Speeding up training and inference
:::

The first step you must take, before even attempting to debug your
program, is to acknowledge that even good machine learning engineers
fail frequently. There are many reasons why machine learning projects
fail, and most have nothing to do with the skills of the engineers, so
don\'t think that just because it\'s not working, you\'re at fault.

If these bugs are spotted early enough, then both time and money can be
saved. Furthermore, in high-stakes environments, including finance-based
situations, such as trading, engineers that are aware can pull the plug
when they notice their model is failing. This should not be seen as
a failure, but as a success to avoid problems.



[]{#ch08lvl1sec99}Debugging data {#debugging-data .title style="clear: both"}
--------------------------------

</div>

</div>

------------------------------------------------------------------------
:::

You\'ll remember that back in the[]{#id624 .indexterm} first chapter of
this book, we discussed how machine learning models are a function of
their training data, meaning that, for example, bad data will lead to
bad models, or as we put it, garbage in, garbage out. If your project is
failing, your data is the most likely culprit. Therefore, in this
chapter we will start by looking at the data first, before moving on to
look at the other possible issues that might cause our model to crash.

However, even if you have a working model, the real-world data coming in
might not be up to the task. In this section, we will learn how to find
out whether you have good data, what to do if you have not been given
enough data, and how to test your data.

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch08lvl2sec103}How to find out whether your data is up to the task {#how-to-find-out-whether-your-data-is-up-to-the-task .title}

</div>

</div>
:::

There are two aspects to consider when wanting to know whether your data
is up to the task of training a good model:

::: {.itemizedlist}
-   Does the data predict what you want it to predict?

-   Do you have enough data?
:::

To find out whether your model does contain predicting information, also
called a signal, you could ask yourself the question, could a human make
a prediction given this data? It\'s important for your AI to be given
data that can be comprehended by humans, because after all, the only
reason we know intelligence is possible is because we observe it in
humans. Humans are good at understanding written text, but if a human
cannot understand a text, then the chances are that your model won\'t
make much sense of it either.

A common pitfall to this test is that humans have context that your
model does not have. A human trader does not only consume financial
data, but they might have also experienced the product of a company or
seen the CEO on TV. This external context flows into the trader\'s
decision but is often forgotten when a model is built. Likewise, humans
are also good at focusing on important data. A human trader will not
consume all of the financial data out there because most of it is
irrelevant.

Adding more inputs to your model won\'t make it better; on the contrary,
it often makes it worse, as the model overfits and gets distracted by
all the noise. On the other hand, humans are irrational; they follow
peer pressure and have a hard time making decisions in abstract and
unfamiliar environments. Humans would struggle to find an optimal
traffic light policy, for instance, because the data that traffic lights
operate on is not intuitive to us.

This brings[]{#id625 .indexterm} us to the second sanity check: a human
might not be able to make predictions, but there might be a causal
(economic) rationale. There is a causal link between a company\'s
profits and its share price, the traffic on a road and traffic jams,
customer complaints and customers leaving your company, and so on. While
humans might not have an intuitive grasp of these links, we can discover
them through reasoning.

There are some tasks for which a causal link is required. For instance,
for a long time, many quantitative trading firms insisted on their data
having a causal link to the predicted outcomes of models. Yet nowadays,
the industry seems to have slightly moved away from that idea as it gets
more confident in testing its algorithms. If humans cannot make a
prediction and there is no causal rationale for why your data is
predictive, you might want to reconsider whether your project is
feasible.

Once you have determined that your data contains enough signal, you need
to ask yourself whether you have enough data to train a model to extract
the signal. There is no clear answer to the question of how much is
enough, but roughly speaking, the amount needed depends on the
complexity of the model you hope to create. There are a couple of rules
of thumb to follow, however:

::: {.itemizedlist}
-   For classification, you should have []{#id626 .indexterm}around 30
    independent samples per class.

-   You should have 10 times as many samples as there are features,
    especially for structured data problems.

-   Your dataset should get bigger as the number of parameters in your
    model gets bigger.
:::

Keep in mind these rules are only rules of thumb and might be very
different for your specific application. If you can make use of transfer
learning, then you can drastically reduce the number of samples you
need. This is why most computer vision applications use transfer
learning.

If you have any reasonable amount of data, say, a few hundred samples,
then you can start building your model. In this case, a sensible
suggestion would be to start with a simple model that you can deploy
while you collect more data.
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch08lvl2sec104}What to do if you don\'t have enough data {#what-to-do-if-you-dont-have-enough-data .title}

</div>

</div>
:::

Sometimes, you find yourself in a situation[]{#id627 .indexterm} where
despite starting your project, you simply do not have enough data. For
example, the legal team might have changed its mind and decided that you
cannot use the data, for instance due to GDPR, even though they greenlit
it earlier. In this case, you have multiple options.

Most of the time, one of the best options would be to \"augment your
data.\" We\'ve already seen some data augmentation in [Chapter
3,](https://subscription.packtpub.com/book/data/9781789136364/3){.link}
[*Utilizing Computer Vision*]{.emphasis}. Of course, you can augment all
kinds of data in various ways, including slightly changing some database
entries. Taking augmentation a step further, you might be able to
[*generate your data*]{.emphasis}, for example, in a simulation. This is
effectively how most reinforcement learning researchers gather data, but
this can also work in other cases.

The data we used for fraud detection back in [Chapter
2](https://subscription.packtpub.com/book/data/9781789136364/2){.link},
[*Applying Machine Learning to Structured Data*]{.emphasis} was obtained
from simulation. The simulation requires you to be able to write down
the rules of your environment within a program. Powerful learning
algorithms tend to figure out these often over-simplistic rules, so they
might not generalize to the real world as well. Yet, simulated data can
be a powerful addition to real data.

Likewise, you can often [*find external data*]{.emphasis}. Just because
you haven\'t tracked a certain data point, it does not mean that nobody
else has. There is an astonishing amount of data available on the
internet. Even if the data was not originally collected for your
purpose, you might be able to retool data by either relabeling it or by
[]{#id628 .indexterm}using it for [**transfer learning**]{.strong}. You
might be able to train a model on a large dataset for a different task
and then use that model as a basis for your task. Equally, you can find
a model that someone else has trained for a different task and repurpose
it for your task.

Finally, you might be able to[]{#id629 .indexterm} create a [**simple
model**]{.strong}, which does not capture the relationship in the data
completely but is enough to ship a product. Random forests and other
tree-based methods often require much less data than neural networks.

It\'s important to remember that for data, quality trumps quantity in
the majority of cases. Getting a small, high-quality dataset in and
training a weak model is often your best shot to find problems with data
early. You can always scale up data collection later. A mistake many
practitioners make is that they spend huge amounts of time and money on
getting a big dataset, only to find that they have the wrong kind of
data for their project.
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch08lvl2sec105}Unit testing data {#unit-testing-data .title}

</div>

</div>
:::

If you build a model, you\'re making[]{#id630 .indexterm} assumptions
about your data. For example, you assume that the data you feed into
your time series model is actually a time series with dates that follow
each other in order. You need to test your data to make sure that this
assumption is true. This is something that is especially true with live
data that you receive once your model is already in production. Bad data
might lead to poor model performance, which can be dangerous, especially
in a high-stakes environment.

Additionally, you need to test whether your data is clean from things
such as personal information. As we\'ll see in the following section on
privacy, personal information is a liability that you want to get rid
of, unless you have good reasons and consent from the user to use it.

Since monitoring data quality[]{#id631 .indexterm} is important when
trading based on many data sources, Two Sigma Investments LP, a New York
City-based international hedge fund, has created an open source library
for data monitoring. It is called [*marbles*]{.emphasis}, and []{#id632
.indexterm}you can read more about it here:
<https://github.com/twosigma/marbles>. marbles builds on Python\'s
`unittest`{.literal} library.

You can install it with the following command:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
pip install marbles
```
:::

::: {.note style="margin-left: 0.5in; margin-right: 0.5in;"}
### []{#note36}Note {#note .title}

[**Note**]{.strong}: You can find a Kaggle kernel[]{#id633 .indexterm}
demonstrating marbles here:
<https://www.kaggle.com/jannesklaas/marbles-test>.
:::

The following code sample shows a simple marbles unit test. Imagine you
are gathering data about the unemployment rate in Ireland. For your
models to work, you need to ensure that you actually get the data for
consecutive months, and don\'t count one month twice, for instance.

We can ensure this happens by running the following code:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
import marbles.core                                 #1
from marbles.mixins import mixins

import pandas as pd                                 #2
import numpy as np
from datetime import datetime, timedelta

class TimeSeriesTestCase(marbles.core.TestCase,mixins.MonotonicMixins):                            #3
    def setUp(self):                                            #4

        self.df = pd.DataFrame({'dates':[datetime(2018,1,1),datetime(2018,2,1),datetime(2018,2,1)],'ireland_unemployment':[6.2,6.1,6.0]})   #5
        
    
    def tearDown(self):
        self.df = None                                          #6
        
    def test_date_order(self):                                  #7
        
        self.assertMonotonicIncreasing(sequence=self.df.dates,note = 'Dates need to increase monotonically')                                                 #8
```
:::

Don\'t worry if you don\'t fully understand the code. We\'re now going
to go through each stage of the code:

::: {.orderedlist}
1.  Marbles features two[]{#id634 .indexterm} main components. The
    `core`{.literal} module does the actual testing, while the
    `mixins`{.literal} module provides a number of useful tests for
    different types of data. This simplifies your test writing and gives
    you more readable and semantically interpretable tests.

2.  You can use all the libraries, like pandas, that you would usually
    use to handle and process data for testing.

3.  Now it is time to define our test class. A new test class must
    inherit marbles\' `TestCase`{.literal} class. This way, our test
    class is automatically set up to run as a marbles test. If you want
    to use a mixin, you also need to inherit the corresponding mixin
    class.

4.  In this example, we are working with a series of dates that should
    be increasing monotonically. The `MonotonicMixins`{.literal} class
    provides a range of tools that allow you to test for a monotonically
    increasing series automatically.

5.  If you are coming from Java programming, the concept of multiple
    inheritances might strike you as weird, but in Python, classes can
    easily inherit multiple other classes. This is useful if you want
    your class to inherit two different capabilities, such as running a
    test and testing time-related concepts.

6.  The `setUp`{.literal} function is a standard test function in which
    we can load the data and prepare for the test. In this case, we just
    need to define a pandas DataFrame by hand. Alternatively, you could
    also load a CSV file, load a web resource, or pursue any other way
    in order to get your data.

7.  In our DataFrame, we have the Irish unemployment rate for two
    months. As you can see, the last month has been counted twice. As
    this should not happen, it will cause an error.

8.  The `tearDown`{.literal} method is a standard test method that
    allows us to cleanup after our test is done. In this case, we just
    free RAM, but you can also choose to delete files or databases that
    were just created for testing.

9.  Methods describing actual tests should start with `test_`{.literal}.
    marbles will automatically run all of the test methods after setting
    up.

10. We assert that the time indicator of our data strictly increases. If
    our assertion had required intermediate variables, such as a maximum
    value, marbles will display it in the error report. To make our
    error more readable, we can attach a handy note.
:::

To run a unit test in a Jupyter Notebook, we need to tell marbles to
ignore the first argument; we achieve this by running the following:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
if __name__ == '__main__':
    marbles.core.main(argv=['first-arg-is-ignored'], exit=False)
```
:::

It\'s more common to run unit tests directly from the command line. So,
if you saved the preceding code in the command line, you could run it
with this command:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
python -m marbles marbles_test.py
```
:::

Of course, there are problems with our data. Luckily for us, our test
ensures that this error does not get passed on to []{#id635
.indexterm}our model, where it would cause a silent failure in the form
of a bad prediction. Instead, the test will fail with the following
error output:

::: {.note style="margin-left: 0.5in; margin-right: 0.5in;"}
### []{#note37}Note {#note-1 .title}

[**Note**]{.strong}: This code will not run and will fail.
:::

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
F                                                     #1
==================================================================
FAIL: test_date_order (__main__.TimeSeriesTestCase)   #2
------------------------------------------------------------------
marbles.core.marbles.ContextualAssertionError: Elements in 0   2018-01-01
1   2018-02-01
2   2018-02-01                                        #3
Name: dates, dtype: datetime64[ns] are not strictly monotonically increasing

Source (<ipython-input-1-ebdbd8f0d69f>):              #4
     19 
 >   20 self.assertMonotonicIncreasing(sequence=self.df.dates,
     21                           note = 'Dates need to increase monotonically')
     22 
Locals:                                               #5

Note:                                                 #6
    Dates need to increase monotonically


----------------------------------------------------------------------
```
:::

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
Ran 1 test in 0.007s

FAILED (failures=1)
```
:::

So, what exactly caused the data to fail? Let\'s have a look:

::: {.orderedlist}
1.  The top line shows the status of the entire test. In this case,
    there was only one test method, and it failed. Your test might have
    multiple different test methods, and marbles would display the
    progress by showing how tests fail or pass.The next couple
    []{#id636 .indexterm}of lines describe the failed test method. This
    line describes that the `test_date_order`{.literal} method of the
    `TimeSeriesTestCase`{.literal} class failed.

2.  marbles shows precisely how the test failed. The values of the dates
    tested are shown, together with the cause for failure.

3.  In addition to the actual failure, marbles will display a traceback
    showing the actual code where our test failed.

4.  A special feature of marbles is the ability to display local
    variables. This way, we can ensure that there was no problem with
    the setup of the test. It also helps us in getting the context as to
    how exactly the test failed.

5.  Finally, marbles will display our note, which helps the test
    consumer understand what went wrong.

6.  As a summary, marbles displays that the test failed with one
    failure. Sometimes, you may be able to accept data even though it
    failed some tests, but more often than not you\'ll want to dig in
    and see what is going on.
:::

The point of unit testing data is to make the failures loud in order to
prevent data issues from giving you bad predictions. A failure with an
error message is much better than a failure without one. Often, the
failure is caused by your data vendor, and by testing all of the data
that you got from all of the vendors, it will allow you to be aware when
a vendor makes a mistake.

Unit testing data also helps you to ensure you have no data that you
shouldn\'t have, such as personal data. Vendors need to clean datasets
of all personally identifying information, such as social security
numbers, but of course, they sometimes forget. Complying with ever
stricter data privacy regulation is a big concern for many financial
institutions engaging in machine learning.

The next section will therefore discuss how to preserve privacy and
comply with regulations while still gaining benefits from machine
learning.
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch08lvl2sec106}Keeping data private and complying with regulations {#keeping-data-private-and-complying-with-regulations .title}

</div>

</div>
:::

In recent years, consumers []{#id637 .indexterm}have woken up to the
fact that their data is being harvested and analyzed in ways that they
cannot control, and that is sometimes against their own interest.
Naturally, they are not happy about it and regulators have to come
up[]{#id638 .indexterm} with some new data regulations.

At the time of writing, the European Union has introduced the [**General
Data Protection Regulation**]{.strong} ([**GDPR**]{.strong}), but it\'s
likely that other jurisdictions will develop stricter privacy
protections, too.

This text will not go into depth on how to comply with this law
specifically. However, if you wish to expand your understanding of the
topic, then the UK government\'s guide to GDPR is a good starting place
to learn more about the specifics of the regulation and how to comply
with it:
<https://www.gov.uk/government/publications/guide-to-the-general-data-protection-regulation>.

This section will []{#id639 .indexterm}outline both the key principles
of the recent privacy legislation and some technological solutions that
you can utilize in order to comply with these principles.

The overarching rule here is to, \"delete what you don\'t need.\" For a
long time, a large percentage of companies have just stored all of
the data that they could get their hands on, but this is a bad idea.
Storing personal data is a liability for your business. It\'s owned by
someone else, and you are on the hook for taking care of it. The next
time you hear a statement such as, \"We have 500,000 records in our
database,\" think of it more along the lines of, \"We have 500,000
liabilities on our books.\" It can be a good idea to take on
liabilities, but only if there is an economic value that justifies these
liabilities. What happens astonishingly often though is that you might
collect personal data by accident. Say you are tracking device usage,
but accidentally[]{#id640 .indexterm} include the customer ID in your
records. You need practices in place that monitor and prevent such
accidents, here are four of the key ones:

::: {.itemizedlist}
-   [**Be transparent and obtain consent**]{.strong}: Customers want
    good products, and they understand how their data can make your
    product better for them. Rather than pursuing an adversarial
    approach in which you wrap all your practices in a very long
    agreement and then make users agree to it, it is usually more
    sensible to clearly tell users what you are doing, how their data is
    used, and how that improves the product. If you need personal data,
    you need consent. Being transparent will help you down the line as
    users will trust you more and this can then be used to improve your
    product through customer feedback.

-   [**Remember that breaches happen to the best**]{.strong}: No matter
    how good your security is, there is a chance that you\'ll get
    hacked. So, you should design your personal data storage under the
    assumption that the entire database might be dumped on the internet
    one day. This assumption will help you to create stronger privacy
    and help you to avoid disaster once you actually get hacked.

-   [**Be mindful about what can be inferred from data**]{.strong}: You
    might not be tracking personally identifying information in your
    database, but when combined with another database, your customers
    can still be individually identified.

    Say you went for coffee with a friend, paid by credit card, and
    posted a picture of the coffee on Instagram. The bank might collect
    anonymous credit card records, but if someone went to crosscheck the
    credit card records against the Instagram pictures, there would only
    be one customer who bought a coffee and posted a picture of coffee
    at the same time in the same area. This way, all your credit card
    transactions are no longer anonymous. Consumers expect companies to
    be mindful of these effects.

-   [**Encrypt and Obfuscate data**]{.strong}: Apple, for instance,
    collects phone data but adds random noise to the collected data. The
    noise renders each individual record incorrect, but in aggregate the
    records still give a picture of user behavior. There[]{#id641
    .indexterm} are a few caveats to this approach; for example, you can
    only collect so many data points from a user before the noise
    cancels out, and the individual behavior is revealed.

    Noise, as introduced by obfuscation, is random. When averaged over a
    large sample of data about a single user, the mean of the noise will
    be zero as it does not present a pattern by itself. The true profile
    of the user will be revealed. Similarly, recent research has shown
    that deep learning models can learn on homomorphically encrypted
    data. Homomorphic encryption is a method of encryption that
    preserves the underlying algebraic properties of the data.
    Mathematically, this can be expressed as follows:

    ::: {.mediaobject}
    ![](2_files/B10354_08_001.jpg)
    :::

    ::: {.mediaobject}
    ![](2_files/B10354_08_002.jpg)
    :::

    Here [*E*]{.emphasis} is an encryption function, [*m*]{.emphasis} is
    some plain text data, and [*D*]{.emphasis} is a decryption function.
    As you can see, adding the encrypted data is the same as first
    adding the data and then encrypting it. Adding the data, encrypting
    it, and then decrypting it is the same as just adding the data.

    This means you can encrypt the data and still train a model on it.
    Homomorphic encryption is still in its infancy, but through
    approaches like this, you can ensure that in the case of a data
    breach, no sensitive individual information is leaked.

-   [**Train locally, and upload only a few gradients**]{.strong}: One
    way to avoid uploading user data is to train your model on the
    user\'s device. The user accumulates data on the device. You can
    then download your model on to the device and perform a single
    forward and backward pass on the device.

    To avoid the possibility of inference of user data from the
    gradients, you only upload a few gradients at random. You can then
    apply the gradients to your master model.

    To further increase the overall privacy of the system, you do not
    need to download all the newly update weights from the master model
    to the user\'s device, but only a few. This way, you train your
    model asynchronously without ever accessing any data. If your
    database gets breached, no user data is lost. However, we need to
    note that this only works if you have a large enough user base.
:::
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch08lvl2sec107}Preparing the data for training {#preparing-the-data-for-training .title}

</div>

</div>
:::

In earlier chapters, we have []{#id642 .indexterm}seen the benefits of
normalizing and scaling features, we also discussed how you should scale
all numerical features. There are four ways of feature scaling; these
include s[*tandardization, Min-Max, mean normalization,*]{.emphasis} and
[*unit length scaling*]{.emphasis}. In this section we\'ll break down
each one:

::: {.itemizedlist}
-   [**Standardization**]{.strong} []{#id643 .indexterm}ensures that all
    of the data has a mean of zero and a standard deviation of one. It
    is computed by subtracting the mean and dividing by the standard
    deviation of the data:

    ::: {.mediaobject}
    ![](2_files/B10354_08_003.jpg)
    :::

    This is probably the most common way of scaling features. It\'s
    especially useful if you suspect that your data contains outliers as
    it is quite robust. On the flip side, standardization does not
    ensure that your features are between zero and one, which is the
    range in which neural networks learn best.

-   [**Min-Max**]{.strong} rescaling does[]{#id644 .indexterm} exactly
    that. It scales all data between zero and one by first subtracting
    the minimum value and then dividing by the range of values. We can
    see this expressed in the formula below:

    ::: {.mediaobject}
    ![](2_files/B10354_08_004.jpg)
    :::

    If you know for sure that your data contains no outliers, which is
    the case in images, for instance, Min-Max scaling will give you a
    nice scaling of values between zero and one.

-   Similar to Min-Max, [**mean normalization**]{.strong} ensures your
    data has values between minus one []{#id645 .indexterm}and one with
    a mean of zero. This is done by subtracting the mean and then
    dividing by the range of data, which is expressed in the following
    formula:

    ::: {.mediaobject}
    ![](2_files/B10354_08_005.jpg)
    :::

    Mean normalization is done less frequently but, depending on your
    application, might be a good approach.

-   For some applications, it is better to not scale[]{#id646
    .indexterm} individual features, but instead vectors of features. In
    this case, you would apply [**unit length scaling**]{.strong} by
    dividing each element in[]{#id647 .indexterm} the vector by the
    total length of the vector, as we can see below:

    ::: {.mediaobject}
    ![](2_files/B10354_08_006.jpg)
    :::

    The length of the vector usually means the L2 norm of the vector
    [![](2_files/B10354_08_007.jpg)]{.inlinemediaobject}, that is, the
    square root of the sum of squares. For some applications, the vector
    length means the L1 norm of the vector,
    [![](2_files/B10354_08_008.jpg)]{.inlinemediaobject}, which is the
    sum of vector elements.
:::

However you scale, it is important to only measure the scaling factors,
mean, and standard deviation on the test set. These factors include only
a select amount of the information about the data. If you measure them
over your entire dataset, then the algorithm might perform better on the
test set than it will in production, due to this information advantage.

Equally importantly, you should check that your production code has
proper feature scaling as well. Over time, you should recalculate your
feature distribution and adjust your scaling.
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch08lvl2sec108}Understanding which inputs led to which predictions {#understanding-which-inputs-led-to-which-predictions .title}

</div>

</div>
:::

Why did your model[]{#id648 .indexterm} make the prediction it made? For
complex models, this question is pretty hard to answer. A global
explanation for a very complex model might in itself be very complex.
The[ **Local Interpretable Model-Agnostic Explanations**]{.strong}
([**LIME**]{.strong}) is, a popular algorithm for model explanation that
focuses on local explanations. Rather than trying to answer; \"How does
this model make predictions?\" LIME tries to answer; \"Why did the model
make [*this*]{.emphasis} prediction on [*this*]{.emphasis} data?\"

::: {.note style="margin-left: 0.5in; margin-right: 0.5in;"}
### []{#note38}Note {#note-2 .title}

[**Note**]{.strong}: The authors of LIME, Ribeiro, Singh, and Guestrin,
curated a great GitHub repository around their algorithm with many
explanations and tutorials, which you can find here:
<https://github.com/marcotcr/lime>.
:::

On Kaggle kernels, LIME is installed by default. However, you can
install LIME locally with the following command:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
pip install lime
```
:::

The LIME algorithm works with any classifier, which is why[]{#id649
.indexterm} it is model agnostic. To make an explanation, LIME cuts up
the data into several sections, such as areas of an image or utterances
in a text. It then creates a new dataset by removing some of these
features. It runs this new dataset through the black box classifier and
obtains the classifiers predicted probabilities for different classes.
LIME[]{#id650 .indexterm} then encodes the data as vectors describing
what features were present. Finally, it trains a linear model to predict
the outcomes of the black box model with different features removed. As
linear models are easy to interpret, LIME will use the linear model to
determine the most important features.

Let\'s say that you are using a text classifier, such as TF-IDF, to
classify emails such as those in the 20 newsgroup dataset. To get
explanations from this classifier, you would use the following snippet:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
from lime.lime_text import LimeTextExplainer               #1
explainer = LimeTextExplainer(class_names=class_names)     #2
exp = explainer.explain_instance(test_example,             #3classifier.predict_proba, #4num_features=6)           #5
                                 
exp.show_in_notebook()                                     #6
```
:::

Now, let\'s understand what\'s going on in that code snippet:

::: {.orderedlist}
1.  The LIME package has several classes for different types of data.

2.  To create a new blank explainer, we need to pass the names of
    classes of our classifier.

3.  We\'ll provide one text example for which we want an explanation.

4.  We provide the prediction function of our classifier. We need to
    provide a function that provides probabilities. For Keras, this is
    just `model.predict`{.literal};for scikit models, we need to use the
    `predict_proba`{.literal} method.

5.  LIME shows the maximum number of features. We want to show only
    the importance of the six most important features in this case.

6.  Finally, we can render a visualization of our prediction, which
    looks like this:
:::

::: {.mediaobject}
![](2_files/B10354_08_01.jpg)

::: {.caption}
LIME text output
:::
:::

The explanation shows the classes with different features that the text
gets classified as most often. It shows the words that most contribute
to the classification in the two most frequent classes. Under that, you
can see the words that contributed to the classification highlighted in
the text.

As you can see, our model []{#id651 .indexterm}picked up on parts of the
email address of the sender as distinguishing features, as well as the
name of the university, \"Rice.\" It sees \"Caused\" to be a strong
indicator that the text is about atheism. Combined, these are all things
we want to know when debugging datasets.

LIME does not perfectly solve the problem of explaining models. It
struggles if the interaction of multiple features leads to a certain
outcome for instance. However, it does well enough to be a useful data
debugging tool. Often, models pick up on things they should not be
picking up on. To debug a dataset, we need to remove all these
\"give-away\" features that statistical models like to overfit to.

Looking back at this section, you\'ve now seen a wide range of tools
that you can use to debug your dataset. Yet, even with a perfect
dataset, there can be issues when it comes to training. The next section
is about how to debug your model.



[]{#ch08lvl1sec100}Debugging your model {#debugging-your-model .title style="clear: both"}
---------------------------------------

</div>

</div>

------------------------------------------------------------------------
:::

Complex deep learning models are []{#id652 .indexterm}prone to error.
With millions of parameters, there are a number things that can go
wrong. Luckily, the field has developed a number of useful tools to
improve model performance. In this section, we will introduce the most
useful tools that you can use to []{#id653 .indexterm}debug and improve
your model.

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch08lvl2sec109}Hyperparameter search with Hyperas {#hyperparameter-search-with-hyperas .title}

</div>

</div>
:::

Manually tuning the hyperparameters of a neural network can be a tedious
task. Despite you possibly having some[]{#id654 .indexterm} intuition
about what works and what does not, there are no hard rules to apply
when it comes to tuning hyperparameters. This is why practitioners with
lots []{#id655 .indexterm}of computing power on hand use automatic
hyperparameter search. After all, hyperparameters form a search space
just like the model\'s parameters do. The difference is that we cannot
[]{#id656 .indexterm}apply backpropagation to them and cannot take
derivatives of them. We can still apply all non-gradient based
optimization algorithms to them.

There are a number of different hyperparameter optimization tools, but
we will look at Hyperas because of its ease of use. Hyperas is a wrapper
for `hyperopt`{.literal}, a popular optimization library made for
working with Keras.

::: {.note style="margin-left: 0.5in; margin-right: 0.5in;"}
### []{#note39}Note {#note .title}

[**Note**]{.strong}: You can find Hyperas on[]{#id657 .indexterm}
GitHub: <https://github.com/maxpumperla/hyperas>.
:::

We can install Hyperas with `pip`{.literal}:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
pip install hyperas
```
:::

Depending on your setup, you might []{#id658 .indexterm}need to make a
few adjustments to the installation. If this is the case, then the
Hyperas GitHub page, link above, offers more information.

Hyperas offers two optimization methods, [**Random Search**]{.strong}
and [**Tree of Parzen Estimators**]{.strong}. Within a range of
parameters that we think are reasonable, the random search will sample
randomly and train a model with random hyperparameters. It will then
pick the best-performing model as the solution.

[**Random search**]{.strong} is simple and robust, and it can be scaled
easily. It basically makes no assumption about the hyperparameters,
their relation, and the loss surface. On the flip side, it is relatively
slow.

The [**Tree of Parzen**]{.strong} ([**TPE**]{.strong})
algorithm[]{#id659 .indexterm} models the relation
[*P(x\|y),*]{.emphasis} where [*x*]{.emphasis} represents the
hyperparameters and [*y*]{.emphasis} the associated performance. This is
the exact opposite modeling of Gaussian processes, which model
[*P(y\|x)*]{.emphasis} and are popular with many researchers.

Empirically, it turns out that TPE performs better. For the precise
details, see the 2011 paper, [*Algorithms for Hyper-Parameter
Optimization*]{.emphasis}, available at:
<https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization>
\-- that was authored by James S. Bergstra and others. TPE is faster
than random search but can get stuck in local minima and struggles with
some difficult loss surfaces. As a rule of thumb, it makes sense to
start with TPE, and if TPE struggles, move to random search.

::: {.note style="margin-left: 0.5in; margin-right: 0.5in;"}
### []{#note40}Note {#note-1 .title}

[**Note**]{.strong}: The code for this example can be found at:
<https://www.kaggle.com/jannesklaas/Hyperas>.
:::

The following example will show you how to use Hyperas and Hyperopt for
an MNIST dataset classifier:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
from hyperopt import Trials, STATUS_OK, tpe        #1
from hyperas import optim                          #2
from hyperas.distributions import choice, uniform 
```
:::

While the code []{#id660 .indexterm}was short, let\'s explain what it
all means:

::: {.orderedlist}
1.  As Hyperas is built on Hyperopt, we need to import some pieces
    directly from `hyperopt`{.literal}. The `Trials`{.literal} class
    runs the actual trials, `STATUS_OK`{.literal} helps communicate that
    a test went well, and `tpe`{.literal} is an implementation of the
    TPE algorithm.

2.  Hyperas provides a number of handy functions that make working with
    Hyperopt easier. The `optim`{.literal} function finds optimal
    hyperparameters and can be used just like Keras\' `fit`{.literal}
    function. `choice`{.literal} and `uniform`{.literal} can be used
    to choose between[]{#id661 .indexterm} discrete and []{#id662
    .indexterm}continuous hyperparameters respectively.
:::

To build on the previous ideas that we\'ve explored, let\'s now add the
following, which we will explain in more detail once the code has been
written:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
def data():                                      #1
    import numpy as np                           #2
    from keras.utils import np_utils
    
    from keras.models import Sequential 
    from keras.layers import Dense, Activation, Dropout
    from keras.optimizers import RMSprop
    
    path = '../input/mnist.npz'                  #3
    with np.load(path) as f:
        X_train, y_train = f['x_train'], f['y_train']
        X_test, y_test = f['x_test'], f['y_test']

    X_train = X_train.reshape(60000, 784)        #4
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    nb_classes = 10
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    
    return X_train, y_train, X_test, y_test      #5
```
:::

Let\'s take a moment to look at the code we\'ve just produced:

::: {.orderedlist}
1.  Hyperas expects a function that loads the data; we cannot just pass
    on a dataset from memory.

2.  To scale the search, Hyperas creates a new runtime in which it does
    model creation and evaluation. This also means imports that we did
    in a notebook do not always transfer into the runtime. To be sure
    that all modules are available, we need to do all imports in the
    `data`{.literal} function. This is also true for modules
    that[]{#id663 .indexterm} will only be used for the model.

3.  We now load the data. Since Kaggle kernels do not have access to the
    internet, we need to load the MNIST data from disk. If you have
    internet, but no local version of the files, you can get the data
    using following code:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.programlisting .language-markup}
    from keras.datasets import mnist
    (Y_train, y_train), (X_test, y_test) = mnist.load_data() 
    ```
    :::

    I would still keep the no internet version around because it is the
    default setting.

4.  The `data`{.literal} function also needs to preprocess the data. We
    do the standard reshaping[]{#id664 .indexterm} and scaling that we
    did when we worked with MNIST earlier.

5.  Finally, we return[]{#id665 .indexterm} the data. This data will be
    passed into the function that builds and evaluates the model:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.programlisting .language-markup}
    def model(X_train, y_train, X_test, y_test):                   #1
        model = Sequential()                                       #2
        model.add(Dense(512, input_shape=(784,)))
        
        model.add(Activation('relu'))
        
        model.add(Dropout({{uniform(0, 0.5)}}))                    #3
        
        model.add(Dense({{choice([256, 512, 1024])}}))             #4
        
        model.add(Activation({{choice(['relu','tanh'])}}))         #5
        
        model.add(Dropout({{uniform(0, 0.5)}}))
        
        model.add(Dense(10))
        model.add(Activation('softmax'))

        rms = RMSprop()
        model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])

        model.fit(X_train, y_train,                                #6batch_size={{choice([64, 128])}},epochs=1,verbose=2,validation_data=(X_test, y_test))
        score, acc = model.evaluate(X_test, y_test, verbose=0)     #7
        print('Test accuracy:', acc)
        return {'loss': -acc, 'status': STATUS_OK, 'model': model} #8
    ```
    :::
:::

As you can see, the preceding snippet of code is made up of eight
defining pieces. Let\'s now explore them so that we\'re able to fully
understand the code we\'ve just produced:

::: {.orderedlist}
1.  The `model`{.literal} function both defines the model and evaluates
    it. Given a training dataset from the `data`{.literal} function, it
    returns a set of quality metrics.

2.  When fine-tuning with Hyperas, we can define a Keras model just as
    we usually would. Here, we only have to replace the hyperparameters
    we want to tune with Hyperas functions.

3.  To tune dropout,[]{#id666 .indexterm} for instance, we replace the
    `Dropout`{.literal} hyperparameter with
    `{{uniform(0, 0.5)}}`{.literal}. Hyperas will automatically sample
    and evaluate dropout rates between `0`{.literal} and
    `0.5`{.literal}, sampled from a uniform distribution.

4.  To sample from discrete distributions, for instance, the size of a
    hidden layer, we replace the hyperparameter with
    `{{choice([256, 512, 1024])}}`{.literal}. Hyperas will choose from a
    hidden layer size of 256, 512, and 1,024 now.

5.  We can do the same to choose activation functions.

6.  To evaluate[]{#id667 .indexterm} the model, we need to compile and
    fit it. In this process, we can also choose between different batch
    sizes. In this case, we only train for one epoch, to keep the time
    needed for this example short. You could also run a whole training
    process with Hyperas.

7.  To gain insight[]{#id668 .indexterm} into how well the model is
    doing, we evaluate it on test data.

8.  Finally, we return the model\'s score, the model itself, and an
    indicator that everything went okay. Hyperas tries to minimize a
    loss function. To maximize accuracy, we set the loss to be the
    negative accuracy. You could also pass the model loss here,
    depending on what the best optimization method is for your problem.

    Finally, we run the optimization:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.programlisting .language-markup}
    best_run, best_model = optim.minimize(model=model,
                                        data=data,
                                        algo=tpe.suggest,
                                        max_evals=5,
                                        trials=Trials(),
                    notebook_name='__notebook_source__')
    ```
    :::
:::

We pass the `model`{.literal} method and the `data`{.literal} method,
and we specify how many trials we want to run and which class should
govern the trials. Hyperopt also offers a distributed trials class in
which workers communicate via MongoDB.

When working in a Jupyter Notebook, we need to provide the name of the
notebook we are working in. Kaggle Notebooks all have the filename
`__notebook_source__`{.literal}, regardless of the name you gave them.

After it\'s run, Hyperas returns the best-performing model as well as
the hyperparameters of the best model. If you print out
`best_run`{.literal}, you should see output similar to this:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
{'Activation': 1,
 'Dense': 1,
 'Dropout': 0.3462695171578595,
 'Dropout_1': 0.10640021656377913,
 'batch_size': 0}
```
:::

For `choice`{.literal} selections, Hyperas []{#id669 .indexterm}shows
the index. In this case, the activation function `tanh`{.literal} was
chosen.

In this case we ran the hyperparameter []{#id670 .indexterm}search only
for a few trials. Usually, you would run a few hundred or thousand
trials. To do this we would use automated hyperparameter search, which
can be a great tool to improve model[]{#id671 .indexterm} performance
if you have enough compute power available.

However, it won\'t get a model that does not work at all to work. When
choosing this approach, you need to be sure to have a somewhat-working
approach first before investing in hyperparameter search.
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch08lvl2sec110}Efficient learning rate search {#efficient-learning-rate-search .title}

</div>

</div>
:::

One of the most important hyperparameters is the learning rate. Finding
a good learning rate is hard. Too small []{#id672 .indexterm}and your
model might train so slowly that you believe it is not training at all,
but if it\'s too large it will overshoot and not reduce the loss as
well.

When it comes to finding a learning rate, standard hyperparameter search
techniques are not the best choice. For the learning rate, it is better
to perform a line search and visualize the loss for different learning
rates, as this will give you an understanding of how the loss function
behaves.

When doing a line search, it is better to increase the learning rate
exponentially. You are more likely to care about the region of smaller
learning rates than about very large learning rates.

In our example below, we perform 20 evaluations and double the learning
rate in every evaluation. We can run this by executing the following
code:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
init_lr = 1e-6                                            #1
losses = [] 
lrs = []
for i in range(20):                                       #2
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu')) 
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    opt = Adam(lr=init_lr*2**i)                           #3
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['acc'])


    hist = model.fit(X_train, Y_train, batch_size = 128, epochs=1)                                                 #4

    loss = hist.history['loss'][0]                        #5
    losses.append(loss)
    lrs.append(init_lr*2**i)
```
:::

Let\'s now take a more[]{#id673 .indexterm} detailed look at the
preceding featured code:

::: {.orderedlist}
1.  We specify a low, but still reasonable, initial learning rate from
    which we start our search.

2.  We then perform training 20 times with different learning rates. We
    need to set up the model from scratch each time.

3.  We calculate our new learning rate. In our case, we double the
    learning rate in each evaluation step. You could also use a smaller
    increase if you want a more fine-grained picture.

4.  We then fit the model with our new learning rate.

5.  Finally, we keep track of the loss.
:::

If your dataset is very large, you can perform this learning rate search
on a subset of the data. The interesting part comes from the
visualization of learning rates:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
fig, ax = plt.subplots(figsize = (10,7))
plt.plot(lrs,losses)
ax.set_xscale('log')
```
:::

When you run this code, it will then output the following chart:

::: {.mediaobject}
![](3_files/B10354_08_02.jpg)

::: {.caption}
Learning rate finder
:::
:::

As you can see, the[]{#id674 .indexterm} loss is optimal between 1e-3
and 1e-2. We can also see that the loss surface is relatively flat in
this area. This gives us insight that we should use a learning rate
around 1e-3. To avoid overshooting, we select a learning rate somewhat
lower than the optimum found by line search.
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch08lvl2sec111}Learning rate scheduling {#learning-rate-scheduling .title}

</div>

</div>
:::

Why stop at using one learning rate? In the[]{#id675 .indexterm}
beginning, your model might be far away from the optimal solution, and
so because of that you want to move as fast as possible. As you approach
the minimum loss, however, you want to move slower to avoid
overshooting. A popular method is to anneal the learning rate so that it
represents a cosine function. To this end, we need to the find a
learning rate scheduling function, that given a time step,
[*t*]{.emphasis}, in epochs returns a learning rate. The learning rate
becomes a function of [*t*]{.emphasis}:

::: {.mediaobject}
![](3_files/B10354_08_009.jpg)
:::

Here [*l*]{.emphasis} is the cycle length and
[![](3_files/B10354_08_010.jpg)]{.inlinemediaobject} is the initial
learning rate. We modify this function to ensure that [*t*]{.emphasis}
does not become larger than the cycle length:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
def cosine_anneal_schedule(t):
    lr_init = 1e-2                               #1
    anneal_len = 5
    if t >= anneal_len: t = anneal_len -1        #2
    cos_inner = np.pi * (t % (anneal_len))       #3
    cos_inner /= anneal_len
    cos_out = np.cos(cos_inner) + 1
    return float(lr_init / 2 * cos_out)
```
:::

The preceding code features three key features:

::: {.orderedlist}
1.  In our function, we need to set up a starting point from which we
    anneal. This can be a relatively large learning rate. We also need
    to specify how many epochs we want to anneal.

2.  A cosine []{#id676 .indexterm}function does not monotonically
    decrease; it goes back up after a cycle. We will use this property
    later; for now, we will just make sure that the learning rate does
    not go back up.

3.  Finally we calculate the new learning rate using the preceding
    formula. This is the new learning rate.
:::

To get a better understanding of what the learning rate scheduling
function does, we can plot the learning rate it would set over 10
epochs:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
srs = [cosine_anneal_schedule(t) for t in range(10)]
plt.plot(srs)
```
:::

With the output of the code being shown in the following graph:

::: {.mediaobject}
![](3_files/B10354_08_03.jpg)

::: {.caption}
Cosine anneal
:::
:::

We can use this function to schedule learning rates with Keras\'
`LearningRateScheduler`{.literal} callback:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
from keras.callbacks import LearningRateScheduler
cb = LearningRateScheduler(cosine_anneal_schedule)
```
:::

We now have a[]{#id677 .indexterm} callback that Keras will call at the
end of each epoch in order to get a new learning rate. We pass this
callback to the `fit`{.literal} method and voilà, our model trains with
a decreasing learning rate:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
model.fit(x_train,y_train,batch_size=128,epochs=5,callbacks=[cb])
```
:::

A version of the learning rate annealing is to add restarts. At the end
of an annealing cycle, we move the learning rate back up. This is a
method used to avoid overfitting. With a small learning rate, our model
might find a very narrow minimum. If the data we want to use our model
on is slightly different from the training data, then the loss surface
might change a bit, and our model could be out of the narrow minimum for
this new loss surface. If we set the learning rate back up, our model
will get out of narrow minima. Broad minima, however, are stable enough
for the model to stay in them:

::: {.mediaobject}
![](3_files/B10354_08_04.jpg)

::: {.caption}
Shallow broad minima
:::
:::

As the cosine[]{#id678 .indexterm} function goes back up by itself, we
only have to remove the line to stop it from doing so:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
def cosine_anneal_schedule(t):
    lr_init = 1e-2 
    anneal_len = 10 
    cos_inner = np.pi * (t % (anneal_len))  
    cos_inner /= anneal_len
    cos_out = np.cos(cos_inner) + 1
    return float(lr_init / 2 * cos_out)
```
:::

The new learning rate schedule now looks like this:

::: {.mediaobject}
![](3_files/B10354_08_05.jpg)

::: {.caption}
Learning rate restarts
:::
:::
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch08lvl2sec112}Monitoring training with TensorBoard {#monitoring-training-with-tensorboard .title}

</div>

</div>
:::

An important[]{#id679 .indexterm} part of debugging a model is knowing
when things go wrong before you have invested significant amounts of
time training the model. TensorBoard is a TensorFlow extension that
allows you to easily monitor your model in a browser.

To provide an interface from which you can watch your model\'s progress,
TensorBoard also offers some options useful for debugging. For example,
you can observe the distributions of the model\'s weights and gradients
during training.

::: {.note style="margin-left: 0.5in; margin-right: 0.5in;"}
### []{#note41}Note {#note-2 .title}

[**Note**]{.strong}: TensorBoard does not run on Kaggle. To try out
TensorBoard, install Keras and TensorFlow on your own machine.
:::

To use TensorBoard with Keras, we set up a new callback. TensorBoard has
many options, so let\'s walk through them step by step:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
from keras.callbacks import TensorBoard
tb = TensorBoard(log_dir='./logs/test2',           #1
                 histogram_freq=1,                 #2
                 batch_size=32,                    #3
                 write_graph=True,                 #4
                 write_grads=True, 
                 write_images=True, 
                 embeddings_freq=0,                #5
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)
```
:::

There are five key pieces of the preceding code that we need to take
into consideration:

::: {.orderedlist}
1.  First, we need to specify where Keras should save the data that
    TensorBoard later visualizes. Generally, it is a good idea to save
    all logs of your different[]{#id680 .indexterm} runs in one
    `logs`{.literal} folder and give every run its own subfolder, such
    as `test2`{.literal} in this case. This way, you can easily compare
    different runs within TensorBoard but also keep different runs
    separate.

2.  By default, TensorBoard would just show you the loss and accuracy
    of your model. In this case, we are interested in histograms showing
    weights and distributions. We save the data for the histograms every
    epoch.

3.  To generate data, TensorBoard runs batches through the model. We
    need to specify a batch size for this process.

4.  We need to tell TensorBoard what to save. TensorBoard can visualize
    the model\'s computational graph, its gradients, and images showing
    weights. The more we save however, the slower the training.

5.  TensorBoard can also visualize trained embeddings nicely. Our model
    does not have embeddings, so we are not interested in saving them.
:::

Once we have the callback set up, we can pass it to the training
process. We will train the MNIST model once again. We multiply the
inputs by 255, making training much harder. To achieve all of this we
need to run the following code:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
hist = model.fit(x_train*255,y_train,batch_size=128,epochs=5,callbacks=[tb],validation_data=(x_test*255,y_test))
```
:::

To start TensorBoard, open your console and type in the following:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
tensorboard --logdir=/full_path_to_your_logs
```
:::

Here `full_path_to_your_logs`{.literal} is the path you saved your logs
in, for example, `logs`{.literal} in our case. TensorBoard runs on port
`6006`{.literal} by default, so in your browser,
go to `http://localhost:6006`{.literal} to see TensorBoard.

Once the page has loaded, navigate to the [**HISTOGRAMS**]{.strong}
section; this section should look something like this:

::: {.mediaobject}
![](3_files/B10354_08_06.jpg)

::: {.caption}
TensorBoard histograms
:::
:::

You can see[]{#id681 .indexterm} the distribution of gradients and
weights in the first layer. As you can see, the gradients are uniformly
distributed and extremely close to zero. The weights hardly change at
all over the different epochs. We are dealing[]{#id682 .indexterm} with
a [**vanishing gradient problem**]{.strong}; we will cover this problem
in depth later.

Armed with the real-time insight that this problem is happening, we can
react faster. If you really want to dig into your model, TensorBoard
also offers a visual debugger. In this debugger, you can step through
the execution of your TensorFlow model and examine every single value
inside it. This is especially useful if you are working on complex
models, such as generative adversarial networks, and are trying to
understand why something complex goes wrong.

::: {.note style="margin-left: 0.5in; margin-right: 0.5in;"}
### []{#note42}Note {#note-3 .title}

[**Note**]{.strong}: The TensorFlow debugger does not work well with
models trained in Jupyter Notebooks. Save your model training code to a
Python `.py`{.literal} script and run that script.
:::

To use the TensorFlow debugger, you have to set your model\'s runtime to
a special debugger runtime. In specifying the debugger runtime, you also
need to specify which port you want the debugger to run, in this case,
port `2018`{.literal}:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import keras

keras.backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.Session(), "localhost:2018"))
```
:::

Once Keras begins to work[]{#id683 .indexterm} with the debugger
runtime, you can debug your model. For the debugger to work, you need to
name your Keras model to `model`{.literal}. However, you do not need
to train the model with a TensorBoard callback.

Now, start TensorBoard and activate the debugger by specifying the
debugger port as follows:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
tensorboard --logdir=/full_path_to_your_logs --debugger_port 2018
```
:::

Now you can open TensorBoard as usual in your browser on port
`6006`{.literal}. TensorBoard now has a new section called
[**DEBUGGER**]{.strong}:

::: {.mediaobject}
![](3_files/B10354_08_07.jpg)

::: {.caption}
TensorBoard debugger
:::
:::

By clicking [**STEP**]{.strong}, you execute the next step in the
training process. With [**CONTINUE...,**]{.strong} you[]{#id684
.indexterm} can train your model for one or more epochs. By navigating
the tree on the left side, you can view the components of your model.
You can visualize individual elements of your model, to see how
different actions affect them. Using the debugger effectively requires a
bit of practice, but if you are working with complex models, it is a
great tool.
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch08lvl2sec113}Exploding and vanishing gradients {#exploding-and-vanishing-gradients .title}

</div>

</div>
:::

The vanishing gradient problem[]{#id685 .indexterm} describes the issue
that sometimes gradients in a deep neural network become very small and
as a result, training occurs very slowly. Exploding []{#id686
.indexterm}gradients are the opposite problem; they are gradients that
become so large that the network does not converge.

Of the two, the vanishing gradient problem is the more persistent issue.
Vanishing gradients are caused by the fact that in deep networks,
gradients of earlier layers depend on gradients of layers closer to the
output. If the output gradients are small, then the gradients behind
them are even smaller. Thus, the deeper the network, the more issues
that occur with regard to vanishing gradients.

The key causes of small gradients include sigmoid and `tanh`{.literal}
activation functions. If you look at the following sigmoid function,
you\'ll see that it is very flat toward large values:

::: {.mediaobject}
![](3_files/B10354_08_08.jpg)

::: {.caption}
Sigmoid vanishing
:::
:::

The small gradients []{#id687 .indexterm}of the sigmoid function are the
reason why the ReLU activation function has become popular for training
deep neural networks. Its gradient is equal to one for all positive
input values. However, it is zero for all negative input values.

Another cause[]{#id688 .indexterm} of vanishing gradients is saddle
points in the loss function. Although no minimum was reached, the loss
function is very flat in some areas, producing small gradients.

To combat the vanishing gradient problem, you should use ReLU
activation. If you see that your model is training slowly, consider
increasing the learning rate to move out of a saddle point faster.
Finally, you might just want to let the model train longer if it suffers
from small gradients.

The exploding gradient problem is usually caused by large absolute
weight values. As backpropagation multiplies the later layers\'
gradients with the layers\' weights, large weights amplify gradients. To
counteract the exploding gradient problem, you can use weight
regularization, which incentivizes smaller weights. Using a method
called [**gradient clipping**]{.strong}, you can ensure that gradients
do not become larger than a certain value. In Keras, you can clip both
the norm and the absolute value of gradients:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
from keras.optimizers import SGD

clip_val_sgd = SGD(lr=0.01, clipvalue=0.5)
clip_norm_sgd = SGD(lr=0.01, clipnorm=1.)
```
:::

Convolutional layers[]{#id689 .indexterm} and [**long short-term memory
(LSTM) networks**]{.strong} are less susceptible to both vanishing and
exploding gradients. ReLU and batch normalization generally stabilize
the network. Both[]{#id690 .indexterm} of these problems might be caused
by non-regularized inputs, so you should check your data too. Batch
normalization also counteracts exploding gradients.

If exploding gradients are a problem, you can add a batch normalization
layer to your model as follows:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
from keras.layers import BatchNormalization
model.add(BatchNormalization())
```
:::

Batch normalization also reduces the risk of vanishing gradients and has
enabled the construction of much deeper networks recently.

You have now seen a wide range of tools that can be used to debug your
models. As a final step, we are going to learn some methods to run
models in production and speed up the machine learning process.



[]{#ch08lvl1sec101}Deployment {#deployment .title style="clear: both"}
-----------------------------

</div>

</div>

------------------------------------------------------------------------
:::

Deployment into production is often seen as separate from the creation
of models. At many companies, data scientists create models in isolated
development environments on training, validation, and []{#id691
.indexterm}testing data that was collected to create models.

Once the model performs well on the test set, it then gets passed on to
deployment engineers, who know little about how and why the model works
the way it does. This is a mistake. After all, you are developing models
to use them, not for the fun of developing them.

Models tend to perform worse over time for several reasons. The world
changes, so the data you trained on might no longer represent the real
world. Your model might rely on the outputs of some other systems that
are subject to change. There might be unintended side effects and
weaknesses of your model that only show with extended usage. Your model
might influence the world that it tries to model. [**Model
decay**]{.strong} describes how models have a lifespan after which
performance deteriorates.

Data scientists should have the full life cycle of their models in mind.
They need to be aware of how their model works in production in the long
run.

In fact, the production environment is the perfect environment to
optimize your model. Your datasets are only an approximation for the
real world. Live data gives a much fresher and more accurate view of the
world. By using online learning []{#id692 .indexterm}or active learning
methods, you can drastically reduce the need for training data.

This section describes some best practices for getting your models to
work in the real world. The exact method of serving your model can vary
depending on your application. See the upcoming section [*Performance
Tips*]{.emphasis} for more details on choosing a deployment method.

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch08lvl2sec114}Launching fast {#launching-fast .title}

</div>

</div>
:::

The process of developing []{#id693 .indexterm}models depends on
real-world data as well as an insight into how the performance of the
model influences business outcomes. The earlier you can gather data and
observe how model behavior influences outcomes, the better. Do not
hesitate to launch your product with a simple heuristic.

Take the case of fraud detection, for instance. Not only do you need to
gather transaction data together with information about occurring
frauds, you also want to know how quick fraudsters are at finding ways
around your detection methods. You want to know how customers whose
transactions have been falsely flagged as fraud react. All of this
information influences your model design and your model evaluation
metrics. If you can come up with a simple heuristic, deploy the
heuristic and then work on the machine learning approach.

When developing a machine learning model, try simple models first. A
surprising number of tasks can be modeled with simple, linear models.
Not only do you obtain results faster, but you can also quickly identify
the features that your model is likely to overfit to. Debugging your
dataset before working on a complex model can save you many headaches.

A second advantage of getting a simple approach out of the door quickly
is that you can prepare your infrastructure. Your infrastructure team is
likely made up of different people from the modeling team. If the
infrastructure team does not have to wait for the modeling team but can
start optimizing the infrastructure immediately, then you gain a time
advantage.
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch08lvl2sec115}Understanding and monitoring metrics {#understanding-and-monitoring-metrics .title}

</div>

</div>
:::

To ensure that []{#id694 .indexterm}optimizing metrics such as the mean
squared error or cross-entropy loss actually lead to a better outcome,
you need to be mindful of how your model metrics relate to higher order
metrics, which you can see visualized in the following diagram. Imagine
you have some consumer-facing app in which you recommend different
investment products to retail investors.

::: {.mediaobject}
![](4_files/B10354_08_09.jpg)

::: {.caption}
Higher order effects
:::
:::

You might predict whether the user is interested in a given product,
measured by the user reading the product description. However, the
metric you want to optimize in your application is not your model
accuracy, but the click-through rate of users going to the description
screen. On a higher order, your business is not designed to maximize the
click-through rate, but revenue. If your users only click on low-revenue
products, your click-through rate []{#id695 .indexterm}does not help
you.

Finally, your business\' revenue might be optimized to the detriment of
society. In this case, regulators will step in. Higher order effects are
influenced by your model. The higher the order of the effect, the harder
it is to attribute to a single model. Higher order effects have large
impacts, so effectively, higher order effects serve as meta-metrics to
lower-order effects. To judge how well your application is doing, you
align its metrics, for example, click-through rates, with the metrics
relevant for the higher order effect, for example, revenue. Equally,
your model metrics need to be aligned with your application metrics.

This alignment is often an emergent feature. Product managers eager to
maximize their own metrics pick the model that maximizes their metrics,
regardless of what metrics the modelers were optimizing. Product
managers that bring home a lot of revenue get promoted. Businesses that
are good for society receive subsidies and favorable policy. By making
the alignment explicit, you can design a better monitoring process. For
instance, if you have two models, you can A/B test them to see which one
improves the application metrics.

Often, you will find that to align with a higher order metric, you\'ll
need to combine several metrics, such as accuracy and speed of
predictions. In this case, you should craft a formula that combines the
metrics into one single number. A single number will allow you to
doubtlessly[]{#id696 .indexterm} choose between two models and help your
engineers to create better models.

For instance, you could set a maximum latency of 200 milliseconds and
your metric would be, \"Accuracy if latency is below 200 milliseconds,
otherwise zero.\" If you do not wish to set one maximum latency value,
you could choose, \"Accuracy divided by latency in milliseconds.\" The
exact design of this formula depends on your application. As you observe
how your model influences its higher order metric, you can adapt your
model metric. The metric should be simple and easy to quantify.

Next, to regularly test your model\'s impact on higher order metrics,
you should regularly test your models own metrics, such as accuracy. To
this end, you need a constant stream of ground truth labels together
with your data. In some cases, such as detecting fraud, ground truth
data is easily collected, although it might come in with some latency.
In this case, customers might need a few weeks to find out they have
been overcharged.

In other cases, you might not have ground truth labels. Often, you can
hand-label data for which you have no ground truth labels coming in.
Through good UI design, the process of checking model predictions can be
fast. Testers only have to decide whether your model\'s prediction was
correct or not, something they can do through button presses in a web or
mobile app. If you have a good review system in place, data scientists
who work on the model should regularly check the model\'s outputs. This
way, patterns in failures (our model does poorly on dark images) can be
detected quickly, and the model can be improved.
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch08lvl2sec116}Understanding where your data comes from {#understanding-where-your-data-comes-from .title}

</div>

</div>
:::

More often than not, your data gets collected by some other system that
you as the model developer have no control over. Your data might be
collected by a data vendor or by a different department in your firm. It
might even be collected for different purposes than your model. The
collectors of the data might not even know you are using the data for
your model.

If, say, the collection[]{#id697 .indexterm} method of the data changes,
the distribution of your data might change too. This could break your
model. Equally, the real world might just change, and with it the data
distribution. To avoid changes in the data breaking your model, you
first need to be aware of what data you are using and assign an owner to
each feature. The job of the feature owner is to investigate where the
data is coming from and alert the team if changes in the data are
coming. The feature owner should also write down the assumptions
underlying the data. In the best case, you test these assumptions for
all new data streaming in. If the data does not pass the tests,
investigate and eventually modify your model.

Equally, your model outputs might get used as inputs of other models.
Help consumers of your data reach you by[]{#id698 .indexterm} clearly
identifying yourself as the owner of the model.

Alert users of your model of changes to your model. Before deploying
a model, compare the new model\'s predictions to the old model\'s
predictions. Treat models as software and try to identify \"breaking
changes,\" that would significantly alter your model\'s behavior. Often,
you might not know who is accessing your model\'s predictions. Try to
avoid this by clear communication and setting access controls
if necessary.

Just as software has dependencies, libraries that need to be installed
for the software to work, machine learning models have data
dependencies. Data dependencies are not as well understood as software
dependencies. By investigating your model\'s dependencies, you can
reduce the risk of your model breaking when data changes.



[]{#ch08lvl1sec102}Performance tips {#performance-tips .title style="clear: both"}
-----------------------------------

</div>

</div>

------------------------------------------------------------------------
:::

In many financial applications, speed is of[]{#id699 .indexterm} the
essence. Machine learning, especially deep learning, has a reputation
for being slow. However, recently, there have been many advances in
hardware and software that enable faster machine learning applications.

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch08lvl2sec117}Using the right hardware for your problem {#using-the-right-hardware-for-your-problem .title}

</div>

</div>
:::

A lot of progress in deep learning has been driven by the use of
[**graphics processing units**]{.strong} ([**GPUs**]{.strong}). GPUs
enable highly []{#id700 .indexterm}parallel computing at the expense of
operating frequency. Recently, multiple manufacturers have started
working on specialized []{#id701 .indexterm}deep learning hardware. Most
of the time, GPUs are a good choice for deep learning models or other
parallelizable algorithms such as XGboost gradient-boosted trees.
However, not all applications benefit equally.

In [**natural language processing (NLP)**]{.strong}, for instance, batch
sizes often need to be small, so the parallelization of operations does
not work as well since not that many samples are processed at the same
time. Additionally, some words appear much more often than others,
giving large benefits to caching frequent words. Thus, many NLP tasks
run faster on CPUs than GPUs. If you can work with large batches,
however, a GPU or even specialized hardware is preferable.
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch08lvl2sec118}Making use of distributed training with TF estimators {#making-use-of-distributed-training-with-tf-estimators .title}

</div>

</div>
:::

Keras is not only a standalone[]{#id702 .indexterm} library that can use
TensorFlow, but it is also an integrated part of TensorFlow. TensorFlow
features multiple high-level APIs that can be used to create and train
models.

From version 1.8 onward, the estimator API\'s features distribute
training on multiple machines, while the Keras API does not feature them
yet. Estimators also have a number of other speed-up tricks, so they are
usually faster than Keras models.

::: {.note style="margin-left: 0.5in; margin-right: 0.5in;"}
### []{#note43}Note {#note .title}

You can find information on how to set up your cluster for distributed
TensorFlow here: <https://www.tensorflow.org/deploy/distributed>.
:::

By changing the `import`{.literal} statements, you can easily use Keras
as part of TensorFlow and don\'t have to change your main code:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
import tensorflow as tf
from tensorflow.python import keras

from tensorflow.python.keras.models import Sequentialfrom tensorflow.python.keras.layers import Dense,Activation
```
:::

In this section, we will create a model to learn the MNIST problem
before training it using the estimator API. First, we load and prepare
the dataset as usual. For more efficient dataset loading, see the next
section:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train.shape = (60000, 28 * 28)
x_train = x_train / 255
y_train = keras.utils.to_categorical(y_train)
```
:::

We can create a Keras model as usual:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
model = Sequential()
model.add(Dense(786, input_dim = 28*28))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(160))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9),loss='categorical_crossentropy',metric='accuracy')
```
:::

The TensorFlow version of Keras offers a one-line conversion to a TF
estimator:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
estimator = keras.estimator.model_to_estimator(keras_model=model)
```
:::

To set up training, we need to know the name assigned to the model
input. We can quickly check this with the following code:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
model.input_names
['dense_1_input']
```
:::

Estimators get trained[]{#id703 .indexterm} with an input function. The
input function allows us to specify a whole pipeline, which will be
executed efficiently. In this case, we only want an input function that
yields our training set:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'dense_1_input': x_train},y=y_train,num_epochs=1,shuffle=False)
```
:::

Finally, we train the estimator on the input. And that is it; you can
now utilize distributed TensorFlow with estimators:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
estimator.train(input_fn=train_input_fn, steps=2000)
```
:::
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch08lvl2sec119}Using optimized layers such as CuDNNLSTM {#using-optimized-layers-such-as-cudnnlstm .title}

</div>

</div>
:::

You will often find that someone created a special layer optimized to
perform certain tasks on certain hardware. Keras\' `CuDNNLSTM`{.literal}
layer, for example, only runs on GPUs supporting CUDA, a programming
[]{#id704 .indexterm}language specifically for GPUs.

When you lock in your model to specialized hardware, you can often make
significant gains in your performance. If you have the resources, it
might even make sense to write your own specialized layer in CUDA. If
you want to change hardware later, you can usually export weights and
import them to a different layer.
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch08lvl2sec120}Optimizing your pipeline {#optimizing-your-pipeline .title}

</div>

</div>
:::

With the right hardware[]{#id705 .indexterm} and optimized software in
place, your model often ceases to be the bottleneck. You should check
your GPU utilization by entering the following command in your Terminal:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
nvidia-smi -l 2
```
:::

If your GPU utilization is not at around 80% to 100%, you can gain
significantly by optimizing your pipeline. There are several[]{#id706
.indexterm} steps you can take to optimize your pipeline:

::: {.itemizedlist}
-   [**Create a pipeline running parallel to the model**]{.strong}:
    Otherwise, your GPU will be idle while the data is loading. Keras
    does this by default. If you have a generator and want to have a
    larger queue of data to be held ready for preprocessing, change the
    `max_queue_size`{.literal} parameter of the
    `fit_generator`{.literal} method. If you set the `workers`{.literal}
    argument of the `fit_generator`{.literal} method to zero, the
    generator will run on the main thread, which slows things down.

-   [**Preprocess data in parallel**]{.strong}: Even if you have a
    generator working independently of the model training, it might not
    keep up with the model. So, it is better to run multiple generators
    in parallel. In Keras, you can do this by setting
    `use_multiprocessing`{.literal} to `true`{.literal} and setting the
    number of workers to anything larger than one, preferably to the
    number of CPUs available. Let\'s look at an example:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.programlisting .language-markup}
    model.fit_generator(generator, steps_per_epoch = 40, workers=4, use_multiprocessing=False)
    ```
    :::

    You need to make sure your generator is thread safe. You can make
    any generator thread safe with the following code snippet:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.programlisting .language-markup}
    import threading

    class thread_safe_iter:                   #1
        def __init__(self, it):
            self.it = it
            self.lock = threading.Lock()

        def __iter__(self):
            return self

        def next(self):                       #2
            with self.lock:
                return self.it.next()

    def thread_safe_generator(f):             #3
        def g(*a, **kw):
            return thread_safe_iter(f(*a, **kw))
        return g

    @thread_safe_generator
    def gen():
    ```
    :::

    Let\'s look at the three key components of the preceding code:

    ::: {.orderedlist}
    1.  The `thread_safe_iter`{.literal} class makes any iterator thread
        safe by locking threads when the iterator has to produce the
        next yield.

    2.  When `next()`{.literal} is called on the iterator, the iterators
        thread is locked. Locking means that no other function, say,
        another variable, can access variables from the thread while it
        is locked. Once the thread is locked, it yields the next
        element.

    3.  `thread_safe_generator`{.literal} is a Python decorator that
        turns any iterator it decorates into a[]{#id707 .indexterm}
        thread-safe iterator. It takes the function, passes it to the
        thread-safe iterator, and then returns the thread-safe version
        of the function.
    :::

    You can also use the `tf.data`{.literal} API together with an
    estimator, which does most of the work for you.

-   [**Combine files into large files**]{.strong}: Reading a file takes
    time. If you have to read thousands of small files, this can
    significantly slow you down. TensorFlow offers its own data format
    called TFRecord. You can also just fuse an entire batch into
    a single NumPy array and save that array instead of every example.

-   [**Train with the** ]{.strong} `tf.data.Dataset`{.literal} [
    **API**]{.strong}: If you are using the TensorFlow version of Keras,
    you can use the `Dataset`{.literal} API, which optimizes data
    loading and processing for you. The `Dataset`{.literal} API is the
    recommended way to load data into TensorFlow. It offers a wide range
    of ways to load data, for instance, from a CSV file with
    `tf.data.TextLineDataset`{.literal}, or from TFRecord files with
    `tf.data.TFRecordDataset`{.literal}.

    ::: {.note style="margin-left: 0.5in; margin-right: 0.5in;"}
    ### []{#note44}Note {#note-1 .title}

    [**Note**]{.strong}: For a more comprehensive guide to the
    `Dat`{.literal} `aset`{.literal} API, see
    <https://www.tensorflow.org/get_started/datasets_quickstart>.
    :::

    In this example, we will use the []{#id708 .indexterm}dataset API
    with NumPy arrays that we have already loaded into RAM, such as the
    MNIST database.

    First, we create two plain datasets for data and targets:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.programlisting .language-markup}
    dxtrain = tf.data.Dataset.from_tensor_slices(x_test)
    dytrain = tf.data.Dataset.from_tensor_slices(y_train)
    ```
    :::

    The `map`{.literal} function allows us to perform operations on data
    before passing it to the model. In this case, we apply one-hot
    encoding to our targets. However, this could be any function. By
    setting the `num_parallel_calls`{.literal} argument, we can specify
    how many processes we want to run in parallel:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.programlisting .language-markup}
    def apply_one_hot(z):
        return tf.one_hot(z,10)

    dytrain = dytrain.map(apply_one_hot,num_parallel_calls=4)
    ```
    :::

    We zip the data and targets into one dataset. We instruct TensorFlow
    to shuffle the data when loading, keeping[]{#id709 .indexterm} 200
    instances in memory from which to draw samples. Finally, we make the
    dataset yield batches of batch size `32`{.literal}:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.programlisting .language-markup}
    train_data = tf.data.Dataset.zip((dxtrain,dytrain)).shuffle(200).batch(32)
    ```
    :::

    We can now fit a Keras model on this dataset just as we would fit it
    to a generator:

    ::: {.informalexample}
    ::: {.toolbar .clearfix}
    Copy
    :::

    ``` {.programlisting .language-markup}
    model.fit(dataset, epochs=10, steps_per_epoch=60000 // 32)
    ```
    :::
:::

If you have truly large datasets, the more you can parallelize, the
better. Parallelization does come with overhead costs, however, and not
every problem actually features huge datasets. In these cases, refrain
from trying to do too much in parallel and focus on slimming down your
network, using CPUs and keeping all your data in RAM if possible.
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch08lvl2sec121}Speeding up your code with Cython {#speeding-up-your-code-with-cython .title}

</div>

</div>
:::

Python is a[]{#id710 .indexterm} popular language because developing
code in Python is easy and fast. However, Python can be slow, which is
why many production applications are written in either C or C++. Cython
is Python with C data types, which significantly speeds up execution.
Using this language, you can write pretty much normal Python code, and
Cython converts it to fast-running C code.

::: {.note style="margin-left: 0.5in; margin-right: 0.5in;"}
### []{#note45}Note {#note-2 .title}

[**Note**]{.strong}: You can read the full Cython[]{#id711 .indexterm}
documentation here:
[http://cython.readthedocs.io](http://cython.readthedocs.io/){.ulink}.
This section is a short introduction to Cython. If performance is
important to your application, you should consider diving deeper.
:::

Say you have a Python function that prints out the Fibonacci series up
to a specified point. This code snippet is taken straight from the
Python documentation:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
from __future__ import print_function
def fib(n):
    a, b = 0, 1
    while b < n:
        print(b, end=' ')
        a, b = b, a + b
    print()
```
:::

Note that we have to import the `print_function`{.literal} to make sure
that `print()`{.literal} works in the Python 3 style. To use this
snippet with Cython, save it as `cython_fib_8_7.pyx`{.literal}.

Now create a new file called `8_7_cython_setup.py`{.literal}:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
from distutils.core import setup                   #1
from Cython.Build import cythonize                 #2

setup(                                             #3ext_modules=cythonize("cython_fib_8_7.pyx"),)
```
:::

The three main[]{#id712 .indexterm} features of the code are these:

::: {.orderedlist}
1.  The `setup`{.literal} function is a Python function to create
    modules, such as the ones you install with `pip`{.literal}.

2.  `cythonize`{.literal} is a function to turn a `pyx`{.literal} Python
    file into Cython C code.

3.  We create a new model by calling `setup`{.literal} and passing on
    our Cythonized code.
:::

To run this, we now run the following command in a Terminal:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
python 8_7_cython_setup.py build_ext --inplace
```
:::

This will create a C file, a build file, and a compiled module. We can
import this module now by running:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
import cython_fib_8_7
cython_fib_8_7.fib(1000)
```
:::

This will print out the Fibonacci numbers up to 1,000. Cython also comes
with a handy debugger that shows where Cython has to fall back onto
Python code, which will slow things down. Type the following command
into your Terminal:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
cython -a cython_fib_8_7.pyx
```
:::

This will create an HTML file that looks similar to this when opened in
a browser:

::: {.mediaobject}
![](5_files/B10354_08_10.jpg)

::: {.caption}
Cython profile
:::
:::

As you can see, Cython has to fall back on Python all the time in our
script because we did not specify the types of variables. By letting
Cython know what data type a variable has, we can speed[]{#id713
.indexterm} up the code significantly. To define a variable with a type,
we use `cdef`{.literal}:

::: {.informalexample}
::: {.toolbar .clearfix}
Copy
:::

``` {.programlisting .language-markup}
from __future__ import print_function
def fib(int n):
    cdef int a = 0
    cdef int b = 1
    while b < n:
        print(b, end=' ')
        a, b = b, a + b
    print()
```
:::

This snippet is already better. Further optimization is certainly
possible, by first calculating the numbers before printing them, we can
reduce the reliance on Python `print`{.literal} statements. Overall,
Cython is a great way to keep the development speed and ease of Python
and gain execution speed.
:::

::: {.section lang="en" lang="en"}
::: {.titlepage}
<div>

<div>

### []{#ch08lvl2sec122}Caching frequent requests {#caching-frequent-requests .title}

</div>

</div>
:::

An under-appreciated way to make models run faster is to cache frequent
requests in a database. You can go so far as to []{#id714
.indexterm}cache millions of predictions in a database and then look
them up. This has the advantage that you can make your model as large as
you like and expend a lot of computing power to make predictions.

By using a MapReduce database, looking up requests in a very large pool
of possible requests and predictions is entirely possible. Of course,
this requires requests to be somewhat discrete. If you have continuous
features, you can[]{#id715 .indexterm} round them if precision is not as
important.



[]{#ch08lvl1sec103}Exercises {#exercises .title style="clear: both"}
----------------------------

</div>

</div>

------------------------------------------------------------------------
:::

Now that we\'re at the end of this chapter, it\'s time to put what
we\'ve learned into use. Using the knowledge that you\'ve gained in this
chapter, why not try the following exercises?

::: {.itemizedlist}
-   Try to build any model that features exploding gradients in
    training. Hint: Do not normalize inputs and play with the
    initialization of layers.

-   Go to any example in this book and try to optimize performance by
    improving the data pipeline.



[]{#ch08lvl1sec104}Summary {#summary .title style="clear: both"}
--------------------------

</div>

</div>

------------------------------------------------------------------------
:::

In this chapter, you have learned a number of practical tips for
debugging and improving your model. Let\'s recap all of the things that
we have looked at:

::: {.itemizedlist}
-   Finding flaws in your data that lead to flaws in your learned model

-   Using creative tricks to make your model learn more from less data

-   Unit testing data in production or training to make sure standards
    are met

-   Being mindful of privacy

-   Preparing data for training and avoiding common pitfalls

-   Inspecting the model and peering into the \"black box\"

-   Finding optimal hyperparameters

-   Scheduling learning rates in order to reduce overfitting

-   Monitoring training progress with TensorBoard

-   Deploying machine learning products and iterating on them

-   Speeding up training and inference
:::

You now have a substantial number of tools in your toolbox that will
help you run actual, practical machine learning projects and deploy them
in real-life (for example, trading) applications.

Making sure your model works before deploying it is crucial and failure
to properly scrutinize your model can cost you, your employer, or your
clients millions of dollars. For these reasons, some firms are reluctant
to deploy machine learning models into trading at all. They fear that
they will never understand the models and thus won\'t be able to manage
them in a production environment. Hopefully, this chapter alleviates
that fear by showcasing some practical tools that can make models
understandable, generalizable, and safe to deploy.

In the next chapter, we will look at a special, persistent, and
dangerous problem associated with machine learning models: bias.
Statistical models tend to fit to and amplify human biases. Financial
institutions have to follow strict regulations to prevent them from
being racially or gender biased. Our focus will be to see how we can
detect and remove biases from our models in order to make them both fair
and compliant.
