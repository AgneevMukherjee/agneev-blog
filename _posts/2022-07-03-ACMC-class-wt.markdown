---
layout: post
title:  "11. ACMC - Class weighting and MCC"
date:   2022-07-03 12:00:00 +0100
category: ['Machine Learning', 'Computer Vision', 'Python', 'Metric']
tag: ['TensorFlow 2', 'Keras', 'Matplotlib', 'Seaborn', 'Pandas', 'Scikit-learn', 'Matthews correlation coefficient']
---

<div class="post-nav">
  <p>
    {% if page.previous.url %}
    <big><b>
    <a href="{{ site.baseurl }}{{page.previous.url}}">&#8672;&nbsp;{{page.previous.title}}</a></b></big>
    {% endif %}
  </p>
  <p style = "text-align:right;">
    {% if page.next.url %}
    <big><b>
    <a href="{{ site.baseurl }}{{page.next.url}}">{{page.next.title}}&nbsp;&#8674;</a>
    </b></big>
    {% endif %}
  </p>
</div>

<br>

<span style="font-family:Helvetica; font-size:1.5em;">Categories:</span>
<div class="post-categories">
<p style="font-size:20px">
  {% if post %}
    {% assign categories = post.categories %}
  {% else %}
    {% assign categories = page.categories %}
  {% endif %}
  {% for category in categories %}
  <a href="{{site.baseurl}}/categories/#{{category|slugize}}">{{category}}</a>
  {% unless forloop.last %}&nbsp;{% endunless %}
  {% endfor %}
  </p>
</div>
<br/>

<span style="font-family:Helvetica; font-size:1.5em;">Tags:</span><br/>
<p style="font-size:18px">
{{page.tag | join: ', ' }}
</p>
<br/>

Hello! [In the previous post](https://agneevmukherjee.github.io/agneev-blog/ACMC-undersampling/), we looked at the pros and cons of using undersampling for dealing with the imbalance inherent in the [Archie Comics Multiclass (ACMC) dataset](https://www.kaggle.com/datasets/agneev/archie-comics-multi-class). The next thing to explore, logically, would be the opposite tactic - oversampling to increase the number of samples in the minority classes. That's _not_ what we are going to do today. As I mentioned last time, things are a bit busy with me at the moment, and as oversampling deserves a fairly detailed look, I will defer that till next time. Instead, we are going to look at another important method of dealing with class imbalance - class weighting.
{: style="text-align: justify"}

## Class weights <a id="wts"></a>

Class weighting as a means of handling class imbalance is as simple as assigning a higher weight to the minority classes, so that the classifier places a greater emphasis on these classes while training - that's it. This makes intuitive sense - if we want our model to do as well on a class having 50 samples as on one having 500, then the classifier better pay a lot more attention to the minority class, right?
{: style="text-align: justify"}

More technically, during model training, the loss is calculated at the end of training each batch, and the model parameters are then updated in an attempt to reduce this loss. Without class weights, every sample in a batch contributes equally to the loss. On the other hand, if class weights are provided, then the contribution of a particular sample to the loss becomes proportional to the class weight. Thus, if a minor class has a weight ten times higher than a majority class, then every minor class sample will contribute ten times more to the loss than a majority class sample. This can make the model a little slower to train, as the training loss declines more slowly than in an unweighted model, but the end product is more even-handed in classifying unbalanced classes.
{: style="text-align: justify"}

Let us jump right into the code. As always, the full code is [available on Github](https://github.com/AgneevMukherjee/agneev-blog/blob/main/tf-acmc-imagenet-class-weights-20.ipynb), and we will look at the important bits here. By now, the procedure for creating and running the model must be quite familiar, and so I will not go over that. The only difference is the insertion of class weights:
{: style="text-align: justify"}

{% highlight python linenos %}

class_weights={}
for i in range (num_classes):
    class_weights[i]=max_samples/samples_per_class[i]
for key, value in class_weights.items():
    print ( key, ' : ', value)
    # Based on https://stackoverflow.com/questions/66501676/how-to-set-class-weights-by-dictionary-for-imbalanced-classes-in-keras-or-tensor

{% endhighlight %}

Step-by-step, the procedure is like this: first, we create an empty dictionary called class_weights (Line 1). We then run a for-loop (Lines 2 and 3) to assign the weights to each class. In Line 3, max_samples refers to a pre-calculated value, the maximum number of samples in any one class. In our case, it happens to be 1284, which is the number of samples in the 'Archie' class. This value is divided by the number of samples in each class to obtain the class weight for that class. As an example, the 'Kleats' class has 41 samples, and so its class weight is 1284 / 41 = 31.3. The 'Jughead' class has a far greater number of samples, and so its weight is 1284 / 962 = 1.3.
{: style="text-align: justify"}

That's essentially all that we need to do to get the class weights. If we want to check if we have assigned the weights correctly, we can print out the dictionary (Lines 4 and 5) to get:
{: style="text-align: justify"}

![Image_1](/agneev-blog/assets/img/img_11_1.png?raw=true){: width="200", height="100" }

The 'key' numbers have been assigned sequentially in the for-loop - note the 'class_weights[i]' in the code above. We can check which number corresponds to which class via the 'inv_index_classes_dict' created earlier in the notebook, which has for its content:
{: style="text-align: justify"}

![Image_2](/agneev-blog/assets/img/img_11_2.png?raw=true){: width="150", height="100" }

<br>
The only thing that now needs to be done is to pass the class_weights dictionary as an argument while fitting the model:
{: style="text-align: justify"}

{% highlight python %}

history = model.fit(
        train,
        validation_data=valid,
        epochs=20,
        callbacks=[stopping, checkpoint],
        class_weight=class_weights
)

{% endhighlight %}

Et voilà! We obtain a validation accuracy of around 48%, which is lower than that obtained without class weights, around 56%. Something more interesting is the change in the shape of the training curves:
{: style="text-align: justify"}

![Image_3](/agneev-blog/assets/img/img_11_3.png?raw=true){: width="750", height="400" }

<p style="color:grey;font-size:100%;text-align: center;">
 Unweighted model curves on top, weighted bottom
</p>

<br>
Looking at the model loss curves, we see that for the unweighted model (top), the training loss starts at around 2.75 and drops to around 0.75 after 20 epochs. The weighted training loss (bottom), however, starts at around 15 and remains around 4 at the end of the run. This reflects in a difference in the accuracy curves as well, with a training accuracy of around 78% obtained for the original model after 20 epochs and only 56% for the weighted model. The validation loss, meanwhile, has a similar pattern for both the models, and so the drop in validation accuracy is less pronounced. All this adds up to a far lower degree of overfitting in the weighted model as compared to the original. In other words, class weighting here is also acting as a regularisation technique!
{: style="text-align: justify"}

As we are not overfitting after 20 epochs and the validation accuracy appears to have room to rise further, I also created a notebook [looking at a run of 50 epochs](https://github.com/AgneevMukherjee/agneev-blog/blob/main/tf-acmc-imagenet-class-weights-50.ipynb). We will see the results at the end of this post.
{: style="text-align: justify"}
.
{: style="text-align: justify"}
.
{: style="text-align: justify"}
So our original intention was to produce a model which would be roughly equally accurate on the majority and minority classes. How does class weighting fare in that respect? We will come to that, but first a detour.
{: style="text-align: justify"}

Last time around, I had [described F-scores](https://agneevmukherjee.github.io/agneev-blog/ACMC-undersampling/#f-score) and how they can be used to quantitatively compare classification models. I may, however, have mistakenly given the impression that F-scores are the only way of doing this, or that they are infallible. Neither is true, and there are in fact plenty of statistical measures that may be used depending on the circumstances. Perhaps one day I will write a blog post going into several of these, but today I will focus on one that is relevant to balanced classification on the ACMC - the Matthews correlation coefficient (MCC).
{: style="text-align: justify"}

## Matthews correlation coefficient <a id="mcc"></a>

The Matthews correlation coefficient (MCC), more generally known as the [phi coefficient](https://en.wikipedia.org/wiki/Phi_coefficient), is calculated for binary classification as:
{: style="text-align: justify"}

![Image_4](/agneev-blog/assets/img/img_11_4.png?raw=true){: width="350", height="200" }

While the  F-score ranges from 0 (completely useless model) to 1 (perfection), the MCC ranges from -1 to 1, with a higher score better, and 0 indicating a random model. We can see that, like the F-score, its calculation includes the true positives (TP), false positives (FP) and false negatives (FN). However, unlike the F-score, we also take into account the true negatives (TN). Therefore, a high MCC score is only obtained if a model does well on all four confusion matrix categories. There has been [some](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7)... [work](https://biodatamining.biomedcentral.com/articles/10.1186/s13040-021-00244-z)... [published](https://clevertap.com/blog/the-best-metric-to-measure-accuracy-of-classification-models/)... suggesting that the MCC is more reliable and robust than the F-score and other metrics for binary classification, especially for unbalanced datasets, while others [contest this](https://www.sciencedirect.com/science/article/abs/pii/S016786552030115X). I am not competent enough to arbitrate on this, but at the very least the MCC should be a significant addition to our toolbox, either in addition to or in place of the F-score.
{: style="text-align: justify"}

While the MCC can be adapted to multiclass classification via micro- and macro-averaging like the F<sub>1</sub> score, it also has a generalised equation that is given [in Wikipedia](https://en.wikipedia.org/wiki/Phi_coefficient#Multiclass_case) (as well as articles like [this](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0041882)). The formula is rather indimidating, and while it is certainly possible to code it from scratch, a better option may be to use Scikit-learn's [in-built method](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html) that can be used for either binary or multiclass classification. The code for this is simplicity itself - we simply import _matthews_corrcoef_ from _sklearn.metrics_ and then pass it the true labels and the predictions:
{: style="text-align: justify"}

![Image_5](/agneev-blog/assets/img/img_11_5.png?raw=true){: width="500", height="300" }

## Results <a id="result"></a>

All right then, let us have a look at the results. The table below compares 3 approaches - using the whole dataset, undersampling, and class weighting - on 7 different criteria. You may want to refresh your memory on the details of the [undersampling-min](https://agneevmukherjee.github.io/agneev-blog/ACMC-undersampling/#min_sample) and [undersampling-200](https://agneevmukherjee.github.io/agneev-blog/ACMC-undersampling/#arb_thresh) approaches, otherwise let's go!
{: style="text-align: justify"}

Before we discuss the results, though, a couple of notes: first, unlike in the previous post, the undersampling-200 approach code looked at here does not use training set augmentations, so as to enable a fairer comparison with the other approaches, none of which use augmentations either. Second, I have bolded the best results for each criterion in the table, but given the stochastic nature of the models, close results may often be flipped, and that is why I have also highlighted values that are close to the best in the table.
{: style="text-align: justify"}

|    Approach    |   Epochs |   RT (s)<sup>a</sup>| Train   acc. (%)  | Val. acc. (%)|   Macro-F<sub>1</sub> |   Micro-F<sub>1</sub> | MCC | Code
|:--|:-:|:-:||:-:|:-:|:-:||:-:||:-:|:--|
| Whole dataset  |    20  |   614  |   78 |  57   |  0.37 |           0.57 |      0.51    |[1](https://github.com/AgneevMukherjee/agneev-blog/blob/main/tf-acmc-simple-imagenet-f-mcc.ipynb)
| Whole dataset  |    50  |   1535  |   **96** |  **60**   |  **0.45** |        **0.6** |      **0.55**  | [2](https://github.com/AgneevMukherjee/agneev-blog/blob/main/tf-acmc-simple-imagenet-50-f-mcc.ipynb)
| Undersampling-min |    50   |   **218**  |   **98** | 31  | 0.3   |        0.31 |    0.28 |[3](https://github.com/AgneevMukherjee/agneev-blog/blob/main/undersampling-acmc-f-mcc.ipynb)
| Undersampling-200 |   50   |  735   |   **97**  | 46    |  0.43 |       0.46 |   0.43 |[4](https://github.com/AgneevMukherjee/agneev-blog/blob/main/undersampling-acmc-200-f-mcc.ipynb)
| Class weighting   |    20   |   618  |   56 | 48  | 0.39   |        0.48 |    0.43 |[5](https://github.com/AgneevMukherjee/agneev-blog/blob/main/tf-acmc-imagenet-class-weights-20.ipynb)
| Class weighting |   50   |  1534   |   83 | **58**    |  **0.47** |     **0.58** |   **0.53** |[6](https://github.com/AgneevMukherjee/agneev-blog/blob/main/tf-acmc-imagenet-class-weights-50.ipynb)

<sup>a</sup>: Approx run time in 2022 on Kaggle Notebooks using GPU

Now, looking at the table, a few things stand out. Firstly, for the whole dataset, overfitting is evident even after 20 epochs, and this only increases further after 50. However, the macroaverage F<sub>1</sub> score does show considerable improvement under further training, far more than the micro-average F<sub>1</sub> score does. Intuitively, this would suggest that the model first learns mostly on the majority classes before turning its attention towards the minority classes in an effort to raise the accuracy further.
{: style="text-align: justify"}

In case of undersampling, using the minimum number of samples per class makes for very fast training due to the much-reduced number of samples. That's all that it has going for it, though - while training accuracy approaches 100%, the other parameters are very bad indeed. The undersampling-200 approach fares much better, although compared with using the whole dataset, again the only area it does better is the shorter run time. The MCC is considerably lower than for the whole dataset, and the macro-F<sub>1</sub> score slightly inferior, which means that our hope of building a more equitable model hasn't really come to pass using undersampling.
{: style="text-align: justify"}

Class weighting, as we saw earlier, narrows the overfitting considerably, which is a plus. The remaining results, however, are merely comparable with using the full dataset, rather than an improvement. The fact that the training accuracy still has upward room may mean that further training will improve the other values, but this would of course be at the cost of increased training time.
{: style="text-align: justify"}

A final comment. We see that the MCC tend to lie between the micro- and macro-F<sub>1</sub> scores, and rise and fall in tandem with them. For this dataset, therefore, it does not appear to provide any novel insights. It would be unfair to draw any conclusions based on this alone, however, and so I will continue using the MCC in conjunction with the F<sub>1</sub> score, and see if they diverge in future studies, and what such divergence could mean.
{: style="text-align: justify"}

## Conclusion <a id="conc"></a>

So far we have tested two approaches for developing a classifier that is similarly accurate on the different classes of the ACMC dataset. Unfortunately, neither undersampling nor class weighting were able to significantly improve upon simply using the original dataset. Perhaps oversampling will do the trick? Or maybe a combination of different options? We shall see...for now, bye!
{: style="text-align: justify"}

<br>
<div class="post-nav">
  <p>
    {% if page.previous.url %}
    <big><b>
    <a href="{{ site.baseurl }}{{page.previous.url}}">&#8672;&nbsp;{{page.previous.title}}</a></b></big>
    {% endif %}
  </p>
  <p style = "text-align:right;">
    {% if page.next.url %}
    <big><b>
    <a href="{{ site.baseurl }}{{page.next.url}}">{{page.next.title}}&nbsp;&#8674;</a>
    </b></big>
    {% endif %}
  </p>
</div>
